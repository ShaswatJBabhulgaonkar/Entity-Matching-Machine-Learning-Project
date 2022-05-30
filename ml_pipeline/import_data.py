"""
This module imports the net new records from various different sources
"""
import logging
import datetime
import re
import pandas as pd
import unidecode
from ml_em.util import Utilities, load_config
from ml_em.data.hudi.person import Person
from ml_em.data.hudi.person_block import PersonBlock
from ml_em.data.hudi.feedback import Feedback

import pyspark.sql.functions as F
import pyspark.sql.types as T

logger = logging.getLogger('ml_em.import_data')


class Importer():
    """
    Class that does the net new data importing for record and feedback tables
    """

    def __init__(self, config):
        """
            :param config: configuration object passed across all modules to define necessary values across the EM run
        """
        self.config = config
        self.utils = Utilities(config)

        logger.debug(f'Initialized {self.__class__.__name__}')

        # Create all requisite hudi objecrts

        # What type of record type is being imported (right now only Person but could grow to orgs)
        # TODO: point to utils
        person_type_hudi_lookup = {
            "gpo": Person,
            "npi": Person,
            "pm": Person,
            "op": Person,
        }
        self.HudiRecordConnectorType = person_type_hudi_lookup[self.config["import_record_type"]](self.config)

    @staticmethod
    @F.udf(returnType=T.StringType())
    def first_letter_if_exists(name_val):
        """
        Pyspark udf used to determine the null first letter of a given first, middle or last name
        MUST HAVE staticmethod DECORATOR SO SPARK CAN REGISTER UDF FROM WITHIN CLASS
        """
        # for situations where null value is 'null'
        name_val = name_val.replace("null", "")

        if pd.isnull(name_val):
            return None

        name_val = unidecode.unidecode(name_val).lower()
        name_val = re.sub('[^a-z -]', '', name_val)

        if name_val == '':
            return None

        return name_val[0]

    def run(self):
        """
        Function that runs relevant functions for record and feedback import
        """
        logger.info('Begin run()')
        import_record_type = self.config["import_record_type"]

        # import net new records according to import_record_type config value
        logger.info(f"import_record_type : {import_record_type}")
        self.import_records(record_type=import_record_type)

        # import net new feedback
        # todo how to handle when import_record_type = 'pm' -- no dedupe feedback yet
        if import_record_type == 'pm':
            return

        self.import_feedback(src_record_type=import_record_type, tgt_record_type="pm")

        logger.info('Finished import successfully.')

    def import_records(self, record_type):
        """
        Function that :
            - imports the net new records given a schema and outputs to the record table in athena
            - assigns each net new record to a blocking key then outputs to the block_record table in athena

        :param record_type: schema to import net new records for
        """
        logger.info('Begin import_records()')

        logger.info(f'Importing {record_type}. Determining import range.')

        rec_start, rec_end = self.last_imported_record(record_type)
        logger.info(f'rec_start to rec_end date range : {rec_start} - {rec_end}')

        # schemas that need temp table data prep beforehand
        data_prep_import_lookup = {
            'gpo': 'tess_gpo_prep.sql',
            'npi': 'tess_npi_prep.sql',
            'op': 'tess_op_prep.sql',
        }

        # prepare the temp tables in tess for each schema
        if record_type in data_prep_import_lookup:
            data_prep_file = data_prep_import_lookup[record_type]

            # prep temp tables if needed
            logger.info(f'Starting import from onboarding-db using {data_prep_file} to prep')
            if record_type in data_prep_import_lookup:
                with open(f"{self.config['local_file_root']}/input/{data_prep_file}", "r") as fp:
                    prep_q = fp.read()

                    logger.info(f'Building temporary tables for {record_type}.')
                    # logger.info(prep_q.format(start_date=rec_start,end_date=rec_end,tess_ob_db=self.config["tess_ob_db"]))
                    self.utils.tess_conn.execute(
                        prep_q.format(
                            start_date=rec_start,
                            end_date=rec_end,
                            tess_ob_db=self.config["tess_ob_db"]
                        )
                    )
        # get prepped data for this schema's batch
        import_df = self.utils.read_data('onboarding-db', record_type, rec_start=rec_start, rec_end=rec_end)

        # check if there are any new records
        if import_df.shape[0] == 0:
            logger.warning(f"No new records found for {record_type}")
            return

        # add metadata
        import_df['import_process_id'] = self.config['process_id']
        import_df['imported_at'] = datetime.datetime.now()

        # needed because people master does not have tess_modified_at (precombine field)
        # TODO handle this in a better way for slowly changin dimension issue
        if record_type == "pm":
            import_df['tess_modified_at'] = import_df['imported_at']

        # write to person table (hudi)
        # convert pandas df to spark df
        # TODO better handling of columns where everything is Null. Right now we do replace w tring but what if numeric column has all null values?
        # import_df = import_df.head(500)  # TODO delete after final testing
        import_df = import_df.fillna("null")
        import_spark_df = self.utils.spark.createDataFrame(import_df)

        # handle spark column casting
        required_types = {
            'publication_count': 'int',
            'clinical_trial_count': 'int',
            'collection_source_id': 'int',
            'record_id': 'string',
        }
        import_spark_df = self.utils.cast_spark_df_cols(spark_df=import_spark_df, mapping_dict=required_types)

        # Write out to hudi
        self.HudiRecordConnectorType.upsert_data(import_spark_df, write_method='append')

        logger.info(f'Imported {import_df.shape[0]} rows. Starting to compute_blocks().')

        # assign imported records to blocks
        blocks_df = self.compute_blocks(import_spark_df, record_type)

        # add metadata to blocks_df
        blocks_df = (
            blocks_df.withColumn("run_id", F.lit(self.config['run_id']))
                     .withColumn("created_at", F.lit(datetime.datetime.now()))
        )

        # write out to s3
        # TODO paramterize for when we expand out of person schemas -- hard coding for now
        block_person_hudi_conn = PersonBlock(self.config)

        logger.info(f"writing out block_person table to {block_person_hudi_conn._s3_path}")
        block_person_hudi_conn.upsert_data(blocks_df, write_method="append")
        logger.info(f" DONE writing out block_person table to {block_person_hudi_conn._s3_path}")

    def last_imported_record(self, record_type):
        """
        Gets the index of the most recently imported data present in athena based on the record_type passed

        :param record_type: schema for which net new records are imported
        """
        logger.debug('Beging last_imported_record()')

        import_key_lookup = {
            'pm': 'record_id',  # finalmaster doesn't contain ModifiedDate
            'gpo': 'tess_modified_at',
            'npi': 'tess_modified_at',
            'op': 'tess_modified_at',
        }

        import_key = import_key_lookup[record_type]

        # get the most recenntly imported data from hudi
        last_imported = (
            self.HudiRecordConnectorType.get_most_recent_by_record_type(
                col_to_get=import_key,
                record_type=record_type
            )
        )

        if import_key == 'tess_modified_at':
            # TODO how to handle when a record_type doesn't have any records imported
            import_until = last_imported + datetime.timedelta(14)
            last_imported = last_imported.strftime('%Y-%m-%d %H:%M:%S:%f')[:-3]
            import_until = import_until.strftime('%Y-%m-%d %H:%M:%S:%f')[:-3]
        else:
            if not last_imported:
                # if a given record_type has no imported records
                # TODO validate if there is a better way to do this
                last_imported = 1
            import_until = last_imported + 500000

        return last_imported, import_until

    def compute_blocks(self, spark_df, record_type):
        # calculate the number of rows upon which blocking calculations need to be done
        init_block_nrows = spark_df.count()

        # determine the columns that we block on for a given record type
        cols_lookup = {
            'pm': ['first_name'],
            'gpo': ['first_name', 'middle_name', 'last_name'],
            'npi': ['first_name'],
            'op': ['first_name'],
        }

        blocking_cols = cols_lookup[record_type]
        logger.info(f"Determine blocking cols for {record_type} : {blocking_cols}")

        # declare the primary key columns
        id_cols = ["record_type", "record_id"]

        # primary key + blocking column list of cols - used to keep only relevant columns
        all_cols = id_cols + blocking_cols

        # apply first_letter_if_exists() to only the blocking_cols
        block_df = (
            spark_df.select(
                *[self.first_letter_if_exists(F.col(col_name)).name(
                    col_name + "_block") if col_name in blocking_cols else col_name for col_name in all_cols]
            )
        )
        logger.info(f"Done applying first letter blocking udf for {record_type}")

        # create useful subsets of columns
        block_col_str = [col_name for col_name in block_df.columns if
                         "_block" in col_name.lower()]  # calc'd block col names as strings
        block_col_obj = [F.col(col_name) for col_name in block_col_str]  # calc'd block col names as column objects
        cols_to_fetch = ['record_id', 'record_type'] + [
            "block_arr"]  # block_arr is array of all block assignments for givent rec_type/rec_id

        # create a new column - array of all block assignments for givent rec_type/rec_id
        tmp_block_df = block_df.withColumn("block_arr", F.array(block_col_obj)).select(*cols_to_fetch)

        # use block_arr to create a skinny df similar to pandas stack function
        final_block_explode_df = tmp_block_df.selectExpr("record_id", "record_type", "explode(block_arr) as block")

        # remove any record_type / record_id's that have a null value for block
        final_block_df = final_block_explode_df.filter(F.col("block").isNotNull())
        final_block_df_nrows = final_block_df.count()

        logger.debug(f'{final_block_df_nrows} blocks computed for {init_block_nrows} rows of {record_type} data')

        return final_block_df

    def import_feedback(self, src_record_type, tgt_record_type="pm"):
        """
            This function gets feedback for a given source record type. Currently it assumes that the target record type
            is always 'pm'
        """
        logger.info('Beging import_feedback()')

        # determine the date range to fetch for feedback from tess -- acount for null fetched_start_date by making start_date 1/1/2021
        feedback_hudi_conn = Feedback(self.config)
        fetched_start_date = feedback_hudi_conn.get_most_recent_feedback(input_type=src_record_type,
                                                                         target_type=tgt_record_type)
        logger.info(f"Fetched start date from get_most_recent_feedback() : {fetched_start_date}")
        start_date = datetime.datetime(2021, 1, 1) if pd.isnull(fetched_start_date) else fetched_start_date

        end_date = start_date + datetime.timedelta(30)
        start_date_str = start_date.strftime('%Y-%m-%d %H:%M:%S:%f')[:-3]
        end_date_str = end_date.strftime('%Y-%m-%d %H:%M:%S:%f')[:-3]

        logger.info(f'Loading feedback from {start_date_str} to {end_date_str}')

        # get the target side projectid - only need target side as source side project id's hardcoded in .sql files
        project_id_lookup = {
            'pm': 251,
            'gpo': 300,
            'npi': 268,
            'op': 277
        }
        tgt_project_id = project_id_lookup[tgt_record_type]

        # get data from tess
        logger.info('Started import from onboarding db')
        import_df = (
            self.utils.read_data(
                'onboarding-db',
                f'feedback_{src_record_type}',
                start_date=start_date_str,
                end_date=end_date_str,
                tess_em_db=self.config["tess_em_db"],
                tgt_project_id=tgt_project_id
            )
        )

        # check if any new feedback occurred
        if import_df.shape[0] == 0:
            logger.warning('No new records found for feedback.')
            return

        # string -> boolean dtype change
        import_df['is_match'] = import_df['is_match'].astype('boolean')

        # add metadata
        import_df['import_process_id'] = self.config['process_id']
        import_df['imported_at'] = datetime.datetime.now()
        import_df['input_type'] = src_record_type
        import_df['target_type'] = tgt_record_type

        # convert to spark df for write out
        # TODO better way to handle null values in pandas df
        import_df = import_df.fillna("null")
        import_spark_df = self.utils.spark.createDataFrame(import_df)
        import_spark_df_nrows = import_spark_df.count()
        logger.info(f'Imported {import_spark_df_nrows} rows.')

        logger.info(f"writing out feedback table to {feedback_hudi_conn._s3_path}")
        feedback_hudi_conn.upsert_data(import_spark_df, write_method="append")  # todo change to append after final testing
        logger.info(f" DONE writing out feedbac table to {feedback_hudi_conn._s3_path}")


if __name__ == '__main__':
    # config = load_config()
    config = load_config()

    importer = Importer(config)
    importer.run()

