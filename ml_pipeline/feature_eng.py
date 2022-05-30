import pandas as pd

from ml_em.util import Utilities, load_config
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

import inspect
import os
import logging
import datetime
import sys

import string

logger = logging.getLogger('ml_em.feature_eng')


class FeatureGroupBase():

    def __init__(self, config):
        self.config = config
        self.utils = Utilities(config)
        try:
            with open(os.path.join(os.path.dirname(inspect.getfile(self.__class__)), 'ddl.sql')) as fp:
                self.ddl = fp.read().format(**self.config)
        except FileNotFoundError:
            logger.warning(f'DDL statement not found for {self.__class__.__name__}')

        logger.info(f"Initialized {self.__class__.__name__}")

    def run(self):
        """
        Wrapper function that performs data loading, preprocessing, and feature calculation for a feature group
        """
        logger.info('Begin run()')

        df_src, df_tgt = self.read_data()

        if df_src.shape[0] == 0 or df_tgt.shape[0] == 0:
            logger.warning('No work found. Exiting!')
            sys.exit(0)

        df_src, df_tgt = self.preprocess(df_src, df_tgt)

        fe_chunk_size = self.config.get('fe_chunk_size')

        tgt_chunk_size = int(fe_chunk_size / df_src.shape[0])
        if tgt_chunk_size:
            logger.info('Processing data in chunks')
            chunk_start = 0
            while chunk_start < df_tgt.shape[0]:
                chunk_end = min(df_tgt.shape[0], chunk_start + tgt_chunk_size)
                logger.debug(f'Processing chunk: {chunk_start} to {chunk_end}')

                df_src_tmp = df_src.copy()  # copy because we do some in-place modifications in prep_for_pairwise()
                df_tgt_tmp = df_tgt.iloc[chunk_start:chunk_end].copy()

                df_pair = self.make_pairwise_df(df_src_tmp, df_tgt_tmp)

                if df_pair.shape[0] == 0:
                    logger.warning("No pairs found, moving on to the next target chunk.")

                else:
                    feature_df = self.compute_features(df_pair)
                    self.write_data(feature_df)

                chunk_start = chunk_end

        else:
            logger.info('No chuck size detected. Attempting to process all pairs')
            df_pair = self.make_pairwise_df(df_src, df_tgt)
            feature_df = self.compute_features(df_pair)
            self.write_data(feature_df)

    def read_data(self):
        # gets the source and target data as pandas df's from either db or local store
        # if blocking_key provided, only gets the source and target data for that blocking key
        logger.info('Begin read_data()')

        blocking_key = self.config.get('blocking_key')

        if blocking_key:
            logger.info(f"Found blocking_key: '{blocking_key}'")

            if self.config['record_read_scheme'] == 'local-csv':
                logger.warning(f"Blocking not implemented for {self.config['record_read_scheme']}")

            blocking_key_str = f"'{blocking_key}'"

        else:
            logger.warning('No blocking key found. Doing full run!')

            blocking_key_str = "'" + "','".join(string.ascii_lowercase) + "'"

        df_src = self.utils.read_data(self.config['record_read_scheme'], self.config['src_record_type'], blocking_key_str=blocking_key_str)
        df_src.drop_duplicates(['record_type', 'record_id'], inplace=True)
        df_src.reset_index(drop=True, inplace=True)

        logger.debug(f"Loaded {df_src.shape[0]} rows from source {self.config['src_record_type']}")

        df_tgt = self.utils.read_data(self.config['record_read_scheme'], self.config['tgt_record_type'], blocking_key_str=blocking_key_str)
        df_tgt.drop_duplicates(['record_type', 'record_id'], inplace=True)
        df_tgt.reset_index(drop=True, inplace=True)

        logger.debug(f"Loaded {df_tgt.shape[0]} rows from source {self.config['tgt_record_type']}")

        return df_src, df_tgt

    def write_data(self, feature_df):
        """
        Takes the calculated feature df, adds requisite fields / metadata and writes out
        to athena table according to the class name write_data() is being called from (FgLegacy -> fg_legacy)


        """

        logger.info('Begin write_data()')

        # unwind index
        feature_df.reset_index(inplace=True)
        feature_df[["source_type", "source_id"]] = feature_df.apply(
            lambda row: row.pk_src.split("_"),
            axis=1,
            result_type="expand"
        )
        feature_df.drop("pk_src", axis=1, inplace=True)
        feature_df['source_id'] = feature_df['source_id'].astype(int)

        feature_df[["target_type", "target_id"]] = feature_df.apply(
            lambda row: row.pk_tgt.split("_"),
            axis=1,
            result_type="expand"
        )
        feature_df.drop("pk_tgt", axis=1, inplace=True)
        feature_df['target_id'] = feature_df['target_id'].astype(int)

        feature_df['run_id'] = self.config['run_id']
        feature_df['process_id'] = self.config['process_id']
        feature_df['created_at'] = datetime.datetime.utcnow()

        output_name = self.utils.camel_to_snake(self.__class__.__name__)
        self.utils.write_data(self.config['feature_write_scheme'], output_name, feature_df, load_partitions=True)

    def prep_for_pairwise(self, df, suffix):
        """
        This function adds a primary key column to a dataframe, and append a suffix to all columns names

        :param df : dataframe to which we want to add a primary key (either a source or target df)
        :param suffix : the table from which the data originates
        """

        df['pk'] = df['record_type'] + '_' + df['record_id'].map(str)
        columns = [c + '_' + suffix for c in df.columns]
        df.columns = columns
        df.set_index('pk_' + suffix, inplace=True)

        # set df to self for later reference
        setattr(self, 'df_' + suffix, df)

    def make_pairwise_df(self, df_src, df_tgt):
        """
        Creates the pairwise df for any source and target df. Also:
         * renames columns such that any duplicate column names across source and target "{col_name}_src" and "{col_name}_tgt"
         * creates primary key at the pair level "{pk_src}_{pk_tgt}"

        :param df_src : source dataframe with primary key created
        :param df_tgt : target dataframe with primary key created

        :return df_pair : pairwise df for each source/target doc + pair level primary key
        """
        logger.info('Begin make_pairwise_df()')

        # src and tgt will be indexed and saved to self for efficient lookups
        self.prep_for_pairwise(df_src, 'src')
        self.prep_for_pairwise(df_tgt, 'tgt')

        df_pair = pd.DataFrame(index=pd.MultiIndex.from_product([self.df_src.index, self.df_tgt.index], names=['pk_src', 'pk_tgt']))

        logger.debug(f'df_pair.shape: {df_pair.shape}')
        return df_pair

    def run_backfill(self, dependent_variable=None):

        if dependent_variable is None:
            if 'dependent_variable' not in self.config:
                raise Exception('Need a dependent_variable declared in the config to begin backfill')

            dependent_variable = self.config['dependent_variable']

        df_src, df_tgt, df_pair = self.get_backfill_data(dependent_variable)
        df_src, df_tgt = self.preprocess(df_src, df_tgt)

        # prep our src and tgt lookups
        self.prep_for_pairwise(df_src, 'src')
        self.prep_for_pairwise(df_tgt, 'tgt')

        feature_df = self.compute_features(df_pair)
        self.write_data(feature_df)

    def get_backfill_data(self, dependent_variable):

        feature_group = self.utils.camel_to_snake(self.__class__.__name__)

        df_src = self.utils.read_data(self.config['record_read_scheme'], 'backfill_src', dependent_variable=dependent_variable, feature_group=feature_group)
        df_src.drop_duplicates(['record_type', 'record_id'], inplace=True)
        df_src.reset_index(drop=True, inplace=True)

        df_tgt = self.utils.read_data(self.config['record_read_scheme'], 'backfill_tgt', dependent_variable=dependent_variable, feature_group=feature_group)
        df_tgt.drop_duplicates(['record_type', 'record_id'], inplace=True)
        df_tgt.reset_index(drop=True, inplace=True)

        df_pair = self.utils.read_data(self.config['record_read_scheme'], 'backfill_pairs', dependent_variable=dependent_variable, feature_group=feature_group)
        df_pair['pk_src'] = df_pair['source_type'] + '_' + df_pair['source_id'].astype(str)
        df_pair['pk_tgt'] = df_pair['target_type'] + '_' + df_pair['target_id'].astype(str)
        df_pair.set_index(['pk_src', 'pk_tgt'], inplace=True)
        df_pair.drop(['source_type', 'source_id', 'target_type', 'target_id'], axis=1, inplace=True)
        df_pair = df_pair.iloc[~df_pair.index.duplicated()]

        return df_src, df_tgt, df_pair

    def prep_txt_data(self, txt):
        """
        Helper function to remove leading spaces and lower case a string

        returns prepared text
        """
        txt = txt.lower().strip()

        return txt

    def vectorize_one_col(self, pd_series, vec_type, *args, **kwargs):
        vec_type = vec_type.lower()
        data_to_vec = pd_series.tolist()

        if vec_type == "tfidf":
            vectorized = TfidfVectorizer(min_df=kwargs["min_df"],
                                         analyzer=kwargs["analyzer"])
        elif vec_type == "count":
            # vectorized = CountVectorizer() #latest changes as per rajeev & srivani discussion on 4-10-2020
            vectorized = CountVectorizer(min_df=kwargs["min_df"],
                                         analyzer=kwargs["analyzer"])  # latest changes as per rajeev & srivani & Avinash discussion on 5-21-2020
        vectorized_data = vectorized.fit_transform(data_to_vec)

        return vectorized_data

    def preprocess(self, df_src, df_tgt):
        """
        Preprocess function that will be overridden in the feature groups child class.
        """
        logging.info(f'No preprocessing implemented for {self.__class__.__name__}')

        return df_src, df_tgt

    def compute_features(self, pairs_df, *args, **kwargs):
        """
        compute_features function that will be overridden in the feature groups child class.
        """
        raise Exception("compute_features() not implimented on FeatureGroupBase")


if __name__ == '__main__':
    config = load_config()

    fe = FeatureGroupBase(config)

    fe.run()