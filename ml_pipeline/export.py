from ml_em.util import Utilities
import datetime
import logging
import re
import json
import pandas as pd
from collections import OrderedDict
import sys

logger = logging.getLogger('ml_em.model.export')

# This should be somewhere in tess, but I can't find it. Just hardcoding it for now
PROJECT_ID_LOOKUP = {
    'gpo': 300,
    'pm': 251
}

TESS_BATCH_STATUS_NEW = 1
MASTER_RECORD_TYPES = ['pm']

DEFAULT_MIN_MATCH_SCORE = 0.2
DEFAULT_AUTO_QA_APPROVE_THRESHOLD = 0.9

# this should be in the tess db itself see https://h1insights.atlassian.net/browse/DSML-36
BLOCK_NO_MATCH_COLLECTION_SOURCE_IDS = {17, 42, 44, 66, 108, 120, 122}


class Exporter():

    # we might want to export slightly different columns that we read (eg full name)
    raw_col_to_export = ['first_name', 'middle_name', 'last_name', 'designation', 'full_name_oth', 'address', 'city', 'state', 'country', 'city_oth', 'state_oth', 'country_oth', 'npi', 'orcid', 'specialty', 'title', 'role_name', 'email', 'specialty_oth', 'organization_name', 'organization_name_oth', 'social_urls', 'person_listing_url', 'other_ids', 'is_npi_deactivated']

    def __init__(self, config):
        self.config = config
        self.utils = Utilities(config)

    def run(self):
        logger.debug('Begin run()')

        self.get_export_info()
        match_df, no_match_df = self.get_data()
        if match_df.shape[0] == 0:
            logger.warning('No data found to export. Aborting')
            sys.exit(0)

        output_dict = self.transform(match_df, no_match_df)
        self.write_data(output_dict)

        logger.info(f'Finished run for {self.__class__.__name__}.')

    def get_export_info(self):
        logger.debug('Begin get_export_info()')

        scheme = self.config['feature_read_scheme']

        target_query_lookup = {
            'local-csv': 'run_model_inference_j_pre_train_model',
            'feature-store': """
                SELECT run.post_train_model_id, model.*
                FROM {athena_db}.run_model_inference run
                JOIN {athena_db}.pre_train_model model ON run.pre_train_model_id = model.pre_train_model_id
                WHERE run.run_id = '{run_id}'
                  AND run.is_primary = true
            """
        }

        t_q = target_query_lookup[scheme]

        export_info_df = self.utils.read_data(scheme, t_q)

        if scheme == 'local-csv':
            i_row = (export_info_df.run_id == self.config['run_id']) & (export_info_df.is_primary)
            export_info_df = export_info_df[i_row].reset_index(drop=True)

        if export_info_df.shape[0] == 0:
            raise Exception(f"No primary models found for run: {self.config['run_id']}")

        if export_info_df.shape[0] > 1:
            raise Exception(f"Too many primary models found for run: {self.config['run_id']}")

        export_info = export_info_df.loc[0]
        export_info['input_features'] = export_info['input_features'].split('|')

        logger.info(f"Found export target:\n{export_info}")

        self.pre_train_model_id = export_info['pre_train_model_id']
        self.post_train_model_id = export_info['post_train_model_id']
        self.feature_cols = export_info['input_features']

        # eventually, load this from db here
        self.block_no_match_collection_source_ids = BLOCK_NO_MATCH_COLLECTION_SOURCE_IDS

        return export_info

    def get_data(self):
        logger.debug('Begin get_data()')

        if self.config['feature_read_scheme'] == 'local-csv':
            data_type = 'for_export'
        else:
            data_type = self.build_query()

        export_df = self.utils.read_data(self.config['feature_read_scheme'], data_type)

        i_bad_rows = export_df['collection_source_id_src'] == 0
        if i_bad_rows.any():
            bad_ids = list(export_df[i_bad_rows]['source_id'].unique())
            logger.warning(f'Found record with missing collection source for PersonIDs: {bad_ids}')
            export_df = export_df[~i_bad_rows]

        i_match = export_df['ds_combined_score'] >= DEFAULT_MIN_MATCH_SCORE
        match_df = export_df[i_match].reset_index(drop=True)
        no_match_df = export_df[~i_match].reset_index(drop=True)

        logger.info(f'Found {match_df.shape[0]} match proposals and {no_match_df.shape[0]} no match proposals')

        if match_df.shape[0] == 0:
            return match_df, no_match_df

        # process cols. names should come in first for column ordering in UI
        match_df.insert(0, 'full_name_src', match_df[['first_name_src', 'middle_name_src', 'last_name_src']].apply(lambda row: row.str.cat(sep=' '), axis=1))
        match_df.insert(0, 'full_name_tgt', match_df[['first_name_tgt', 'middle_name_tgt', 'last_name_tgt']].apply(lambda row: row.str.cat(sep=' '), axis=1))

        match_df.drop(
            axis='columns',
            inplace=True,
            labels=['first_name_src', 'middle_name_src', 'last_name_src', 'first_name_tgt', 'middle_name_tgt', 'last_name_tgt']
        )

        return match_df, no_match_df

    def build_query(self):
        logger.debug('Begin build_query()')

        features_dict = {}

        for feature in self.feature_cols:
            feature_group, feature_name = feature.split('.')
            features = features_dict.setdefault(feature_group, [])
            features.append(feature_name)

        db = self.config['athena_db']

        src_cols_str = ','.join([f'{col} {col}_src' for col in self.raw_col_to_export])
        tgt_cols_str = ','.join([f'{col} {col}_tgt' for col in self.raw_col_to_export])

        # this doesn't follow good python indent rules to better format query
        cte_c = f"""
WITH
all_src AS (
  SELECT DISTINCT record_id source_id
  FROM {db}.block_record
  WHERE run_id = '{self.config['run_id']}'
    AND record_type = '{self.config['src_record_type']}'
),
scores AS (
  SELECT source_type, source_id, target_type, target_id, score
  FROM (
    SELECT *, row_number() OVER (PARTITION BY run_id, source_id ORDER BY score DESC) row_idx
    FROM {db}.record_pair_match_score s
    WHERE run_id = '{self.config['run_id']}'
      AND post_train_model_id = '{self.post_train_model_id}'
      AND s.source_type = '{self.config['src_record_type']}'
      AND s.target_type = '{self.config['tgt_record_type']}'
  )
  -- for no match, we want the highest score match if there are any, even below our minimum match threshold
  WHERE (row_idx <= 5 AND score >= {DEFAULT_MIN_MATCH_SCORE}) -- TODO: set as config
     OR row_idx = 1
)
, src AS (
  SELECT
    record_id source_id,
    collection_source_id collection_source_id_src,
    {src_cols_str}
  FROM (
    SELECT *,
      row_number() OVER (PARTITION BY r.record_type, r.record_id ORDER BY r.tess_modified_at DESC) as row_idx
    FROM {db}.record r
    WHERE record_type = '{self.config['src_record_type']}'
  )
  WHERE row_idx = 1
)
, tgt AS (
  SELECT
    record_id target_id,
    {tgt_cols_str}
  FROM (
    SELECT *,
      row_number() OVER (PARTITION BY r.record_type, r.record_id ORDER BY r.tess_modified_at DESC) as row_idx
    FROM {db}.record r
    WHERE record_type = '{self.config['tgt_record_type']}'
  )
  WHERE row_idx = 1
)
"""

        select_c = """
SELECT
    s.score ds_combined_score
  , src.*
  , tgt.*
"""

        # for no-matches, we need all the src side we have records for, then any tgt data for proposed matches
        from_c = """
FROM all_src
LEFT JOIN scores s ON all_src.source_id = s.source_id
JOIN src ON src.source_id = all_src.source_id
LEFT JOIN tgt ON tgt.target_id = s.target_id
"""

        for group, features in features_dict.items():
            # add each feature group as a join. actually easier not to alias it
            cte_c += f"""
, {group} AS (
  SELECT *
  FROM (
    SELECT *,
      row_number() OVER (PARTITION BY fg.source_type, fg.source_id, fg.target_type, fg.target_id ORDER BY fg.created_at DESC) as row_idx
    FROM {db}.{group} fg
    WHERE run_id = '{self.config['run_id']}'
  )
  WHERE row_idx = 1
)
"""

            from_c += f"""
LEFT JOIN {group}
  ON {group}.source_type = s.source_type
  AND {group}.source_id = s.source_id
  AND {group}.target_type = s.target_type
  AND {group}.target_id = s.target_id"""

            # add each feature to the select clause with expliecit naming using '_'
            for feature in features:
                select_c += f"  , {group}.{feature} {self.utils.feature_col_name(group, feature)}\n"

        query = cte_c + select_c + from_c

        logger.debug(f"Query to get export data:{query}")

        return query

    def split_data(self, match_df):
        logger.debug('Begin split_data()')

        collection_sources = match_df['collection_source_id_src'].unique()

        output_dict = {}
        for collection_source_id in collection_sources:
            i = match_df.collection_source_id_src == collection_source_id
            output_dict[collection_source_id] = {
                'em.MatchingResultDetail': match_df[i].copy()
            }

        return output_dict

    def transform(self, match_df, no_match_df):
        logger.debug('Begin transform()')

        # self.get_next_ids()

        batch_config_df = self.get_batch_config()
        column_schema, default_batch_column_configuration, default_match_column_remap, no_match_column_remap = self.build_schema(match_df)
        column_schema_json = json.dumps(column_schema)
        output_dict = self.split_data(match_df)

        for collection_source_id, data in output_dict.items():

            config_info = batch_config_df.loc[collection_source_id]

            if pd.isnull(config_info.BatchId) or pd.isnull(config_info.ColumnSchema) or json.loads(config_info.ColumnSchema) != column_schema:
                logger.info(f"No batch found for '{config_info.collectionSourceName}'. Creating batch.")

                # use our auto-incrementing counters for ids
                # data['_batch_id'] = next(self._BatchId)
                # data['_config_id'] = next(self._ConfigurationId)
                # data['_matching_result_id'] = next(self._MatchingResultId)

                logger.debug("Creating em.BatchColumnConfiguration")
                data['em.BatchColumnConfiguration'] = default_batch_column_configuration.copy()

                logger.debug("Creating em.Configuration")
                data['em.Configuration'] = pd.DataFrame({
                    # 'ConfigurationId': [id_dict['ConfigurationId']], # Identity keys can't do explicit inserts, only auto-incrementing
                    'ConfigurationName': [f'CollectionSourceId {collection_source_id} - {datetime.date.today()}'],
                    'SourceProjectId': [PROJECT_ID_LOOKUP[self.config['src_record_type']]],
                    'TargetProjectId': [PROJECT_ID_LOOKUP[self.config['tgt_record_type']]],
                    'ColumnSchema': [column_schema_json],
                })

                logger.debug("Creating em.Batch")
                data['em.Batch'] = pd.DataFrame({
                    # 'BatchId': [id_dict['BatchID']], # Identity keys can't do explicit inserts, only auto-incrementing
                    # 'ConfigurationId': [data['_config_id']],
                    'CreatedBy': [self.config.get('tess_user', 'EMUser')],
                    'CreatedDate': [datetime.datetime.utcnow()],
                    'Description': [f"EM - {config_info.collectionSourceName}"],
                    'Status': [TESS_BATCH_STATUS_NEW],
                    'IsPremastertoMaster': [(self.config['src_record_type'] not in MASTER_RECORD_TYPES) & (self.config['tgt_record_type'] in MASTER_RECORD_TYPES)],
                })

                logger.debug("Creating em.BatchCollectionSource")
                data['em.BatchCollectionSource'] = pd.DataFrame({
                    # 'BatchId': [data['_batch_id']],
                    'CollectionSourceId': [collection_source_id],
                    'CreatedBy': [self.config.get('tess_user', 'EMUser')],
                    'CreatedDate': [datetime.datetime.utcnow()],
                    'IsActive': [1],
                })

                logger.debug("Creating em.MatchingResult")
                data['em.MatchingResult'] = pd.DataFrame({
                    # 'MatchingResultId': [data['_matching_result_id']], # Identity keys can't do explicit inserts, only auto-incrementing
                    # 'BatchId': [data['_batch_id']],
                    'RuleName': ['people_generic_rule'],
                    'CreatedBy': [self.config.get('tess_user', 'EMUser')],
                    'CreatedDate': [datetime.datetime.utcnow()],
                    'IsSuccessFul': [1],
                    'SourceIndexName': ['XXXXXXXX-XXXXXXX'],
                    'TargetIndexName': ['finalmaster']
                })

                match_column_remap = default_match_column_remap

            else:
                logger.info(f'Found existing batch for {config_info.collectionSourceName}')

                data['_config_id'] = int(config_info.ConfigurationId)
                data['_batch_id'] = int(config_info.BatchId)
                data['_matching_result_id'] = int(config_info.MatchingResultId)
                data['_append_mode'] = 1

                _, match_column_remap = self.get_batch_column_configuration(data['_batch_id'])

            # final transforms and column additions

            # matchcount handled by SP
            # matchcount_lookup = data['em.MatchingResultDetail']['source_id'].value_counts()
            # data['em.MatchingResultDetail']['matchcount'] = data['em.MatchingResultDetail']['source_id'].apply(lambda x: matchcount_lookup[x])

            data['em.MatchingResultDetail']['CreatedBy'] = self.config.get('tess_user', 'EMUser')
            data['em.MatchingResultDetail']['CreatedDate'] = datetime.datetime.utcnow()

            # rename columns appropriately and move all "fieldvalue" columns to strings
            data['em.MatchingResultDetail'].rename(match_column_remap, axis='columns', inplace=True)
            for col in data['em.MatchingResultDetail'].columns:
                if col.startswith('fieldvalue'):
                    data['em.MatchingResultDetail'][col] = data['em.MatchingResultDetail'][col].astype('string').str.slice(0, 1000)

        i_no_match = ~no_match_df.collection_source_id_src.isin(self.block_no_match_collection_source_ids)
        no_match_df = no_match_df[i_no_match]
        no_match_df = no_match_df[no_match_column_remap.keys()].reset_index(drop=True)

        no_match_df.rename(no_match_column_remap, axis='columns', inplace=True)

        output_dict['gpov1.DSNonMatches'] = no_match_df

        return output_dict

    def write_data(self, output_dict):
        logger.debug('Begin write_data()')

        scheme = self.config['score_export_write_scheme']

        for collection_source_id, data in output_dict.items():

            if collection_source_id == 'gpov1.DSNonMatches':
                continue

            logger.debug(f"Starting to write data for collection_source_id: {collection_source_id}")

            if 'em.Configuration' in data:
                df = data['em.Configuration'].copy()
                self.utils.write_data(scheme, 'em.Configuration', df, if_exists='append')

                if scheme != 'local-csv':
                    result = self.utils.read_data(scheme, 'SELECT max(ConfigurationId) ConfigurationId FROM em.Configuration')
                    config_id = result.loc[0, 'ConfigurationId']
                else:
                    config_id = 0

            else:
                config_id = data['_config_id']

            # Write em.Batch
            if 'em.Batch' in data:
                df = data['em.Batch'].copy()
                df['ConfigurationId'] = config_id
                self.utils.write_data(scheme, 'em.Batch', df, if_exists='append')

                if scheme != 'local-csv':
                    result = self.utils.read_data(scheme, 'SELECT max(BatchId) BatchId FROM em.Batch')
                    batch_id = result.loc[0, 'BatchId']
                else:
                    batch_id = 0
            else:
                batch_id = data['_batch_id']

            # Write em.MatchingResult
            if 'em.MatchingResult' in data:
                df = data['em.MatchingResult'].copy()
                df['BatchId'] = batch_id
                self.utils.write_data(scheme, 'em.MatchingResult', df, if_exists='append')

                if scheme != 'local-csv':
                    result = self.utils.read_data(scheme, 'SELECT max(MatchingResultId) MatchingResultId FROM em.MatchingResult')
                    matching_result_id = result.loc[0, 'MatchingResultId']
                else:
                    matching_result_id = 0
            else:
                matching_result_id = data['_matching_result_id']

            # Write em.BatchColumnConfiguration
            if 'em.BatchColumnConfiguration' in data:
                df = data['em.BatchColumnConfiguration'].copy()
                df['BatchId'] = batch_id
                self.utils.write_data(scheme, 'em.BatchColumnConfiguration', df, if_exists='append')

            # Write em.BatchColumnConfiguration
            if 'em.BatchCollectionSource' in data:
                df = data['em.BatchCollectionSource'].copy()
                df['BatchId'] = batch_id
                self.utils.write_data(scheme, 'em.BatchCollectionSource', df, if_exists='append')

            df = data['em.MatchingResultDetail'].copy()
            df['MatchingResultId'] = matching_result_id
            self.utils.write_data(scheme, 'em.MatchingResultDetail_NoIndex', df, if_exists='append')

            if scheme == 'onboarding-db':
                append_mode = data.get('_append_mode', 0)
                if append_mode == 1:
                    logger.info(f'Appending to batch {batch_id}')
                else:
                    logger.info(f'Creating new batch {batch_id}')

                # sqlalchemy isn't handeling transactions well with MS SQL SPs
                # so, we need to explicitly open a transaction with .begin()
                # which will be commited in the cleanup from `with`
                batch_prep_cmd = f"exec dbo.proc_dataSetPostRulesExecutionProcess {batch_id}, {append_mode}, 'EMUser'"
                with self.utils.tess_conn.begin():
                    self.utils.tess_conn.execute(batch_prep_cmd)

                    # approval has 2 stages: "qa_approved" and then "approved" that both involve manual review
                    # we will automate the first step
                    if self.config.get('auto_qa_approve', False):
                        auto_qa_approve_threshold = self.config.get('auto_qa_approve_threshold', DEFAULT_AUTO_QA_APPROVE_THRESHOLD)
                        logger.info(f'Automating "QA Approval" for all records with score >= {auto_qa_approve_threshold}')

                        score_column_q = f"""
                            SELECT MappingColumnName
                            FROM EM.BatchColumnConfiguration bcc
                            WHERE bcc.BatchID = {batch_id}
                              AND PrimaryColumnName = 'ds_combined_score'
                        """
                        score_column_name = self.utils.read_data(scheme, score_column_q).loc[0, 'MappingColumnName']

                        auto_qa_approve_q = f"""
                            INSERT INTO gpov1.ApproveRejectData
                                (SourceID, TargetID, CurrentStageID, PreviousStageID, UndoStageID, BatchID, Comments, CreatedBy, ActionType)
                            SELECT
                                mrd.sourceid,
                                mrd.targetid,
                                2 CurrentStageID, -- enum: 1=Pending, 2=ApprovedQAPending, 3=Approved, 4=Rejected
                                1 PreviousStageID,
                                1 UndoStageID,
                                {batch_id} BatchID,
                                'EM - Auto QA Approved' Comments,
                                'EMUser' CreatedBy,
                                'ApprovedQAPending' ActionType
                            FROM (
                                SELECT *, row_number() OVER (PARTITION BY sourceid ORDER BY {score_column_name} DESC) row_num
                                FROM em.MatchingResultDetail_{batch_id} mrd
                                WHERE mrd.{score_column_name} >= {auto_qa_approve_threshold}
                            ) mrd
                            LEFT JOIN gpov1.ApproveRejectData ard
                                ON ard.SourceID = mrd.sourceid AND ard.TargetID = mrd.targetid
                            WHERE mrd.row_num = 1
                                AND ard.ApproveRejectDataID IS NULL
                        """
                        # not performing auto_qa_approval until we have a better plan
                        self.utils.tess_conn.execute(auto_qa_approve_q)

        if 'gpov1.DSNonMatches' in output_dict and output_dict['gpov1.DSNonMatches'].shape[0]:

            df = output_dict['gpov1.DSNonMatches'].copy()
            self.utils.write_data(scheme, "gpov1.DSNonMatches", df, if_exists='append')

            if scheme == 'onboarding-db':

                logger.info('Running No Match SP')
                no_match_sp_cmd = 'gpov1.uspInsertGPOv1EMAdditionalFlags'
                with self.utils.tess_conn.begin():
                    self.utils.tess_conn.execute(no_match_sp_cmd)

    def get_batch_config(self):
        """One giant race condition. Output starts to get dangerous the ms it is returned."""
        logger.debug('Begin get_batch_config()')

        # even though we will read data here, we want to use the export_write_scheme because we need to know what to write
        scheme = self.config['score_export_write_scheme']
        collection_source_db = self.config.get('tess_ob_db', 'Tesseract')

        query_lookup = {
            'local-csv': 'batch_config',
            'onboarding-db': f"""
                WITH
                latest_batches AS (
                    SELECT *
                    FROM (
                        SELECT *, row_number() OVER (PARTITION BY CollectionSourceId ORDER BY BatchId DESC) row_num
                        FROM em.BatchCollectionSource
                    ) bcs
                    WHERE bcs.row_num = 1
                ),
                active_batches AS (
                  SELECT
                    bcs.CollectionSourceId,
                    b.BatchId,
                    mr.MatchingResultId,
                    c.*
                  FROM latest_batches bcs
                  JOIN em.Batch b ON bcs.BatchId = b.BatchId
                  JOIN em.MatchingResult mr ON b.BatchId = mr.BatchId
                  JOIN em.Configuration c ON b.ConfigurationId = c.ConfigurationId
                  WHERE bcs.IsActive = 1
                    AND b.status = 1
                    AND mr.RuleName = 'people_generic_rule'
                )
                SELECT
                  ls.LookupID CollectionSourceId,
                  ls.LookupValue collectionSourceName,
                  ab.BatchId,
                  ab.MatchingResultId,
                  ab.ConfigurationId,
                  ab.ConfigurationName,
                  ab.SourceProjectId,
                  ab.TargetProjectId,
                  ab.ColumnSchema
                FROM {collection_source_db}.dbo.LookupSource ls
                LEFT JOIN active_batches ab ON ab.CollectionSourceId = ls.LookupID
                WHERE ls.IsActive = 1
                """
        }

        batch_config_df = self.utils.read_data(scheme, query_lookup[scheme])

        batch_config_df.set_index('CollectionSourceId', inplace=True)

        return batch_config_df

    def build_schema(self, match_df):
        logger.debug('Begin build_schema()')

        output_schema = {
            "schema": {
                "source": OrderedDict([
                    ("sourceid", {
                        "title": "Source Id",
                        "type": "number"
                    }),
                    ("Match", {
                        "title": "Match",
                        "type": "string"
                    }),
                    ("TargetCount", {
                        "title": "TargetCount",
                        "type": "number"
                    })
                ]),
                "target": OrderedDict([(
                    "targetid", {
                        "title": "Target Id",
                        "type": "number"
                    },
                )]),
                "tags": {
                    "tag": {
                        "title": "tag",
                        "type": "string"
                    }
                },
                "action type": {
                    "action": {
                        "title": "ActionType",
                        "type": "string"
                    }
                },
                "action": {
                    "Approve": {
                        "title": "Approve",
                        "type": "boolean"
                    },
                    "Reject": {
                        "title": "Reject",
                        "type": "boolean"
                    }
                }
            }
        }

        # the batch column configuration determines filtering capabilities
        # need to start with a special source and target id columns that will
        # both have order "0"
        batch_column_configuration = [
            {
                'RuleName': 'people_generic_rule',
                'PrimaryColumnName': 'sourceid',
                'MappingColumnName': 'sourceid',
                'FieldOrder': 0,
                'CreatedDate': datetime.datetime.utcnow(),
                'CreatedBy': self.config.get('tess_user', 'EMUser'),
                'Datatype': 'bigint'
            },
            {
                'RuleName': 'people_generic_rule',
                'PrimaryColumnName': 'targetid',
                'MappingColumnName': 'targetid',
                'FieldOrder': 0,
                'CreatedDate': datetime.datetime.utcnow(),
                'CreatedBy': self.config.get('tess_user', 'EMUser'),
                'Datatype': 'bigint'
            },
        ]

        match_column_remap = {
            'source_id': 'sourceid',
            'target_id': 'targetid',
        }
        field_number = 1

        for col in match_df.columns:
            if col in ['source_type', 'target_type', 'source_id', 'target_id']:
                continue

            # strip off feature group names with regex
            col_mod = re.sub(r'^fg_[a-z]*_', '', col)

            col_mod = col_mod.replace('_', ' ').replace(' f ', ' first ').replace(' l ', ' last ')

            if col_mod.endswith(' src'):
                col_mod = 'source ' + col_mod.replace(' src', '')
                sub_schema = output_schema["schema"]["source"]

            elif col_mod.endswith(' tgt'):
                col_mod = 'target ' + col_mod.replace(' tgt', '')
                sub_schema = output_schema["schema"]["target"]

            else:
                sub_schema = output_schema["schema"]["target"]

            dtype_lookup = {
                'float': 'number',
                'int': 'number',
                'object': 'string',
                'bool': 'string'
            }

            col_type = None
            for key, val in dtype_lookup.items():
                if match_df[col].dtype.name.lower().startswith(key):
                    col_type = val
                    break

            if col_type is None:
                logger.warning(f'Unknown dtype of {match_df[col].dtype.name} for column {col}. Defaulting to string')
                col_type = 'string'

            sub_schema[col] = {
                "title": col_mod.title(),
                "type": col_type,
            }

            # now we set up the em.BatchColumnConfiguration
            column_config_dtype_lookup = {
                'float': 'float',
                'int': 'bigint',
                'object': 'nvarchar(max)',
            }

            col_type = None
            for key, val in column_config_dtype_lookup.items():
                if match_df[col].dtype.name.lower().startswith(key):
                    col_type = val
                    break

            batch_column_configuration.append({
                'RuleName': 'people_generic_rule',
                'PrimaryColumnName': col,
                'MappingColumnName': f"fieldvalue{field_number}",
                'FieldOrder': field_number,
                'CreatedDate': datetime.datetime.utcnow(),
                'CreatedBy': self.config.get('tess_user', 'EMUser'),
                'Datatype': col_type
            })

            match_column_remap[col] = f"fieldvalue{field_number}"

            field_number += 1

        batch_column_configuration = pd.DataFrame(batch_column_configuration)

        # Start no match schema setup. Yet a different inconsistent schema
        no_match_column_remap = {}

        no_match_tgt_columns = ['first_name', 'middle_name', 'last_name', 'npi', 'specialty', 'organization_name', 'other_ids']
        for col in no_match_tgt_columns:
            no_match_column_remap[col + '_tgt'] = 'Target' + col.title().replace('_', '')

        no_match_feature_columns = ['fg_legacy_country_match', 'fg_legacy_npi_match', 'fg_legacy_organization_name_match', 'fg_legacy_specialty_match']
        for col in no_match_feature_columns:
            new_col = re.sub(r'^fg_[a-z]*_', '', col)
            no_match_column_remap[col] = 'Target' + new_col.title().replace('_', '')

        # custom overwrites when even the above doesn't quite cut it
        no_match_column_remap.update({
            'source_id': 'SourceID',
            'target_id': 'TargetID',
            'specialty_tgt': 'TargetSpeciality',
            'fg_legacy_name_match_score': 'DSNameMatchScore',
            'fg_legacy_f_name_freq_src': 'FirstNameFrequency',
            'fg_legacy_l_name_freq_src': 'LastNameFrequency',
            'fg_legacy_first_name_match': 'FirstNameMatch',
            'fg_legacy_middle_name_match': 'MiddleNameMatch',
            'fg_legacy_last_name_match': 'LastNameMatch',
            'fg_legacy_other_ids_match': 'OtherIdsMatch',
            'fg_legacy_specialty_match': 'TargetSpecialityMatch',
        })

        return output_schema, batch_column_configuration, match_column_remap, no_match_column_remap

    def get_batch_column_configuration(self, batch_id):
        logger.debug('Begin get_batch_column_configuration()')

        scheme = self.config['score_export_write_scheme']

        q_lookup = {
            'local-csv': 'em.BatchColumnConfiguration',
            'onboarding-db': f"SELECT * FROM em.BatchColumnConfiguration WHERE BatchID = {batch_id}"
        }

        batch_column_configuration = self.utils.read_data(scheme, q_lookup[scheme])

        if scheme == 'local-csv':
            i = batch_column_configuration['BatchID'] == batch_id
            batch_column_configuration = batch_column_configuration[i]

        match_column_remap = {
            'source_id': 'sourceid',
            'target_id': 'targetid',
        }
        for row in batch_column_configuration.itertuples(index=False):
            match_column_remap[row.PrimaryColumnName] = row.MappingColumnName

        return batch_column_configuration, match_column_remap


if __name__ == '__main__':
    from h1_ml.util import load_config

    config = load_config()
    ex = Exporter(config)

    ex.run()
