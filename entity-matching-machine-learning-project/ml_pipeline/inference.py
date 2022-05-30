from ml_em.model import ModelBase

import pandas as pd

import logging
import string
import datetime
from pprint import pformat

import sys

logger = logging.getLogger('ml_em.models.inference')


class ModelInferer(ModelBase):

    def run(self):
        logger.info("Starting model inference run.")

        run_models_df = self.get_run_models()
        i_model = run_models_df.pre_train_model_id == self.pre_train_model_id
        run_models_df = run_models_df[i_model].reset_index()

        if run_models_df.shape[0] == 0:
            raise Exception(f"No pre_train_model_id '{self.pre_train_model_id}' found for run_id '{self.config['run_id']}'")

        if run_models_df.shape[0] > 1:
            raise Exception(f"Too many trained models found for pre_train_model_id '{self.pre_train_model_id}' and run_id '{self.config['run_id']}'")

        self.post_train_model_id = run_models_df.loc[0, 'post_train_model_id']

        properties_to_print = ['model_path', 'feature_cols', 'pre_train_model_id', 'dependent_variable', 'post_train_model_id']
        prop_dict = {key: getattr(self, key) for key in properties_to_print}
        logger.info(f"Preparing inference for model: \n{pformat(prop_dict)}")

        trained_model = self.load_trained_model()
        feature_df = self.load_data()

        if feature_df.shape[0] == 0:
            logger.warning('No work found. Exiting!')
            sys.exit(0)

        p_df = self.make_inferrence(trained_model, feature_df)
        self.write_data(p_df)

    def load_data(self):
        logger.debug('Beging load_data()')

        if self.config['feature_read_scheme'] == 'local-csv':
            data_type = 'inference'
        else:
            data_type = self.build_query()

        logger.info("Loading data.")
        feature_df = self.utils.read_data(self.config['feature_read_scheme'], data_type)

        if feature_df.shape[0] == 0:
            logger.warning(f"No data found for feature_read_scheme='{self.config['feature_read_scheme']}' and run_id='{self.config['run_id']}'. No work to do.")
            sys.exit(0)

        # safety in case we have invalid or duplicate rows
        feature_df = feature_df[~feature_df.isnull().any(axis=1)]
        feature_df.drop_duplicates(['source_type', 'source_id', 'target_type', 'target_id'], inplace=True)

        logger.debug(f"Loaded features with shape: {feature_df.shape}")

        return feature_df

    def make_inferrence(self, trained_model, feature_df):
        logger.debug('Beging make_inferrence()')

        feature_df.set_index(['source_type', 'source_id', 'target_type', 'target_id'], inplace=True)

        feature_names = [self.utils.feature_col_name(*f.split('.')) for f in self.feature_cols]

        # score is always to class True
        i = list(trained_model.classes_).index(True)

        p_df = trained_model.predict_proba(feature_df[feature_names])

        p_df = pd.DataFrame(p_df[:, i], columns=['score'], index=feature_df.index)

        logger.info("Scores computed.")

        return p_df

    def write_data(self, p_df):
        logger.debug('Beging write_data()')

        p_df = p_df.reset_index()

        p_df['run_id'] = self.config['run_id']
        p_df['created_at'] = datetime.datetime.now()
        p_df['pre_train_model_id'] = self.pre_train_model_id
        p_df['post_train_model_id'] = self.post_train_model_id

        self.utils.write_data(self.config['feature_write_scheme'], 'record_pair_match_score', p_df, load_partitions=True)

    def build_query(self):
        # TODO: need to better handle duplicates
        logger.debug('Beging build_query()')

        if self.config['blocking_key']:
            logger.info(f"Found blocking_key: '{self.config['blocking_key']}'")

            blocking_key_str = f"'{self.config['blocking_key']}'"

        else:
            logger.warning('No blocking key found. Doing full run!')

            blocking_key_str = "'" + "','".join(string.ascii_lowercase) + "'"

        features_dict = {}

        for feature in self.feature_cols:
            feature_group, feature_name = feature.split('.')
            features = features_dict.setdefault(feature_group, [])
            features.append(feature_name)

        db = self.config['athena_db']

        # doesn't follow good python indent rules to better format query
        select_c = """
SELECT
    p.source_type
  , p.source_id
  , p.target_type
  , p.target_id
"""

        from_c = f"FROM {db}.run_pairs p"

        where_c = f"\nWHERE p.block IN ({blocking_key_str}) AND p.run_id = '{self.config['run_id']}'"

        for group, features in features_dict.items():
            # add each feature group as a join. actually easier not to alias it
            from_c += f"""
LEFT JOIN {db}.{group}
  ON {group}.source_type = p.source_type
  AND {group}.source_id = p.source_id
  AND {group}.target_type = p.target_type
  AND {group}.target_id = p.target_id"""

            where_c += f"\n  AND {group}.run_id = '{self.config['run_id']}'"

            # add each feature to the select clause with expliecit naming using '_'
            for feature in features:
                select_c += f"  , {group}.{feature} {self.utils.feature_col_name(group, feature)}\n"

        query = select_c + from_c + where_c

        logger.debug(f"Query to get training data:{query}")

        return query


if __name__ == '__main__':

    from h1_ml.util import load_config

    config = load_config()

    mi = ModelInferer(config)
    mi.run()
