from ml_em.model import ModelBase
import joblib
from importlib import import_module
import datetime

import logging

from imblearn.under_sampling import RandomUnderSampler

logger = logging.getLogger('ml_em.models.training')


class ModelTrainer(ModelBase):

    def __init__(self, config):
        super().__init__(config)

        if self.pre_train_model_id is None:
            raise Exception('Invalid ModelTrainer. pre_train_model_id not found.')

        sample_method = self.config.get("training_sample_method")
        if sample_method:
            sampler_lookup = {
                "RandomUnderSampler": RandomUnderSampler(random_state=42),
            }
            if sample_method not in sampler_lookup:
                raise Exception(f"Unknown sampling method: {sample_method}")

            self.sampler = sampler_lookup[sample_method]
        else:
            self.sampler = None

    def run(self):
        """
        Wrapper function that calls load_data, train and save_model methods
        """
        logger.debug('Beging run()')
        training_df, dependent_df = self.load_data()

        trained_model, metadata = self.train(training_df, dependent_df)

        self.save_model(trained_model, metadata)

        logger.info("Finished run")

    def load_data(self, dropna=None):
        # params = target_col
        # return pairs_df, target_df
        # Finds pairs with appropriate target col
        # Loads in features and target val
        # maybe (src_table, tar_table, target_col) if we train models separately per src/tar pair

        # for now just assume that the is_match field is already merged on to feature df
        # TODO point to non-manufactured data
        logger.debug('Beging load_data()')
        if self.config['feature_read_scheme'] == 'local-csv':
            data_type = 'training'
        else:
            data_type = self.build_query()

        pair_df = self.utils.read_data(self.config['feature_read_scheme'], data_type)
        pair_df.set_index(self.pk_cols, inplace=True)

        # drop any rows that don't have complete data
        if dropna is None:
            dropna = self.config.get('training_dropna', True)

        if dropna:
            logger.debug(f"Dropping rows with NAs. Starting with pair_df.shape: {pair_df.shape}")
            pair_df.dropna(inplace=True)

        # drop any duplicate rows
        pair_df = pair_df[~pair_df.index.duplicated()]

        logger.info(f"Loaded training data with shape: {pair_df.shape}")

        # get the target column as its own df
        dependent_df = pair_df[self.dependent_variable]

        feature_names = [self.utils.feature_col_name(*f.split('.')) for f in self.feature_cols]
        training_df = pair_df[feature_names]

        return training_df, dependent_df

    def train(self, training_df, target_df):
        # params : model_id, pairs_df, target_df -- why do we need model_id as a param?
        # returns : trained_model_id, trained_model where trained_model_id is the posttrain_model_id
        logger.debug('Beging train()')

        logger.info("Beging training with pretrain model id : {}".format(self.pre_train_model_id))

        metadata = {
            'pre_train_model_id': self.pre_train_model_id,
            'model_code_hash': self.model_code_hash,
            'input_features': self.feature_cols,
            'dependent_variable': self.dependent_variable,
            'process_id': self.config['process_id'],
            'created_at': datetime.datetime.utcnow()
        }

        # TRAIN CLASSIFIER
        model = import_module(self.model_path)
        model_pipe = model.pipeline
        logger.debug("model_pipe value : {}".format(model_pipe))

        X = training_df.values
        y = target_df.values

        # train / test split
        if self.sampler:
            logger.info("Performing sampling to balance classes")

            X, y = self.sampler.fit_resample(X, y)

        logger.info(f"Training on {X.shape[0]} samples.")

        # train the model
        trained_model = model_pipe.fit(X, y)

        # as we aren't using this for model seleciton, in_sample score is simple overestimate of future performance
        in_sample_score = trained_model.score(X, y)

        post_train_model_id = joblib.hash(trained_model)

        metadata['post_train_model_id'] = post_train_model_id

        logger.info(f"Done training model. post_training_model_id = {post_train_model_id}, in_sample_score = {in_sample_score}")

        return trained_model, metadata

    def save_model(self, trained_model, metadata):
        """
        Saves the trained_model and associated metadata
        """
        logger.debug('Beging save_model()')

        file_path = f"models/{metadata['pre_train_model_id']}/{metadata['post_train_model_id']}.joblib"
        self.utils.write_obj(self.config['write_obj_scheme'], trained_model, file_path)

        metadata_write_lookup = {
            'local-fs': self._write_metadata_local,
            'object-store': self._write_metadata_athena
        }

        metadata_write_lookup[self.config['write_obj_scheme']](metadata)

    def _write_metadata_local(self, metadata):
        logger.debug('Beging _write_metadata_local()')
        file_path = "models/metadata.jsonl"

        # need to turn our datetime obj into a string
        metadata['created_at'] = metadata['created_at'].isoformat()

        self.utils.write_obj('local-fs', metadata, file_path, append=True)

    def _write_metadata_athena(self, metadata):
        logger.debug('Beging _write_metadata_athena()')

        # the model meta_data stuff is so small, we don't need partitions and
        # can just overwite the file each time
        pre_df = self.utils.read_data('feature-store', 'pre_train_model')
        pre_df
        if metadata['pre_train_model_id'] not in pre_df['pre_train_model_id'].values:
            pre_df = pre_df.append({
                'pre_train_model_id': metadata['pre_train_model_id'],
                'dependent_variable': metadata['dependent_variable'],
                'input_features': '|'.join(metadata['input_features']),  # pyathena cannot handle lists -> array
                'model_code_hash': metadata['model_code_hash'],
            }, ignore_index=True)

            logger.debug(f'Writing new pre_train_model row:\n{pre_df.iloc[-1]}')
            self.utils.write_data('feature-store', 'pre_train_model', pre_df, if_exists="replace")

        else:
            logger.debug('Pre trained model info already present.')

        post_df = self.utils.read_data('feature-store', 'post_train_model')
        post_df = post_df.append({
            'pre_train_model_id': metadata['pre_train_model_id'],
            'post_train_model_id': metadata['post_train_model_id'],
            'process_id': metadata['process_id'],
            'created_at': metadata['created_at'],
            'run_id': self.config['run_id']
        }, ignore_index=True)
        logger.debug(f'Writing new post_train_model row:\n{post_df.iloc[-1]}')
        self.utils.write_data('feature-store', 'post_train_model', post_df, if_exists="replace")

    def build_query(self):
        logger.debug('Beging build_query()')

        features_dict = {}

        for feature in self.feature_cols:
            feature_group, feature_name = feature.split('.')
            features = features_dict.setdefault(feature_group, [])
            features.append(feature_name)

        db = self.config['athena_db']

        # doesn't follow good python indent rules to better format query
        select_c = f"""
SELECT
    fb.source_type
  , fb.source_id
  , fb.target_type
  , fb.target_id
  , fb.{self.dependent_variable}
"""

        # add in row number partition to feedback to make sure we don't get dupes
        from_c = f"""
FROM (
  SELECT
      f.source_type
    , f.source_id
    , f.target_type
    , f.target_id
    , f.{self.dependent_variable}
    , row_number() OVER (PARTITION BY source_type, source_id, target_type, target_id ORDER BY tess_modified_at DESC) rn
  FROM {db}.feedback f
) fb"""

        where_c = f"\nWHERE fb.rn = 1 and fb.{self.dependent_variable} IS NOT NULL"

        for group, features in features_dict.items():
            # add each feature group as a join. actually easier not to alias it
            from_c += f"""
JOIN {db}.{group}
  ON {group}.source_type = fb.source_type
  AND {group}.source_id = fb.source_id
  AND {group}.target_type = fb.target_type
  AND {group}.target_id = fb.target_id"""

            # add each feature to the select clause with expliecit naming using '_'
            for feature in features:
                select_c += f"  , {group}.{feature} {self.utils.feature_col_name(group, feature)}\n"

        query = select_c + from_c + where_c

        logger.debug(f"Query to get training data:{query}")

        return query


if __name__ == '__main__':
    from ml_em.util import load_config

    config = load_config()

    mt = ModelTrainer(config)
    mt.run()