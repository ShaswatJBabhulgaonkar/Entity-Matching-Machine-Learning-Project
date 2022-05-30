from ml_em.util import load_config
from ml_em.import_data import Importer
from ml_em.feature_groups.fg_legacy import FgLegacy
from ml_em.model_training import ModelTrainer

import logging

logger = logging.getLogger('ml_em.run')


if __name__ == '__main__':
    config = load_config(
        model_path="ml_em.model.models.legacy_random_forest",
        strat_shuffle=True,
        feature_cols=['fg_legacy.name_match_score', 'fg_legacy.speciality_match'],
        dependent_variable='is_match'
    )

    importer = Importer(config)
    importer.run()  # local importer not implemented yet, uncomment will hit from tess (not that hard)

    fe = FgLegacy(config)
    fe.run()

    train = ModelTrainer(config)
    train.run()
