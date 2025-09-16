import pickle
import logging
from catboost import CatBoostClassifier

from src.utils.utils import PROJ_PATH
from src.datasets.data_preparation import ts_fts_train, ts_labels_train

logger = logging.getLogger(__name__)

def train_and_save():
    classifier = CatBoostClassifier()
    
    classifier.fit(ts_fts_train, ts_labels_train)
    
    model_path = PROJ_PATH / 'out' / 'model_checkpoints' / 'catboost.pkl'
    
    with open(model_path, 'wb') as model_file:
        pickle.dump(classifier, model_file)

    logger.info(f"Saved CatBoost to {model_path}")
