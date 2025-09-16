import pickle
import logging
from sklearn.neural_network import MLPClassifier

from src.utils.utils import PROJ_PATH
from src.datasets.data_preparation import ts_fts_train, ts_labels_train

logger = logging.getLogger(__name__)

def train_and_save():
    classifier = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000) 
    
    classifier.fit(ts_fts_train, ts_labels_train)
    
    model_path = PROJ_PATH / 'out' / 'model_checkpoints' / 'mlp.pkl'
    
    with open(model_path, 'wb') as model_file:
        pickle.dump(classifier, model_file)

    logger.info(f"Saved MLP to {model_path}")
