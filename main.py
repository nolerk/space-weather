import src.utils.setup_logging

from src.models import lstm
from src.models import rnn
from src.models import knc
from src.models import dt
from src.models import svc
from src.models import mlp
from src.models import lr
from src.models import catboost

from src.utils.utils import PROJ_PATH

(PROJ_PATH / 'out' / 'model_checkpoints').mkdir(parents=True, exist_ok=True)

def main():
    lstm.train_and_save()
    rnn.train_and_save()
    knc.train_and_save()
    dt.train_and_save()
    svc.train_and_save()
    mlp.train_and_save()
    lr.train_and_save()
    catboost.train_and_save()
    

if __name__ == "__main__":
    main()
