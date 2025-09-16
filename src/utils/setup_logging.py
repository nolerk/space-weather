import logging

from datetime import datetime
from pathlib import Path

from src.utils.utils import PROJ_PATH

(PROJ_PATH / 'out' / 'logs').mkdir(parents=True, exist_ok=True)
log_filename = datetime.now().strftime('%Y-%m-%dT%H:%M:%S.log')
logging.basicConfig(filename=PROJ_PATH / 'out' / 'logs' / log_filename, level=logging.INFO)
