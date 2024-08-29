from utils import get_mwz_dataset
import logging
import os
import time
log_file = os.path.join("log", 'log_gen_dialog_testing.txt')
logging.basicConfig(filename=log_file, filemode='a', 
                    format='%(asctime)s %(message)s', 
                    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logging.Formatter.converter = time.gmtime
logger = logging.getLogger(__name__)
dataset = get_mwz_dataset(True,10,logger,None,False)
print(dataset["train"]["slots"][1])
# print(f"utt:{dataset['utt'][:4]}\nslots:{dataset['slots'][:4]}\nbs:{dataset['bs'][:4]}")