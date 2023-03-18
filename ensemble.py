"""
Ensemble
"""

# Other dependencies
from transformers import AlbertTokenizer
import json
import utils as utils
import pickle
import csv

# Own dependencies
from utils import SQuAD
# from util import collate_fn, SQuAD

d = 'test'
with open('./data/dev_gold_dict.json') as f:
    gold_dict = json.load(f)

dev_dataset = SQuAD('./save/saves/cached_{}_eval_xxlarge'.format(d),bidaf = False)
tokenizer = AlbertTokenizer.from_pretrained("albert-xxlarge-v2")
dataset, examples, features = dev_dataset.dataset, dev_dataset.examples, dev_dataset.features
with open('save/saves/xxlarge_224_{}.pickle'.format(d), 'rb') as f:
    all_results1 = pickle.load(f)
with open('save/saves/xxlarge_777_{}.pickle'.format(d), 'rb') as f:
    all_results2 = pickle.load(f)
with open('save/saves/xxlarge_321_{}.pickle'.format(d), 'rb') as f:
    all_results3 = pickle.load(f)
with open('save/saves/xxlarge_123_{}.pickle'.format(d), 'rb') as f:
    all_results4 = pickle.load(f)
    
all_results = [all_results1, all_results2, all_results3, all_results4]

pred_dict = utils.compute_predictions_logits_ensemble(
    examples,
    features,
    all_results,
    5,
    15,
    True,
    'save/temp/predict_temp.json',
    'save/temp/nbest_temp.json',
    'save/temp/nlog_odd.log',
    False,
    True,
    0,
    tokenizer,
)
if d == 'dev':
    print(utils.eval_dicts(gold_dict, pred_dict, True))
with open('save/saves/{}_submission.csv'.format(d), 'w', newline='', encoding='utf-8') as csv_fh:
    csv_writer = csv.writer(csv_fh, delimiter=',')
    csv_writer.writerow(['Id', 'Predicted'])
    for uuid in sorted(pred_dict):
        csv_writer.writerow([uuid, pred_dict[uuid]])