import os
import math
import argparse
import numpy as np
from collections import defaultdict as ddict
import pandas as pd
from pprint import pprint
from tqdm import tqdm

def read_pred(fpred):
    id2pred = {}
    id2src  = {}
    with open(fpred) as f:
        for line in f:
            if line[:2] == "H-":
                idx, score, pred = line.split("\t")
                idx = int(idx.strip().split("-")[-1])
                pred = pred.strip().replace(" ","").replace("@"," ")
                if idx not in id2pred:
                    id2pred[int(idx)] = pred.strip()

    return id2pred


def read_gold(fgold):
    id2gold = {}
    id2tag  = {}
    with open(fgold) as f:
        for idx, line in enumerate(f):
            vals = line.strip().split("\t")
            id2gold[int(idx)] = vals[0].strip()
            id2tag[int(idx)]  = vals[-1].strip().upper()

    return id2gold, id2tag

if __name__ == "__main__":
    # create a parser object

    parser = argparse.ArgumentParser(description='Evaluate the predictions of a model')
    parser.add_argument('--datadir', type=str, help='Path to the directory where the ground truth is stored in fairseq format')
    parser.add_argument('--preddir', type=str, help='Path to the directory where the predictions are stored')
    parser.add_argument('--lang',    type=str,   help='List of languages to evaluate')
    parser.add_argument('--splittype', type=str, help='Type of split to evaluate (dev or test)')
    # parse the arguments from standard input
    args   = parser.parse_args()

    types           =   ['1src']
    langlist        =   ['tur']
    seedlist        =   [str(i) for i in range(0, 10)]
    split           =   'nonce'

    # tags            =   ['VERB;POS;PAST;A1SG', 'NOUN;A3SG;PNON;ACC', 'VERB;NEG;NARR;A2PL', 'NOUN;A3SG;P1SG;DAT']
    tag_analysis    =   ddict(lambda: ddict(lambda: ddict(lambda: ddict(set))))
    
    for type in types:
        for lang in langlist:
            pred_dict               = ddict(list)
            fgold                   = f'/projects/wuggpt/principal_parts_for_inflection/org_data/{lang}.{split}'
            id2gold, id2tag         = read_gold(fgold)

            for seed in tqdm(seedlist):
                preddir     = f'/projects/wuggpt/principal_parts_for_inflection/checkpoints-{type}/{lang}-{seed}/{lang}-predictions/'
                fpred       = os.path.join(preddir,f"{split}-checkpoint_best.pt.txt")    
                id2pred     = read_pred(fpred)

                for id in sorted(id2gold.keys()):
                    pred_dict[id].append(id2pred[id])                    
                    tag_analysis[type][lang][id2tag[id]][id].add(id2pred[id]) 
                
            pred_df = pd.DataFrame(pred_dict)
            pred_df.to_csv(f'/projects/wuggpt/principal_parts_for_inflection/nonce_results/{split}-{type}-{lang}.csv', index=False)

                
    for type in types:
        for lang in langlist:
            for tag in tag_analysis[type][lang]:
                len_tag = []
                for ids in tag_analysis[type][lang][tag]:
                    L = len(tag_analysis[type][lang][tag][ids])
                    len_tag.append(L)
                    
                print(f'{type} {lang} {tag} {len(len_tag)} {np.mean(len_tag)}')
            print()

    