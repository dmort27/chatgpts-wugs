import os
import math
import argparse
import numpy as np

def read_pred(fpred):
    id2pred = {}
    with open(fpred) as f:
        for line in f:
            if line[:2] == "H-":
                idx, score, pred = line.split("\t")
                idx = int(idx.strip().split("-")[-1])
                pred = pred.strip().replace("@"," ")
                if idx not in id2pred:
                    id2pred[idx] = pred.strip()
    return id2pred

def read_gold(fgold):
    id2gold = {}
    idx = 0
    with open(fgold) as f:
        for line in f:
            id2gold[idx] = line.strip()
            idx += 1
    return id2gold

def eval_file(fpred, fgold):
    id2pred = read_pred(fpred)
    id2gold = read_gold(fgold)
    guess = 0
    correct = 0
    try:
        assert len(id2pred) == len(id2gold)
    except:
        import pdb; pdb.set_trace()
    for idx, gold in id2gold.items():
        guess += 1
        if gold == id2pred[idx]:
            correct += 1
    acc = round(100*correct/guess, 2)
    return acc, correct, guess

def _get_squared_dif(acclist, avg):
    sum_dif = 0
    for acc in acclist:
        dif = acc - avg
        sum_dif += dif * dif
    return sum_dif

def get_avg_std(acclist):
    avg = sum(acclist)/len(acclist)
    std = math.sqrt(_get_squared_dif(acclist, avg)/3)
    return round(avg, 2), round(std, 2)


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
    langlist        =   ['deu','eng','tam','tur']
    seedlist        =   [str(i) for i in range(0, 10)]
    
    for type in types:
        for lang in langlist:
            datadir         = f'data/{lang}'
            for splittype in ["dev", 'test']:    
                if splittype == "test" and lang =='tam':
                    continue
                acc_list        = []        
                for seed in seedlist:
                    preddir     = f'checkpoints-{type}/{lang}-{seed}/{lang}-predictions/'
                    fgold       = os.path.join(datadir, splittype+"."+lang+".output")
                    fpred       = os.path.join(preddir, splittype+"-checkpoint_best.pt.txt")
                    acc, correct, guess = eval_file(fpred, fgold)
                    acc_list.append(acc)
                print(f'{lang}\t{splittype}\t{type}\t{round(np.mean(acc_list),2)} +/- {round(np.std(acc_list),2)}')
