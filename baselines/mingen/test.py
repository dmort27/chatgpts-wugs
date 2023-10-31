# -*- coding: utf-8 -*-
# Ex. python 02_run_model.py --language eng --learn_rules --score_rules --prune_rules --rate_wugs

import configargparse, pickle, sys
from pathlib import Path
import pandas as pd

import config
from features import *
#from rules import *
#import mingen
#import reliability
#import pruning
#import wug_test


def main():
    LANGUAGE = 'eng2'

    # Import config (as created by 01_prepare_data)
    config_save = pickle.load(
        open(Path('../data') / f'{LANGUAGE}_config.pkl', 'rb'))
    for key, val in config_save.items():
        setattr(config, key, val)

    #segs = ['ɹ', 'l', 'm', 'f', 'p']
    segs = ['g', 's', 'ɹ', 'w', 'b', 'p', 't']
    ftr_vecs = [config.seg2ftrs_[seg] for seg in segs]
    ftr_unif = ftr_vecs[0]
    for ftr_vec in ftr_vecs:
        ftr_unif, _ = unify_ftrs(ftr_unif, ftr_vec)
    print(f'mingen {segs} =>')
    print(ftrs2str1(ftr_unif))
    print(ftrs2regex1(ftr_unif))


#    print(unif)

if __name__ == "__main__":
    main()
