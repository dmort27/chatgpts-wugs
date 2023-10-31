# -*- coding: utf-8 -*-

import re, string, sys
from pathlib import Path
import pandas as pd
import numpy as np
from collections import namedtuple
#from unicodedata import normalize
from . import phon_config
# todo: delegate to panphon or phoible if possible
# todo: warn about missing/nan feature values in matrix


class FeatureMatrix():

    def __init__(self,
                 symbols,
                 vowels,
                 features,
                 ftr_matrix,
                 ftr_matrix_vec=None):
        self.symbols = symbols  # Special symbols and segments
        self.vowels = vowels  # Symbols that are vowels
        self.features = features  # Feature names
        self.ftr_matrix = ftr_matrix  # Feature matrix {'+', '-', '0'}
        self.ftr_matrix_vec = ftr_matrix_vec  # Feature matrix {+1.0, -1.0, 0.0}

        # Symbol <-> idx
        self.sym2idx = {}
        self.idx2sym = {}
        for idx, sym in enumerate(self.symbols):
            self.sym2idx[sym] = idx
            self.idx2sym[sym] = sym

        # Symbol -> feature-value dict and vector
        self.sym2ftrs = {}
        self.sym2ftr_vec = {}
        for i, sym in enumerate(self.symbols):
            ftrs = ftr_matrix.iloc[i, :].to_dict()
            self.sym2ftrs[sym] = ftrs
            self.sym2ftr_vec[sym] = tuple(ftrs.values())


def import_features(feature_matrix=None,
                    segments=None,
                    standardize=True,
                    save_file=None):
    """
    Import feature matrix from file with segments in initial column. 
    If segments is specified, eliminates constant and redundant features. 
    If standardize flag is set:
    - Add epsilon symbol with all-zero feature vector,
    - Add symbol-presence feature (sym),
    - Add begin/end delimiters and feature to identify them (begin:+1, end:-1),
    - Add feature to identify consonants (C) and vowels (V) (C:+1, V:-1)
    Otherwise these symbols and features are assumed to be already present 
    in the feature matrix or file.
    todo: arrange segments in IPA order
    """

    # Read matrix from file or use arg matrix
    ftr_matrix = pd.read_csv(
        feature_matrix, sep=',', encoding='utf-8', comment='#')
    print(ftr_matrix)

    # Add long segments and length feature ("let there be colons")
    ftr_matrix_short = ftr_matrix.copy()
    ftr_matrix_long = ftr_matrix.copy()
    ftr_matrix_short['long'] = '-'
    ftr_matrix_long['long'] = '+'
    ftr_matrix_long.iloc[:, 0] = [x + 'ː' for x in ftr_matrix_long.iloc[:, 0]]
    ftr_matrix = pd.concat([ftr_matrix_short, ftr_matrix_long],
                           axis=0,
                           sort=False)

    # List all segments and features in the matrix, locate syllabic feature,
    # and remove first column (containing segments)
    # ftr_matrix.iloc[:,0] = [normalize('NFC', x) for x in ftr_matrix.iloc[:,0]]
    segments_all = [x for x in ftr_matrix.iloc[:, 0]]
    features_all = [x for x in ftr_matrix.columns[1:]]
    syll_ftr = [ftr for ftr in features_all \
        if re.match('^(syl|syllabic)$', ftr)][0]
    ftr_matrix = ftr_matrix.iloc[:, 1:]

    # Normalize unicode [partial]
    # no script g, no tiebars, ...
    ipa_substitutions = {'\u0261': 'g', 'ɡ': 'g', '͡': ''}
    for (s, r) in ipa_substitutions.items():
        segments_all = [re.sub(s, r, x) for x in segments_all]
    #print('segments_all:', segments_all)

    # Handle segments with diacritics [partial]
    diacritics = [
        ("ʼ", ('constr.gl', '+')),
        ("ʰ", ('spread.gl', '+')),
        ("[*]", ('constr.gl', '+')),  # Korean
        ("ʷ", ('round', '+'))
    ]
    diacritic_segs = []
    if segments is not None:
        for seg in segments:
            # Detect and strip diacritics
            base_seg = seg
            diacritic_ftrs = []  # features marked by diacritics
            for (diacritic, ftrval) in diacritics:
                if re.search(diacritic, base_seg):
                    diacritic_ftrs.append(ftrval)
                    base_seg = re.sub(diacritic, '', base_seg)
            if len(diacritic_ftrs) == 0:
                continue
            # Specify diacritic features
            idx = segments_all.index(base_seg)
            base_ftr = [x for x in ftr_matrix.iloc[idx, :]]
            for ftr, val in diacritic_ftrs:
                idx = features_all.index(ftr)
                base_ftr[idx] = val
            diacritic_segs.append((seg, base_ftr))
        # Add segments with diacritics and features
        if len(diacritic_segs) > 0:
            new_segs = [x[0] for x in diacritic_segs]
            new_ftr_vecs = pd.DataFrame([ftr for (seg, ftr) in diacritic_segs])
            new_ftr_vecs.columns = ftr_matrix.columns
            segments_all += new_segs
            ftr_matrix = pd.concat([ftr_matrix, new_ftr_vecs],
                                   ignore_index=True)
        #print(segments_all)
        #print(ftr_matrix)

    # Reduce feature matrix to observed segments (if provided), pruning
    # features other than syll_ftr that have constant values
    if segments is not None:
        # Check that all segments appear in the feature matrix
        missing_segments = [x for x in segments if x not in segments_all]
        if len(missing_segments) > 0:
            print(f'Segments missing from feature matrix: '
                  f'{missing_segments}')

        segments = [x for x in segments_all if x in segments]
        ftr_matrix = ftr_matrix.loc[[x in segments for x in segments_all], :]
        ftr_matrix.reset_index(drop=True)

        features = [ftr for ftr in ftr_matrix.columns \
            if ftr == 'syll_ftr' or ftr_matrix[ftr].nunique() > 1]
        ftr_matrix = ftr_matrix.loc[:, features]
        ftr_matrix = ftr_matrix.reset_index(drop=True)
    else:
        segments = segments_all
        features = features_all

    # Syllabic segments
    vowels = [ x for i, x in enumerate(segments) \
        if ftr_matrix[syll_ftr][i] == '+']

    # Standardize feature matrix
    ftr_matrix.index = segments
    fm = FeatureMatrix(segments, vowels, features, ftr_matrix, None)
    if standardize:
        fm = standardize_matrix(fm)

    # Convert feature columns to numpy vectors
    ftr_matrix_vec = fm.ftr_matrix.copy()
    ftr_specs = {'+': 1.0, '-': -1.0, '0': 0.0}
    for (key, val) in ftr_specs.items():
        ftr_matrix_vec = ftr_matrix_vec.replace(to_replace=key, value=val)
    ftr_matrix_vec = np.array(ftr_matrix_vec.values)
    fm = FeatureMatrix(fm.symbols, fm.vowels, fm.features, fm.ftr_matrix,
                       ftr_matrix_vec)

    # Write feature matrix
    if save_file is not None:
        fm.ftr_matrix.to_csv(save_file.with_suffix('.ftr'), index_label='ipa')

    setattr(phon_config, 'feature_matrix', fm)
    return fm


def make_one_hot_features(segments=None,
                          vowels=None,
                          standardize=True,
                          save_file=None):
    """
    Create one-hot feature matrix from list of segments (or number of segments), 
    optionally standardizing with special symbols and features.
    """
    if isinstance(segments, int):
        segments = string.ascii_lowercase[:segments]
    features = segments[:]
    ftr_matrix = np.eye(len(segments))
    fm = FeatureMatrix(segments, vowels, features, ftr_matrix)
    if standardize:
        fm = standardize_matrix(fm)

    if save_file is not None:
        ftr_matrix = fm.ftr_matrix
        ftr_matrix.to_csv(save_file.with_suffix('.ftr'), index_label='ipa')

    setattr(phon_config, 'feature_matrix', fm)
    return fm


def standardize_matrix(fm):
    """
    Add special symbols (epsilon, bos, eos) and features (sym, begin/end, C/V) to feature matrix
    """
    if fm.vowels is None:
        print('Vowels required in standardize_matrix')
        sys.exit(0)

    # # # # # # # # # #
    # Special symbols
    epsilon = phon_config.epsilon
    bos = phon_config.bos
    eos = phon_config.eos
    #wildcard = config.wildcard
    syms = [epsilon, bos, eos, *fm.symbols]

    # Special symbols are unspecified for all ordinary features
    special_sym_vals = pd.DataFrame({ftr: '0' for ftr in fm.features},
                                    index=[0])

    # Special symbols occupy first three rows of revised feature matrix
    ftr_matrix = pd.concat([special_sym_vals] * 3 +
                           [fm.ftr_matrix]).reset_index(drop=True)

    # # # # # # # # # #
    # Special features
    # Sym feature: all symbols except epsilon are +
    sym_ftr_vals = ['0'] + ['+'] * (len(syms) - 1)

    # Delim ftr: bos is +, eos is -, all others syms unspecified
    delim_ftr_vals = ['0', '+', '-'] + ['0'] * (len(syms) - 3)

    # C/V ftr: consonants are +, vowels are -, all other syms unspecified
    cv_ftr_vals = ['0', '0', '0'] + \
        ['-' if seg in fm.vowels else '+' for seg in fm.symbols]

    # Special features occupy first three columns of revised feature matrix
    special_ftrs = pd.DataFrame({
        'sym': sym_ftr_vals,
        'begin/end': delim_ftr_vals,
        'C/V': cv_ftr_vals
    })
    ftr_matrix = pd.concat([special_ftrs, ftr_matrix], axis=1) \
                   .reset_index(drop=True)
    ftr_matrix.index = syms
    features = ['sym', 'begin/end', 'C/V', *fm.features]
    phon_config.sym_ftr = sym_ftr = 0
    phon_config.delim_ftr = delim_ftr = 1
    phon_config.cv_ftr = cv_ftr = 2

    fm = FeatureMatrix(syms, fm.vowels, features, ftr_matrix, None)
    print(ftr_matrix)
    return fm


# [deprecated]
def ftrspec2vec(ftrspecs, feature_matrix=None):
    """
    Convert dictionary of feature specifications (ftr -> +/-/0) 
    to feature + 'attention' vectors.
    If feature_matrix is omitted, default to environ.config.
    """
    if feature_matrix is not None:
        features = feature_matrix.features
    else:
        features = config.ftrs

    specs = {'+': 1.0, '-': -1.0, '0': 0.0}
    n = len(features)
    w = np.zeros(n)
    a = np.zeros(n)
    for ftr, spec in ftrspecs.items():
        if spec == '0':
            continue
        i = features.index(ftr)
        if i < 0:
            print('ftrspec2vec: could not find feature', ftr)
        w[i] = specs[spec]  # non-zero feature specification
        a[i] = 1.0  # 'attention' weight identifying non-zero feature
    return w, a


def test():
    print('***')
    feature_matrix = Path.home() \
        / 'Code/Python/tensormorph_redup/ftrs/hayes_features.csv'
    import_features(
        feature_matrix, segments=['b', 'a'], save_file=Path('./tmp'))


if __name__ == "__main__":
    test()