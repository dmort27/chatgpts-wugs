# -*- coding: utf-8 -*-

# Feature matching and unification, convert feature matrices to/from strings

import re
import config
from functools import lru_cache


def match_ftrs_(F, sym):
    """
    Subsumption relation between feature matrix and symbol
    [feature dicts]
    """
    ftrs = config.sym2ftrs[sym]
    for (ftr, val) in F.items():
        if ftrs[ftr] != val:
            return False
    return True


def match_ftrs(F, sym):
    """
    Subsumption relation between feature matrix and symbol
    [feature vectors]
    """
    ftrs = config.sym2ftr_vec[sym]
    n = len(ftrs)
    for i, Fi in enumerate(F):
        if Fi == '0':
            continue
        if ftrs[i] != Fi:
            return False
    return True


def common_ftrs_(F1, F2):
    """
    Shared values of two feature matrices
    [Args: feature dicts]
    """
    # f(Sigma*,F2) = f(F1,Sigma*) = Sigma*
    if (F1 == 'X') or (F2 == 'X'):
        return 'X'
    # Ordinary unification
    F = {}
    for (ftr, val) in F1.items():
        if (ftr in F2) and (F2[ftr] == val):
            F[ftr] = val
    return F


@lru_cache(maxsize=1000)
def common_ftrs(F1, F2):
    """
    Common values of two feature matrices
    [Args: feature vectors]
    """
    # f(Sigma*,F2) = f(F1,Sigma*) = Sigma*
    if (F1 == 'X') or (F2 == 'X'):
        return 'X'
    # Ordinary unification
    n = len(F1)
    F = ['0'] * n
    any_common = False
    for i, F1_i in enumerate(F1):
        F2_i = F2[i]
        if (F1_i == '0') or (F2_i == '0'):
            continue
        if F1_i == F2_i:
            F[i] = F1_i
            any_common = True
    return tuple(F), any_common


@lru_cache(maxsize=10000)
def subsumes(F1, F2):
    """
    Subsumption relation between feature matrices F1 and F2
    [Args: feature vectors]
    """
    if (F1 == 'X'):
        return True
    if (F2 == 'X'):
        return False
    n = len(F1)
    for i, F1i in enumerate(F1):
        if F1i == '0':
            continue
        if F1i != F2[i]:
            return False
    return True


def ftrs2regex(F):
    """
    Symbol regex for sequence of feature matrices
    Note: excludes X (Sigma*), which is assumed to appear only at edges of rule contexts and is always implicit at those positions in cdrewrite compilation
    """
    return ' '.join([ftrs2regex1(Fi) for Fi in F if Fi != 'X'])


def ftrs2regex1(F):
    """ Symbol regex for single feature matrix """
    if F == 'X':
        return 'X'
    syms = [sym for sym in config.sym2ftrs \
            if match_ftrs(F, sym)]
    return '(' + '|'.join(syms) + ')'


def ftrs2str(F):
    """ String corresponding to sequence of feature matrices """
    return ' '.join([ftrs2str1(Fi) for Fi in F])


def ftrs2str1(F):
    """ String corresponding to feature matrix, with non-zero values only """
    if F == 'X':
        return 'X'
    ftr_names = config.ftr_names
    ftrvals = [f'{F[i]}{ftr_names[i]}' \
                for i in range(len(F)) if F[i] != '0']
    val = '[' + ', '.join(ftrvals) + ']'
    return val


def str2ftrs(x):
    """ String to sequence of feature matrices (inverse of ftrs2str) """
    y = re.sub('X', '[X', x)  # Sigma*
    y = y.split(' [')
    try:
        y = [str2ftrs1(yi) for yi in y]
    except:
        print(f'Error in str2ftrs for input {y}')
        sys.exit(0)
    return tuple(y)


def str2ftrs1(x):
    """ String to feature matrix (inverse of ftrs2str1) """
    y = re.sub('\\[', '', x)
    y = re.sub('\\]', '', y)
    if y == 'X':  # Sigma*
        return 'X'
    # Ordinary feature matrix, non-zero specs only
    y = y.split(', ')
    ftrs = ['0'] * len(config.ftr_names)
    ftr_names = config.ftr_names
    for spec in y:
        if spec == '':  # xxx document
            continue
        val = spec[0]
        ftr = spec[1:]
        ftrs[ftr_names.index(ftr)] = val
    return tuple(ftrs)
