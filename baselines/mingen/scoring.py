# -*- coding: utf-8 -*-

import sys
import numpy as np
from scipy.stats import t as student_t

import config
from rules import *
from phtrs import str_util
import pynini_util

verbosity = 0


def score_rules(R):
    """
    Hits and scope for FtrRules on training data
    todo: apply simultaneously to all inputs encoded as trie?
    """
    # Symbol environment
    syms = [x for x in config.sym2ftrs]
    sigstar, symtable = pynini_util.sigstar(syms)

    # Precompile inputs to FSTs
    dat = config.dat_train
    stems = [str(x) for x in dat['stem']]
    outputs = [str(x) for x in dat['output']]
    stem_ids = list(range(len(dat)))
    wordforms = list(zip(stems, outputs, stem_ids))
    stem_fsts = pynini_util.accep(stems, symtable)

    # Hits and scope for each rule
    print("Hits and scope ...")
    hits_all = [0.0] * len(R)
    scope_all = [0.0] * len(R)
    for idx, rule in enumerate(R):
        # Convert rule to regexes
        (A, B, C, D) = rule.regexes()

        # Subset of data s.t. CAD occurs in input
        CAD = [X for X in [C, A, D] if X != '∅']
        CAD = ' '.join(CAD)
        if CAD != '':
            subdat = [wf for wf in wordforms if re.search(CAD, wf[0])]
        else:
            subdat = wordforms

        # Skip rules with zero scope
        if len(subdat) == 0:
            hits_all[idx] = 0
            scope_all[idx] = 0
            continue

        # Compile rule to FST
        rule_fst = pynini_util.compile_rule(A, B, C, D, sigstar, symtable)

        # Loop over input/output pairs in data subset
        hits, scope = 0.0, 0.0
        for (stem, output, stem_id) in subdat:
            stem_fst = stem_fsts[stem_id]
            rewrite_val = pynini_util.rewrites(rule_fst, stem_fst, output,
                                               sigstar, symtable)
            #print(rule.regexes())
            #print(stem, '->', output, rewrite_val)
            if rewrite_val['in_scope']:
                scope += 1.0
            if rewrite_val['hit']:
                hits += 1.0

        hits_all[idx] = hits
        scope_all[idx] = scope
        if hits == 0.0 and verbosity > 0:
            print('(warning) rule has zero hits')
            print(R[idx])
            print(repr(R[idx]))
        if scope == 0.0:
            print('(warning) rule has zero scope')
            print(R[idx])
            print(repr(R[idx]))
            print(len(subdat))
            sys.exit(0)
        print(f'rule {idx}, hits = {hits}, scope = {scope}, '
              f'raw accuracy = {hits/scope}')

    return hits_all, scope_all


def confidence(hits, scope, alpha=0.55):
    """
    Adjust reliability by scope
    (default alpha from A&H 2003:127)
    """
    # Adjusted reliability
    p_star = (hits + 0.5) / (scope + 1.0)
    # Estimated variance
    var_est = (p_star * (1 - p_star)) / scope
    var_est = var_est**0.5
    # Confidence
    z = student_t.ppf(alpha, scope - 1.0)
    c = p_star - z * var_est
    if np.isnan(c):  # xxx document
        c = 0.0
    return c


def test():
    # Exampes from A&H 2003:127
    hits, scope = 5.0, 5.0
    print(confidence(hits, scope, alpha=0.75))
    hits, scope = 1000, 1000.0
    print(confidence(hits, scope, alpha=0.75))


if __name__ == "__main__":
    test()