# -*- coding: utf-8 -*-

import sys
import numpy as np
from collections import namedtuple
from functools import lru_cache
from features import *
from rules import *

ScoredRule = namedtuple('ScoredRule', ['R', 'score', 'length', 'idx'])


def prune_rules(rules, score_type='confidence', digits=10):
    """
    Prune rules that are bounded by more general rules or have scores <= 0
    """
    print('Prune ...')
    rules = rules.sort_values(by=score_type, ascending=False)
    R = [ScoredRule(FtrRule.from_str(rule),
                        np.round(score, digits),
                        len(rule),
                        idx) \
        for (rule, score, idx) \
            in zip(rules['rule'], rules[score_type], rules['rule_idx']) \
        if score > 0.0]

    i = 0
    pruned = []  # Non-maximal rules
    print(f'rules {len(R)}')
    print('iter pruned')
    while len(R) > 0:
        if i > 0 and i % 100 == 0:
            print(i, len(pruned))
        rulei_ = R.pop(0)
        prune_flag = False
        for j, rulej_ in enumerate(R):
            cmp = rule_cmp(rulei_, rulej_)
            if cmp == -1:
                pruned.append(rulej_)
                R[j] = None
            if cmp == +1:
                prune_flag = True
        if prune_flag:
            pruned.append(rulei_)
        R = [rule_ for rule_ in R if rule_ is not None]
        i += 1

    print(f'{len(pruned)} pruned rules')  # 30261 pruned rules

    # Keep rules that are maximal wrt rule_cmp and have scores >= 0
    idx_pruned = [rule_.idx for rule_ in pruned]
    rules_max = rules[~(rules['rule_idx'].isin(idx_pruned))]
    rules_max = rules_max[(rules_max[score_type] > 0.0)]
    rules_max = rules_max.sort_values(by=score_type, ascending=False)
    return rules_max


def rule_cmp(rule1_: ScoredRule, rule2_: ScoredRule):
    """ Compare rules by score and generality, breaking ties with length
        -1 if score1 > score2 and rule1 ⊒ rule2   -or-
              score1 == score2 and rule1 ⊐ rule2  -or-
              score1 == score2 and rule1 = rule2 and length1 < length2
        +1 if score2 > score1 and rule2 ⊒ rule1   -or-
              score2 == score1 and rule2 ⊐ rule1  -or-
              score2 == score1 and rule2 = rule1 and length2 < length1
        0 otherwise
    """
    rule1, score1, length1, idx1 = rule1_
    rule2, score2, length2, idx2 = rule2_

    # R1 has higher score
    if score1 > score2:
        if rule_mgt(rule1, rule2):
            return -1
    # R2 has higher score
    elif score2 > score1:
        if rule_mgt(rule2, rule1):
            return +1
    # Tied on score
    else:
        mgt12 = rule_mgt(rule1, rule2)
        mgt21 = rule_mgt(rule2, rule1)
        if mgt12:
            if not mgt21 or length1 < length2:
                return -1
        if mgt21:
            if not mgt12 or length2 < length1:
                return +1
    return 0


def rule_mgt(rule1: FtrRule, rule2: FtrRule):
    """ More-general-than-or-equal relation ⊒ on rules """
    # Apply only to rules with same focus and change xxx fixme
    if (rule1.A != rule2.A) or (rule1.B != rule2.B):
        return False

    if not context_mgt(rule1.C, rule2.C, '<-RL'):
        return False

    if not context_mgt(rule1.D, rule2.D, 'LR->'):
        return False

    return True


@lru_cache(maxsize=1000)
def context_mgt(C1, C2, direction='LR->'):
    """
    More-general-than-or-equal relation ⊒ on rule contexts (sequences of feature matrices), inward (<-RL) or outward (LR->) from change location
    """
    assert ((direction == 'LR->') or (direction == '<-RL'))
    if direction == '<-RL':
        C1 = C1[::-1]
        C2 = C2[::-1]
    n1 = len(C1)
    n2 = len(C2)

    # Empty context is always more general
    if n1 == 0:
        return True

    # Longer context cannot be more general
    # (except for special case below)
    if (n1 - n2) > 1:
        return False

    for i in range(n1):
        # Special case: context C1 has one more matrix than C2,
        # test whether it is identical to X (Sigma*)
        if i == n2:
            return (C1[i] == 'X')

        # Matrix C1[i] is not more general than C2[i]
        if not subsumes(C1[i], C2[i]):
            return False

    return True


def test():
    rule1 = ScoredRule(
        FtrRule.from_str(
            "∅ -> d / X [-spread.gl] [-C/V, +syllabic, -consonantal, +sonorant, +continuant, +approximant, -nasal, +voice, -spread.gl, -LABIAL, -round, -labiodental, -CORONAL, -lateral, +DORSAL, -high, -low, +front, -back, -tense] [+C/V, -syllabic, -consonantal, +sonorant, +continuant, +approximant, -nasal, +voice, -spread.gl, -LABIAL, -round, -labiodental, +CORONAL, -anterior, +distributed, -strident, -lateral, -DORSAL] __ [-begin/end]"
        ), 0.9978736, -1, 1)
    rule2 = ScoredRule(
        FtrRule.from_str(
            "∅ -> d / X [-spread.gl, -lateral] [-C/V, +syllabic, -consonantal, +sonorant, +continuant, +approximant, -nasal, +voice, -spread.gl, -LABIAL, -round, -labiodental, -CORONAL, -lateral, +DORSAL, -high, -low, +front, -back, -tense] [+C/V, -syllabic, -consonantal, +sonorant, +continuant, +approximant, -nasal, +voice, -spread.gl, -LABIAL, -round, -labiodental, +CORONAL, -anterior, +distributed, -strident, -lateral, -DORSAL] __ [-begin/end]"
        ), 0.9978424, -1, 2)
    print(rule_cmp(rule1, rule2))