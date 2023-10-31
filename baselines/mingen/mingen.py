# -*- coding: utf-8 -*-

from functools import lru_cache

from features import common_ftrs
from rules import FtrRule


def generalize_rules_rec(R):
    """
    Recursively apply minimal generalization to set of FtrRules
    todo: generalize each context pair at most once
    """
    print('Minimal generalization ...')
    # Word-specific rules
    # Rules grouped by common change [invariant]
    R_base = {}
    for rule in R:
        change = f"{' '.join(rule.A)} -> {' '.join(rule.B)}"
        if change in R_base:
            R_base[change].append(rule)
        else:
            R_base[change] = [rule]
    R_base = {change: set(rules) \
                for change, rules in R_base.items()}
    R_all = {change: rules.copy() \
                for change, rules in R_base.items()}

    # First-step minimal generalization
    print('Iteration 0 (base rules only) ...')
    R_new = {}
    for change, rules_base in R_base.items():
        print(f'\t{change} [{len(rules_base)}^2]')
        rules_base = list(rules_base)
        n = len(rules_base)
        rules_new = set()
        for i in range(n - 1):
            rule1 = rules_base[i]
            for j in range(i + 1, n):
                rule2 = rules_base[j]
                rule = generalize_rules(rule1, rule2)
                if rule is None:
                    continue
                rules_new.add(rule)
        R_new[change] = rules_new
    for change in R_all:
        R_all[change] |= R_new[change]

    # Recursive minimal generalization
    for i in range(10):  # xxx loop forever
        # Report number of rules by change
        print(f'Recursion {i+1} ...')

        # One-step minimal generalization
        R_old = R_new
        R_new = {}
        for change, rules_base in R_base.items():
            rules_old = R_old[change]
            print(f'\t{change} [{len(rules_base)} x {len(rules_old)}]')
            rules_new = set()
            for rule1 in rules_base:
                for rule2 in rules_old:
                    rule = generalize_rules(rule1, rule2)
                    if rule is None:
                        continue
                    rules_new.add(rule)
            R_new[change] = (rules_new - R_all[change])

        # Update rule sets
        new_rule_flag = False
        for change in R_all:
            size_old = len(R_all[change])
            R_all[change] |= R_new[change]
            size_new = len(R_all[change])
            if size_new > size_old:
                new_rule_flag = True

        # Stop if no new rules added for any context
        if not new_rule_flag:
            break

    R_all = [rule for change, rules in R_all.items() for rule in rules]
    return R_all


def generalize_rules(rule1, rule2):
    """
    Apply minimal generalization to pair of FtrRules
    """
    # Check for matching change A -> B
    if (rule1.A != rule2.A) or (rule1.B != rule2.B):
        return None

    # Generalize left and right contexts
    C = generalize_context(rule1.C, rule2.C, '<-RL')
    D = generalize_context(rule1.D, rule2.D, 'LR->')

    R = FtrRule(rule1.A, rule1.B, C, D)
    return R


@lru_cache(maxsize=1000)
def generalize_context(C1, C2, direction='LR->'):
    """
    Apply minimal generalization to pair of feature contexts, working inward (<-RL) or outward (LR->) from change location
    """
    assert ((direction == 'LR->') or (direction == '<-RL'))
    if direction == '<-RL':
        C1 = C1[::-1]
        C2 = C2[::-1]

    n1 = len(C1)
    n2 = len(C2)
    n_min = min(n1, n2)
    n_max = max(n1, n2)

    C = []
    seg_ident_flag = True
    for i in range(n_max):
        # X (Sigma*) and terminate if have exceeded shorter context
        # or have already gone beyond segment identity
        if (i >= n_min) or (not seg_ident_flag):
            C.append('X')
            break
        # X (Sigma*) and terminate if either is X
        if (C1[i] == 'X') or (C2[i] == 'X'):
            C.append('X')
            break
        # Perfect match of feature matrices
        # (NB. Conforms to A&H spec only if at least one
        # of the rules has contexts that are segment-specific)
        if C1[i] == C2[i]:
            C.append(C1[i])
            continue
        # Shared features at first segment mismatch
        ftrs, any_common = common_ftrs(C1[i], C2[i])
        if not any_common:
            C.append('X')
            break
        C.append(ftrs)
        seg_ident_flag = False

    if direction == '<-RL':
        C = C[::-1]

    C = tuple(C)
    return C
