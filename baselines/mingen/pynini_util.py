# -*- coding: utf-8 -*-

import re
import pynini
from pynini import Arc, Fst, SymbolTable
from pynini.lib import pynutil, rewrite
from phtrs import str_util
import config

# Create acceptors, unions, sigstar from symbol lists (cf. strings),
# compile rules from mingen format and apply to symbol lists.
# Note: Pynini word delimiters are "[BOS]", "[EOS]"

# TODO: review documentation for context management
# with pynini.default_token_type(token_type):
#   ...
# https://github.com/kylebgorman/Pynini/issues/8
# see also (on how operations affect SymbolTables):
# https://github.com/kylebgorman/pynini/issues/22

# Mark locus of cdrewrite rule application
markers = ['⟨', '⟩']


def sigstar(syms):
    """
    Symbol table and Sigma* acceptor fom list of symbols
    """
    symtable = SymbolTable()
    symtable.add_symbol(config.epsilon)  # Epsilon has id 0
    for sym in syms:
        symtable.add_symbol(sym)
    for sym in markers:
        symtable.add_symbol(sym)

    #fsts = accep(syms + markers, symtable)
    #sigstar = union(fsts).closure().optimize()
    with pynini.default_token_type(symtable):
        sigstar = pynini.string_map(syms + markers) \
                        .closure().optimize()
    return sigstar, symtable


def accep(x, symtable):
    """
    Map space-separated sequence of symbols to acceptor (identity transducer)
    [pynini built-in, see bottom of "Constructing acceptors" in documentation]
    """
    # List of sequences -> list of Fsts
    if isinstance(x, list):
        return [accep(xi, symtable) for xi in x]
    # Single sequence -> Fst
    fst = pynini.accep(x, token_type=symtable)
    return fst


def accep_(x, symtable):
    """
    Map space-separated sequence of symbols to acceptor (identity transducer)
    [demo version using primitive FST functions]
    """
    fst = Fst()
    fst.set_input_symbols(symtable)
    fst.set_output_symbols(symtable)

    x = x.split(' ')
    n = len(x)
    fst.add_states(n + 1)
    fst.set_start(0)
    fst.set_final(n)
    for i in range(n):
        iolabel = symtable.find(x[i])
        fst.add_arc(i, Arc(iolabel, iolabel, 0, i + 1))
    return fst


# xxx pynini built-in?
def union(fsts):
    """ Union list of Fsts """
    fst = pynini.union(*fsts)
    return fst  # xxx check symbol table


# xxx pynini built-in?
def concat(fsts):
    """ Concatenate list of Fsts """
    n = 0 if fsts is None else len(fsts)
    if n == 0:
        return None
    if n == 1:
        return fsts[0]
    fst = pynini.concat(fsts[0], fsts[1])
    for i in range(2, n):
        fst = pynini.concat(fst, fsts[i])
    return fst  # xxx check symbol table


def compile_context(C, symtable):
    """
    Convert context (sequence of regexs) to Fst
    """
    # Empty context xxx document
    if C == "[ ]*":
        return C
    # Ordinary context
    fsts = []
    for regex in C.split(' '):
        # Remove grouping parens and make list of symbols
        regex = re.sub('[()]', '', regex).split('|')
        fst = union(accep(regex, symtable))
        fsts.append(fst)
    fst = concat(fsts)
    return fst  # xxx check symbol table


def compile_rule(A, B, C, D, sigstar, symtable):
    """
    Compile cdrewrite rule from A -> B / C __D where A and B are space-separated strings, C and D are segment regexs (seg1|seg2|...)
    """
    # Use explicit epsilon for deletion rules, instead of pynutil.delete(),
    # and mark rewrite loci (for checking whether rule applies to input)
    if B == '∅':
        B = config.epsilon
    B = ' '.join(['⟨', B, '⟩'])
    # Insertion rule
    if A == '∅':
        change = pynutil.insert(accep(B, symtable))
    # Change or deletion rule
    else:
        A_fst = accep(A, symtable)
        B_fst = accep(B, symtable)
        change = pynini.cross(A_fst, B_fst)

    with pynini.default_token_type(symtable):
        left_context = compile_context(C, symtable)
        right_context = compile_context(D, symtable)
    fst = pynini.cdrewrite(change, left_context, right_context,
                           sigstar).optimize()
    # xxx check symbol table
    return fst


def rewrites(rule, inpt, outpt, sigstar=None, symtable=None):
    """
    Determines whether inpt is within scope of rule and 
    if so whether application results in outpt
    """
    if isinstance(rule, Fst):
        rule_fst = rule
    else:
        (A, B, C, D) = rule
        rule_fst = compile_rule(A, B, C, D, sigstar, symtable)

    if isinstance(inpt, Fst):
        inpt_fst = inpt
    else:
        inpt_fst = accep(inpt, symtable)

    pred_fst = inpt_fst @ rule_fst
    strpath_iter = pred_fst.paths(
        input_token_type=symtable, output_token_type=symtable)
    pred = [x for x in strpath_iter.ostrings()][0]  # xxx

    in_scope, hit = 0, 0
    if re.search('⟨', pred):
        in_scope = 1
        pred = str_util.remove(pred, markers)
        hit = int(pred == outpt)
    return {'in_scope': in_scope, 'hit': hit}


def apply_rule(rule, inpt, sigstar=None, symtable=None):
    """
    Determines whether inpt is within scope of rule and
    if so returns application of rule to inpt
    """
    if isinstance(rule, Fst):
        rule_fst = rule
    else:
        (A, B, C, D) = rule
        rule_fst = compile_rule(A, B, C, D, sigstar, symtable)

    if isinstance(inpt, Fst):
        inpt_fst = inpt
    else:
        inpt_fst = accep(inpt, symtable)

    pred_fst = inpt_fst @ rule_fst
    strpath_iter = pred_fst.paths(
        input_token_type=symtable, output_token_type=symtable)
    pred = [x for x in strpath_iter.ostrings()][0]  # xxx

    if re.search('⟨', pred):
        pred = str_util.remove(pred, markers)
    return pred


def edit1_fst(sigstar, symtable):
    """ Map inputs to outputs one edit away (e.g., for word neighborhoods) """
    fst = Fst()

    q0 = fst.add_state()
    q1 = fst.add_state()
    fst.set_start(q0)
    fst.set_final(q1)
    for sym1_id, sym1 in symtable:
        if sym1 in [config.bos, config.eos] + []:
            continue
        for sym2_id, sym2 in symtable:
            if sym2_id == sym1_id:
                continue
            if sym2 in [config.bos, config.eos]:
                continue
            fst.add_arc(q0, Arc(sym1_id, sym2_id, 0, q1))
    fst = accep(config.bos, symtable) \
          + sigstar + fst + sigstar \
          + accep(config.eos, symtable)
    fst = fst.optimize()
    fst.set_input_symbols(symtable)
    fst.set_output_symbols(symtable)
    return fst


def istrings(fst, symtable):
    """ Input strings of paths through acyclic FST """
    strpath_iter = fst.paths(input_token_type=symtable)
    return [x for x in strpath_iter.istrings()]


def ostrings(fst, symtable):
    """ Output strings of paths through acyclic FST """
    strpath_iter = fst.paths(output_token_type=symtable)
    return [x for x in strpath_iter.ostrings()]


def test():
    config.epsilon = 'ϵ'
    syms = ['aa', 'bb', 'cc', 'dd']
    sigstar_, symtable = sigstar(syms)

    # Rule aa -> bb / cc __ dd
    rule1 = compile_rule('aa', 'bb', '(cc)', '(dd)', sigstar_, symtable)
    print(rule1.print())
    rule1.draw('rule1.dot', isymbols=symtable, osymbols=symtable, portrait=True)
    # dot -Tpdf rule1.dot -o rule1.pdf

    # Apply rule to input
    input1 = accep('cc aa dd', symtable)
    output1 = (input1 @ rule1)
    print(output1.print(isymbols=symtable, osymbols=symtable))
    print(ostrings(output1, symtable))

    # Rule aa -> ∅ / cc __ dd, apply to input
    rule2 = compile_rule('aa', '∅', 'cc', 'dd', sigstar_, symtable)
    output2 = (input1 @ rule2).project("output").rmepsilon()
    print(output2.print(isymbols=symtable, osymbols=symtable))
    print(ostrings(output2, symtable))


if __name__ == "__main__":
    test()