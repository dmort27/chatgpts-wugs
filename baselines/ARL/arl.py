#!/usr/bin/env python3
"""
Non-neural baseline system for the SIGMORPHON 2020 Shared Task 0.

Author: Mans Hulden
Modified by: Tiago Pimentel
Modified by: Abhishek Vijayakumar
Last Update: 2023-10-21
"""
import getopt
import itertools
import os
import re
import sys
from functools import cache, wraps


def hamming(s, t):
    return sum(1 for x, y in zip(s, t) if x != y)


def halign(s, t):
    """Align two strings by Hamming distance."""
    minscore = len(s) + len(t) + 1
    for upad in range(0, len(t) + 1):
        upper = '_' * upad + s + (len(t) - upad) * '_'
        lower = len(s) * '_' + t
        score = hamming(upper, lower)
        if score < minscore:
            bu = upper
            bl = lower
            minscore = score

    for lpad in range(0, len(s) + 1):
        upper = len(t) * '_' + s
        lower = (len(s) - lpad) * '_' + t + '_' * lpad
        score = hamming(upper, lower)
        if score < minscore:
            bu = upper
            bl = lower
            minscore = score

    zipped = zip(bu, bl)
    newin = ''.join(i for i, o in zipped if i != '_' or o != '_')
    newout = ''.join(o for i, o in zipped if i != '_' or o != '_')
    return newin, newout


def levenshtein(s, t, inscost=1.0, delcost=1.0, substcost=1.0):
    """Recursive implementation of Levenshtein, with alignments returned."""

    @cache
    def lrec_fast(i, j):
        # spast, srem = s[:i], s[i:]
        # tpast, trem = t[:j], t[j:]

        if i == len(s) and j == len(t):
            return 0, None
        elif i == len(s):
            return lrec_fast(i, j + 1)[0] + inscost, 1
        elif j == len(t):
            return lrec_fast(i + 1, j)[0] + delcost, -1

        total_sub = lrec_fast(i + 1, j + 1)[0] + (
            substcost if s[i] != t[j] else 0)
        total_ins = lrec_fast(i, j + 1)[0] + inscost
        total_del = lrec_fast(i + 1, j)[0] + delcost

        retval = min([(total_sub, 0), (total_ins, 1), (total_del, -1)],
                     key=lambda x: x[0])
        return retval

    @memolrec
    def lrec(spast, tpast, srem, trem, cost):
        if len(srem) == 0:
            return spast + len(trem) * '_', tpast + trem, '', '', cost + len(
                trem)
        if len(trem) == 0:
            return spast + srem, tpast + len(srem) * '_', '', '', cost + len(
                srem)

        addcost = 0
        if srem[0] != trem[0]:
            addcost = substcost

        return min((lrec(spast + srem[0], tpast + trem[0], srem[1:], trem[1:],
                         cost + addcost),
                    lrec(spast + '_', tpast + trem[0], srem, trem[1:],
                         cost + inscost),
                    lrec(spast + srem[0], tpast + '_', srem[1:], trem,
                         cost + delcost)),
                   key=lambda x: x[4])

    fast_cost, action = lrec_fast(0, 0)
    salign, talign = [], []
    i, j = 0, 0
    while action is not None:
        if action == 1:
            salign.append("_")
            talign.append(t[j])
            j += 1
        elif action == -1:
            talign.append("_")
            salign.append(s[i])
            i += 1
        else:
            salign.append(s[i])
            talign.append(t[j])
            i += 1
            j += 1
        action = lrec_fast(i, j)[1]

    # answer = lrec('', '', s, t, 0)
    # assert round(fast_cost, 3) == round(answer[4], 3)
    # if "".join(salign) != answer[0]:
    #     print("".join(salign), answer[0])
    # assert "".join(salign) == answer[0]
    # assert "".join(talign) == answer[1]
    # return answer[0],answer[1],answer[4]
    return "".join(salign), "".join(talign), fast_cost


def memolrec(func):
    """Memoizer for Levenshtein."""
    cache = {}

    @wraps(func)
    def wrap(sp, tp, sr, tr, cost):
        if (sr, tr) not in cache:
            res = func(sp, tp, sr, tr, cost)
            cache[(sr, tr)] = (
            res[0][len(sp):], res[1][len(tp):], res[4] - cost)
        return sp + cache[(sr, tr)][0], tp + cache[(sr, tr)][1], '', '', cost + \
               cache[(sr, tr)][2]

    return wrap


def alignprs(lemma, form):
    """Break lemma/form into three parts:
    IN:  1 | 2 | 3
    OUT: 4 | 5 | 6
    1/4 are assumed to be prefixes, 2/5 the stem, and 3/6 a suffix.
    1/4 and 3/6 may be empty.
    """

    al = levenshtein(lemma, form,
                     substcost=1.1)  # Force preference of 0:x or x:0 by 1.1 cost
    alemma, aform = al[0], al[1]
    # leading spaces
    lspace = max(len(alemma) - len(alemma.lstrip('_')),
                 len(aform) - len(aform.lstrip('_')))
    # trailing spaces
    tspace = max(len(alemma[::-1]) - len(alemma[::-1].lstrip('_')),
                 len(aform[::-1]) - len(aform[::-1].lstrip('_')))
    return alemma[0:lspace], alemma[lspace:len(alemma) - tspace], alemma[
                                                                  len(alemma) - tspace:], aform[
                                                                                          0:lspace], aform[
                                                                                                     lspace:len(
                                                                                                         alemma) - tspace], aform[
                                                                                                                            len(alemma) - tspace:]


def prefix_suffix_rules_get(lemma, form):
    """Extract a number of suffix-change and prefix-change rules
    based on a given example lemma+inflected form."""
    lp, lr, ls, fp, fr, fs = alignprs(lemma,
                                      form)  # Get six parts, three for in three for out

    # Suffix rules
    ins = lr + ls + ">"
    outs = fr + fs + ">"
    srules = set()
    for i in range(min(len(ins), len(outs))):
        srules.add((ins[i:], outs[i:]))
    srules = {(x[0].replace('_', ''), x[1].replace('_', '')) for x in srules}

    # Prefix rules
    prules = set()
    if len(lp) >= 0 or len(fp) >= 0:
        inp = "<" + lp
        outp = "<" + fp
        for i in range(0, len(fr)):
            prules.add((inp + fr[:i], outp + fr[:i]))
            prules = {(x[0].replace('_', ''), x[1].replace('_', '')) for x in
                      prules}

    return prules, srules


def apply_best_rule(lemma, msd, allprules, allsrules, num):
    """Applies the longest-matching suffix-changing rule given an input
    form and the MSD. Length ties in suffix rules are broken by frequency.
    For prefix-changing rules, only the most frequent rule is chosen."""

    if num is None:
        base = "<" + lemma + ">"
        if msd not in allprules and msd not in allsrules:
            return lemma  # Haven't seen this inflection, so bail out

        if msd in allsrules:
            applicablerules = [(x[0], x[1], y) for x, y in
                               allsrules[msd].items() if x[0] in base]
            if applicablerules:
                bestrule = max(applicablerules,
                               key=lambda x: (len(x[0]), x[2], len(x[1])))
                base = base.replace(bestrule[0], bestrule[1])

        if msd in allprules:
            applicablerules = [(x[0], x[1], y) for x, y in
                               allprules[msd].items() if x[0] in base]
            if applicablerules:
                bestrule = max(applicablerules, key=lambda x: (x[2]))
                base = base.replace(bestrule[0], bestrule[1])

        base = base.replace('<', '')
        base = base.replace('>', '')
        return base
    else:
        base = "<" + lemma + ">"
        if msd not in allprules and msd not in allsrules:
            return lemma  # Haven't seen this inflection, so bail out

        if msd in allsrules:
            applicablerules = [(x[0], x[1], y) for x, y in
                               allsrules[msd].items() if x[0] in base]
            # if applicablerules:  # should be fine to comment here if sorting
            sorted_rules_s = sorted(applicablerules,
                                    key=lambda x: (len(x[0]), x[2], len(x[1])),
                                    reverse=True)[:num]

            # base = base.replace(bestrule[0], bestrule[1])
            in_all_s = True
        else:
            in_all_s = False
            sorted_rules_s = []

        if msd in allprules:
            applicablerules = [(x[0], x[1], y) for x, y in
                               allprules[msd].items() if x[0] in base]
            # if applicablerules:  # should be fine to comment here if sorting
            sorted_rules_p = sorted(applicablerules, key=lambda x: (x[2]),
                                    reverse=True)[:num]

            # base = base.replace(bestrule[0], bestrule[1])
            in_all_p = True
        else:
            in_all_p = False
            sorted_rules_p = []

        if in_all_p and in_all_s:
            # TODO: decide on appropriate combination for key here
            sorted_rules = sorted(
                itertools.product(sorted_rules_s, sorted_rules_p),
                key=lambda x: (len(x[0][0]), x[0][2] + x[1][2], len(x[0][1])),
                reverse=True)[:num]
            out = [base.replace(sr[0], sr[1]).replace(pr[0], pr[1]) for sr, pr
                   in sorted_rules]
        elif in_all_s:
            out = [base.replace(r[0], r[1]) for r in sorted_rules_s]
        elif in_all_p:
            out = [base.replace(r[0], r[1]) for r in sorted_rules_p]
        else:
            raise ValueError

        out = [b.replace('<', '').replace('>', '') for b in out]
        return out


def numleadingsyms(s, symbol):
    return len(s) - len(s.lstrip(symbol))


def numtrailingsyms(s, symbol):
    return len(s) - len(s.rstrip(symbol))


###############################################################################


def main(argv):
    options, remainder = getopt.gnu_getopt(argv[1:], 'ohp:n:',
                                           ['output', 'help', 'path=',
                                            'number='])
    OUTPUT, HELP, path = False, False, './part1/development_languages/'
    num = None
    for opt, arg in options:
        if opt in ('-o', '--output'):
            OUTPUT = True
        elif opt in ('-h', '--help'):
            HELP = True
        elif opt in ('-p', '--path'):
            path = arg
        elif opt in ('-n', '--number'):
            num = int(arg)

    if HELP:
        print("\n*** Baseline for the SIGMORPHON 2020 shared task ***\n")
        print(
            "By default, the program runs all languages only evaluating accuracy.")
        print("To create output files, use -o")
        print(
            "The training and dev-data are assumed to live in ./part1/development_languages/")
        print("Options:")
        print(
            " -o         create output files with guesses (and don't just evaluate)")
        print(
            " -p [path]  data files path. Default is ./part1/development_languages/")
        quit()

    # SWITCHED FOR WUG DATA
    # train_suffix = "train"
    train_suffix = "trn"
    # SWITCHED FOR WUG DATA
    # test_suffix = "dev"
    # test_suffix = "tst"
    test_suffix = "nonce"
    train_ending = f".{train_suffix}"

    totalavg, numlang = 0.0, 0
    for lang in sorted(list({re.sub(f'\.{train_suffix}.*$', '', d) for d in
                             os.listdir(path) if train_ending in d})):
        allprules, allsrules = {}, {}
        if not os.path.isfile(path + lang + train_ending):
            continue
        lines = [line.strip() for line in
                 open(path + lang + train_ending, "r", encoding="utf8") if
                 line != '\n']
        # First, test if language is predominantly suffixing or prefixing
        # If prefixing, work with reversed strings
        prefbias, suffbias = 0, 0
        for l in lines:
            # SWITCHED FOR WUG DATA
            # lemma, form, _ = l.split(u'\t')
            lemma, _, form = l.split(u'\t')
            aligned = halign(lemma, form)
            if ' ' not in aligned[0] and ' ' not in aligned[1] and '-' not in \
                    aligned[0] and '-' not in aligned[1]:
                prefbias += numleadingsyms(aligned[0], '_') + numleadingsyms(
                    aligned[1], '_')
                suffbias += numtrailingsyms(aligned[0], '_') + numtrailingsyms(
                    aligned[1], '_')
        for l in lines:  # Read in lines and extract transformation rules from pairs
            # SWITCHED FOR WUG DATA
            # lemma, form, msd = l.split(u'\t')
            lemma, msd, form = l.split(u'\t')
            if prefbias > suffbias:
                lemma = lemma[::-1]
                form = form[::-1]
            prules, srules = prefix_suffix_rules_get(lemma, form)

            if msd not in allprules and len(prules) > 0:
                allprules[msd] = {}
            if msd not in allsrules and len(srules) > 0:
                allsrules[msd] = {}

            for r in prules:
                if (r[0], r[1]) in allprules[msd]:
                    allprules[msd][(r[0], r[1])] = allprules[msd][
                                                       (r[0], r[1])] + 1
                else:
                    allprules[msd][(r[0], r[1])] = 1

            for r in srules:
                if (r[0], r[1]) in allsrules[msd]:
                    allsrules[msd][(r[0], r[1])] = allsrules[msd][
                                                       (r[0], r[1])] + 1
                else:
                    allsrules[msd][(r[0], r[1])] = 1

        # Run eval on dev
        devlines = [line.strip() for line in
                    open(path + lang + f".{test_suffix}", "r", encoding="utf8")
                    if line != '\n']

        numcorrect = 0
        numguesses = 0
        if OUTPUT:
            outfile = open(path + lang + test_suffix + ".out", "w",
                           encoding="utf8")
        for l in devlines:
            # SWITCHED FOR WUG DATA
            # lemma, correct, msd, = l.split(u'\t')
            # lemma, msd, correct = l.split(u'\t')
            lemma, msd = l.split(u'\t')
            if prefbias > suffbias:
                lemma = lemma[::-1]
            outform = apply_best_rule(lemma, msd, allprules, allsrules, num)
            if num is None:
                if prefbias > suffbias:
                    outform = outform[::-1]
                    lemma = lemma[::-1]
                # if outform == correct:
                #     numcorrect += 1
                numguesses += 1
                if OUTPUT:
                    # outfile.write(lemma + "\t" + outform + "\t" + msd + "\n")
                    outfile.write(outform + "\n")
            else:
                if prefbias > suffbias:
                    outform = [o[::-1] for o in outform]
                    lemma = lemma[::-1]
                # if correct in outform:
                #     numcorrect += 1
                numguesses += 1
                if OUTPUT:
                    outfile.write(
                        lemma + "\t" + ", ".join(outform) + "\t" + msd + "\n")

        if OUTPUT:
            outfile.close()

        numlang += 1
        totalavg += numcorrect / float(numguesses)

        print(lang + ": " + str(str(numcorrect / float(numguesses)))[0:7])

    print("Average accuracy", totalavg / float(numlang))


if __name__ == "__main__":
    main(sys.argv)
