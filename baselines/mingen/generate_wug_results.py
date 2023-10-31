from pathlib import Path
import pandas as pd

import config
from phtrs import config as phon_config
from rules import *
from phtrs import str_util
import pynini_util

SAVE_DIR = Path(__file__).parent.parent / 'wuggpt'
LANGUAGE = "deu"
SCORE_TYPE = "confidence"
SET = "dev"

MORPHOSYN_TAG_DICT = {
    'eng': ['V;NFIN', 'V;PRS;NOM(3,SG)', 'V;PST', 'V;V.PTCP;PRS', 'V;V.PTCP;PST'],
    'deu': ['V.PTCP;PRS', 'V;IMP;NOM(2,PL)', 'V;IMP;NOM(2,SG)',
            'V;IND;PRS;NOM(1,PL)', 'V;IND;PRS;NOM(1,SG)',
            'V;IND;PRS;NOM(2,PL)', 'V;IND;PRS;NOM(2,SG)',
            'V;IND;PRS;NOM(3,SG)', 'V;IND;PST;NOM(1,PL)',
            'V;IND;PST;NOM(1,SG)', 'V;IND;PST;NOM(2,PL)',
            'V;IND;PST;NOM(3,PL)', 'V;IND;PST;NOM(3,SG)', 'V;NFIN',
            'V;SBJV;PRS;NOM(1,PL)', 'V;SBJV;PST;NOM(1,PL)',
            'V;SBJV;PST;NOM(2,SG)', 'V;SBJV;PST;NOM(3,PL)', 'N;ACC(PL)',
            'N;ACC(SG)', 'N;DAT(PL)', 'N;DAT(SG)', 'N;GEN(PL)', 'N;GEN(SG)',
            'N;NOM(PL)', 'N;NOM(SG)', 'V.PTCP;PST', 'V;IND;PRS;NOM(3,PL)',
            'V;IND;PST;NOM(2,SG)', 'V;SBJV;PRS;NOM(2,PL)',
            'V;SBJV;PRS;NOM(3,PL)', 'V;SBJV;PRS;NOM(3,SG)',
            'V;SBJV;PST;NOM(3,SG)', 'V;SBJV;PRS;NOM(1,SG)',
            'V;SBJV;PST;NOM(1,SG)', 'V;SBJV;PST;NOM(2,PL)',
            'V;SBJV;PRS;NOM(2,SG)'],
    'tam': ['PST-1SG', 'PRS-1SG', 'FUT-1SG', 'PST-2SG', 'PRS-2SG', 'FUT-2SG',
            'PST-3SG.M', 'PRS-3SG.M', 'FUT-3SG.M', 'PST-3SG.F', 'PRS-3SG.F',
            'FUT-3SG.F', 'PST-3SG.HON', 'PRS-3SG.HON', 'FUT-3SG.HON',
            'PST-1PL', 'PRS-1PL', 'FUT-1PL', 'PST-2PL', 'PRS-2PL', 'FUT-2PL',
            'PST-3PL', 'PRS-3PL', 'FUT-3PL'],
    'tur': ['Verb;Pos;Past;A1sg', 'Verb;Neg;Narr;A2pl', 'Noun;A3sg;P1sg;Dat', 'Noun;A3sg;Pnon;Acc']
}

SEG_FIX_DICT = {
    'eng': {'eɪ': 'e', 'oʊ': 'o', 'əʊ': 'o', 'aɪ': 'a ɪ', 'aʊ': 'a ʊ',
            'ɔɪ': 'ɔ ɪ', 'ɝ': 'ɛ ɹ', 'ˠ': '', 'm̩': 'm', 'n̩': 'n', 'l̩': 'l',
            'ɜ': 'ə', 'uːɪ': 'uː ɪ', 'ɔ̃': 'ɔ', 'ː': '', 'r': 'ɹ', 'ɡ': 'g',
            'd͡ʒ': 'dʒ', 't͡ʃ': 'tʃ'},
    'deu': {'ai̯': 'a i', 'au̯': 'a u', 'oi̯': 'o i',
            'iːə': 'iː ə', 'eːə': 'eː ə', 'ɛːə': 'ɛː ə', 'ɡ': 'g',
            'p͡f': 'pf', 't͡s': 'ts', 't͡ʃ': 'tʃ', 'ç': 'ç'},
    'tam': {'t͡ʃ': 'tʃ'},
    'tur': {'t͡ʃ': 'tʃ', 'd͡ʒ': 'dʒ', 'ɡ': 'g'}
}


def format_strings(dat, extra_seg_fixes=None):
    seg_fixes = config.seg_fixes
    if extra_seg_fixes is not None:
        seg_fixes = seg_fixes | extra_seg_fixes

    # Fix transcriptions (conform to phonological feature set)
    dat['stem'] = [str_util.retranscribe(x, seg_fixes) \
                   for x in dat['stem']]
    dat['output'] = [str_util.retranscribe(x, seg_fixes) \
                     for x in dat['output']]
    dat['stem'] = [str_util.add_delim(x) for x in dat['stem']]
    return dat


morphosyn_tags = MORPHOSYN_TAG_DICT[LANGUAGE]

wugs = pd.read_csv(f'../../../../data/{LANGUAGE}_IPA.{SET}', sep='\t',
                   names=['stem', 'output', 'morphosyn', 'stem_orth', 'output_orth'])

formatted = False
wug_ratings = []
for morphosyn_tag in morphosyn_tags:
    config_save = pd.read_pickle(
        open(SAVE_DIR / f'{LANGUAGE}/{morphosyn_tag}/{LANGUAGE}_config.pkl', 'rb'))
    for key, val in config_save.items():
        setattr(config, key, val)
    phon_config.init(config_save)
    config.seg_fixes = SEG_FIX_DICT[LANGUAGE]

    if not formatted:
        wugs = format_strings(wugs)
        formatted = True
    wugs_tmp = wugs[wugs['morphosyn'] == morphosyn_tag]
    print(morphosyn_tag, len(wugs_tmp))

    syms = [x for x in config.sym2ftrs]
    sigstar, symtable = pynini_util.sigstar(syms)
    stems = [str(x) for x in wugs_tmp['stem']]
    wordforms = stems

    rules = pd.read_csv(
        SAVE_DIR / f'{LANGUAGE}/{morphosyn_tag}/{LANGUAGE}_rules_pruned_{SCORE_TYPE}.tsv',
        sep='\t')

    R = [FtrRule.from_str(rule) for rule in rules['rule']]
    R = [(rule, score, idx) for (rule, score, idx) \
         in zip(R, rules[SCORE_TYPE], rules['rule_idx'])]

    max_rating = {}
    max_rule = {}
    max_pred = {}
    print('iter')
    for i, rule_ in enumerate(R):
        if i % 500 == 0:
            print(i)

        # Convert rule to segment regexes
        rule, score, rule_idx = rule_
        (A, B, C, D) = rule.regexes()

        # Subset of wug data s.t. CAD occurs in input
        CAD = [Z for Z in [C, A, D] if Z != '∅']
        CAD = ' '.join(CAD)
        if CAD != '':
            subdat = [wf for wf in wordforms if re.search(CAD, wf)]
        else:
            subdat = wordforms
        if len(subdat) == 0:
            continue

        # Compile rule to FST
        rule_fst = pynini_util.compile_rule(A, B, C, D, sigstar, symtable)

        for stem in subdat:
            pred = pynini_util.apply_rule(rule_fst, stem, sigstar, symtable)
            if stem not in max_rating \
                    or score > max_rating[stem]:
                max_rating[stem] = score
                max_rule[stem] = rule_
                max_pred[stem] = pred

    print()
    for forms, rating in max_rating.items():
        stem = forms
        rule, score, rule_idx = max_rule[stem]
        pred = max_pred[stem]
        wug_ratings.append((stem, pred, rating, rule_idx, morphosyn_tag))

wug_ratings = pd.DataFrame(
    wug_ratings,
    columns=['stem', 'pred', 'model_rating', 'rule_idx', 'morphosyn'])
wug_ratings = wug_ratings.merge(wugs, how='right')

wug_results = wug_ratings[['stem', 'output', 'pred', 'morphosyn', 'stem_orth', 'output_orth']]
wug_results['pred'] = wug_ratings['pred'].str[2:-2]
wug_results.to_csv(
    SAVE_DIR /
    f'results/{LANGUAGE}_{SET}_results.tsv',
    sep='\t',
    index=False
)

print((wug_results['output'] == wug_results['pred']).sum() / len(wug_results))
