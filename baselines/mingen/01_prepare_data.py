# -*- coding: utf-8 -*-

import configargparse, pickle, re, sys
from pathlib import Path
import pandas as pd

# sys.path.append(str(Path('../../phtrs')))

import config
from phtrs import config as phon_config, features, str_util


# Select language and transcription conventions
parser = configargparse.ArgParser(
    config_file_parser_class=configargparse.YAMLConfigFileParser)
parser.add(
    '--language',
    type=str,
    choices=['eng', 'deu', 'tur', 'tam'],
    default='eng')
parser.add(
    '--morphosyn',
    type=str,
    default=None
)
args = parser.parse_args()
LANGUAGE = args.language
MORPHOSYN = args.morphosyn

# String environment
config.epsilon = 'ϵ'
config.bos = '⋊'
config.eos = '⋉'
config.zero = '∅'
config.save_dir = Path(__file__).parent.parent / f'wuggpt/{LANGUAGE}/{MORPHOSYN}'
phon_config.init(config)


def format_strings(dat, extra_seg_fixes=None):
    seg_fixes = config.seg_fixes
    if extra_seg_fixes is not None:
        seg_fixes = seg_fixes | extra_seg_fixes

    # Fix transcriptions (conform to phonological feature set)
    dat['stem'] = [str_util.retranscribe(x, seg_fixes) \
        for x in dat['wordform1']]
    dat['output'] = [str_util.retranscribe(x, seg_fixes) \
        for x in dat['wordform2']]
    dat['stem'] = [str_util.add_delim(x) for x in dat['stem']]
    dat['output'] = [str_util.add_delim(x) for x in dat['output']]

    # Remove prefix from output
    if config.remove_prefix is not None:
        dat['output'] = [re.sub('⋊ ' + config.remove_prefix, '⋊', x) \
            for x in dat['output']]
    return dat


ddata = Path(__file__).parent.parent.parent.parent / 'data'

if LANGUAGE == 'eng':
    wordform_omit = None
    wug_morphosyn = MORPHOSYN
    # Simplify or split diphthongs, zap diacritics, fix unicode
    config.seg_fixes = {
      'eɪ': 'e', 'oʊ': 'o', 'əʊ': 'o', 'aɪ': 'a ɪ', 'aʊ': 'a ʊ', \
      'ɔɪ': 'ɔ ɪ', 'ɝ': 'ɛ ɹ', 'ˠ': '', 'm̩': 'm', 'n̩': 'n', 'l̩': 'l', \
      'ɜ': 'ə', 'uːɪ': 'uː ɪ', 'ɔ̃': 'ɔ', 'ː': '', 'r': 'ɹ', 'ɡ': 'g',
        'd͡ʒ': 'dʒ', 't͡ʃ': 'tʃ'}
    config.remove_prefix = None

if LANGUAGE == 'deu':
    # wordform_omit = '[+]'
    wordform_omit = None
    wug_morphosyn = MORPHOSYN  # 'N;NOM(PL)'
    # Split diphthongs, fix unicode
    config.seg_fixes = {'ai̯': 'a i', 'au̯': 'a u', 'oi̯': 'o i',
                        'iːə': 'iː ə', 'eːə': 'eː ə', 'ɛːə': 'ɛː ə', 'ɡ': 'g',
                        'p͡f': 'pf', 't͡s': 'ts', 't͡ʃ': 'tʃ', 'ç': 'ç'}
    # config.remove_prefix = 'g ə'
    config.remove_prefix = None

if LANGUAGE == 'tam':
    wordform_omit = None
    wug_morphosyn = MORPHOSYN
    config.seg_fixes = {'t͡ʃ': 'tʃ'}
    config.remove_prefix = None

if LANGUAGE == 'tur':
    wordform_omit = None
    wug_morphosyn = MORPHOSYN
    config.seg_fixes = {'t͡ʃ': 'tʃ', 'd͡ʒ': 'dʒ', 'ɡ': 'g'}
    config.remove_prefix = None


# # # # # # # # # #
# Train
fdat = ddata / f'{LANGUAGE}_IPA.trn'
dat = pd.read_csv(fdat, sep='\t', \
    names=['wordform1', 'wordform2', 'morphosyn',
           'wordform1_orth', 'wordform2_orth'])

# Filter rows by characters in wordforms
if wordform_omit is not None:
    dat = dat[~(dat.wordform1.str.contains(wordform_omit))]
    dat = dat[~(dat.wordform2.str.contains(wordform_omit))]
    dat = dat.reset_index()
print(dat)

# Keep rows with wug-tested morphosyn xxx could be list
dat = dat[dat['morphosyn'] == wug_morphosyn]
dat = dat.drop(columns='morphosyn')
dat = dat.drop_duplicates().reset_index()

# Format strings and save
dat = format_strings(dat)
dat.to_csv(config.save_dir / f'{LANGUAGE}_dat_train.tsv', sep='\t', index=False)
config.dat_train = dat
print('Training data')
print(dat)
print()

# # # # # # # # # #
# Wug dev
WUG_DEV = LANGUAGE
fwug_dev = ddata / f'{LANGUAGE}_IPA.dev'
wug_dev = pd.read_csv(
    fwug_dev,
    sep='\t',
    names=['wordform1', 'wordform2', 'morphosyn', 'wordform1_orth', 'wordform2_orth'])
wug_dev = wug_dev[wug_dev['morphosyn'] == wug_morphosyn]
wug_dev = wug_dev.drop(columns='morphosyn')

wug_dev = format_strings(wug_dev)
config.wug_dev = wug_dev
wug_dev.to_csv(
    config.save_dir / f'{LANGUAGE}_wug_dev.tsv', sep='\t', index=False)
print('Wug dev data')
print(wug_dev)
print()

# # # # # # # # # #
# Wug tst
WUG_TST = LANGUAGE
fwug_tst = ddata / f'{LANGUAGE}_IPA.tst'

wug_tst = pd.read_csv(
    fwug_tst, sep='\t', names=['wordform1', 'wordform2', 'morphosyn', 'wordform1_orth', 'wordform2_orth'])
wug_tst = wug_tst[wug_tst['morphosyn'] == wug_morphosyn]
wug_tst = wug_tst.drop(columns='morphosyn')

wug_tst = format_strings(wug_tst)
config.wug_tst = wug_tst
wug_tst.to_csv(
    config.save_dir / f'{LANGUAGE}_wug_tst.tsv', sep='\t', index=False)
print('Wug test data')
print(wug_tst)
print()

# # # # # # # # # #
# Phonological features
segments = set()
for stem in dat['stem']:
    segments |= set(stem.split())
for output in dat['output']:
    segments |= set(output.split())
segments -= {config.bos, config.eos}
segments = [x for x in segments]
segments.sort()
print(f'Segments that appear in training data: '
      f'{segments} (n = {len(segments)})')
print()

# Import features from file
feature_matrix = features.import_features(
    Path(__file__).parent.parent / f'wuggpt/universal.ftr',
    segments)

# Fix up features for mingen
ftr_matrix = feature_matrix.ftr_matrix
ftr_matrix = ftr_matrix.drop(columns='sym')  # Redundant with X (Sigma*)
config.phon_ftrs = ftr_matrix
config.ftr_names = list(ftr_matrix.columns.values)
config.syms = list(ftr_matrix.index)

# Map from symbols to feature-value dictionaries and feature vectors
config.sym2ftrs = {}
config.sym2ftr_vec = {}
for i, sym in enumerate(config.syms):
    ftrs = config.phon_ftrs.iloc[i, :].to_dict()
    config.sym2ftrs[sym] = ftrs
    config.sym2ftr_vec[sym] = tuple(ftrs.values())

# # # # # # # # # #
# Save config
config_save = {}
for key in dir(config):
    if re.search('__', key):
        continue
    config_save[key] = getattr(config, key)

with open(config.save_dir / f'{LANGUAGE}_config.pkl', 'wb') as f:
    pickle.dump(config_save, f)
