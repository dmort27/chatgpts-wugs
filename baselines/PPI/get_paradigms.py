# convert the shared task data into paradigms

import os
import sys
from collections import defaultdict

def _oneFile2paradigm(fname, lemma_paradigm, lemma_paradigm_filled, msdlist, train=False):
    count = 0
    with open(fname) as f:
        for line in f:
            lines = line.strip().split('\t')
            count += 1
            if len(lines) == 2:
                lemma, msd = lines
                form = '?'
            else:
                lemma, msd, form = lines
            if not train:
                form = '?'
            # lemma = lemma.strip().replace(' ', '_')
            # form = form.strip().replace(' ', '_')
            # msd = msd.strip().replace(' ', '_')
            lemma = lemma.replace(' ', '_')
            form = form.replace(' ', '_')
            msd = msd.replace(' ', '_')

            lemma_paradigm_filled[lemma].append((form, msd))
            if '.dev' in fname:
                form = '-'
            lemma_paradigm[lemma].append((form, msd))
            msdlist.append(msd)
    if '.trn' in fname:
        print('train #:', count)
    elif '.dev' in fname:
        print('dev #:', count)
    else:
        print('tst #:', count)
    return lemma_paradigm, lemma_paradigm_filled, msdlist


def files2paradigm(ftrn_name, fdev_name, ftst_name):
    lemma_paradigm = defaultdict(list)
    lemma_paradigm_filled = defaultdict(list)
    msdlist = []
    lemma_paradigm, lemma_paradigm_filled, msdlist = _oneFile2paradigm(ftrn_name, lemma_paradigm, lemma_paradigm_filled, msdlist, train=True)
    lemma_paradigm, lemma_paradigm_filled, msdlist = _oneFile2paradigm(fdev_name, lemma_paradigm, lemma_paradigm_filled, msdlist, train=False)
    lemma_paradigm, lemma_paradigm_filled, msdlist = _oneFile2paradigm(ftst_name, lemma_paradigm, lemma_paradigm_filled, msdlist, train=False)
    msdlist = sorted(list(set(msdlist)))
    pos_msd_dict = defaultdict(list)
    for msd in msdlist:
        pos = msd.split(';')[0]
        if '.' in pos:
            pos = pos.split('.')[0]
        pos_msd_dict[pos].append(msd)
    for k, v in pos_msd_dict.items():
        print('->', k, len(v), v)
    return lemma_paradigm, lemma_paradigm_filled, pos_msd_dict

def _paradigm2output(paradigm, foutname, pos_msd_dict):
    paradigmcount = 0
    with open(foutname, 'w') as fout:
        for lemma, forms in paradigm.items():
            msd_form_dict = {}
            poslist = []
            for form, msd in forms:
                msd_form_dict[msd] = form
                pos = msd.strip().split(';')[0]
                if '.' in pos:
                    pos = pos.split('.')[0]
                if pos not in poslist:
                    poslist.append(pos)

            for posnow in poslist:
                canonicalform = lemma
                canonicalmsd = posnow+';CANONICAL'
                fout.write('\t'.join([canonicalform, canonicalform, canonicalmsd]) + '\n')
                msdlist = pos_msd_dict[posnow]
                for msd in msdlist:
                    if msd in msd_form_dict:
                        form = msd_form_dict[msd]
                    else:
                        form = '*'
                    fout.write('\t'.join([lemma, form, msd]) + '\n')
                fout.write('\n')
                paradigmcount += 1
    print('paradigm #:', paradigmcount)

def generate_paradigms(lemma_paradigm, lemma_paradigm_filled, msdlist, lang):
    paradigm_dir = 'paradigms/'
    if not os.path.exists(paradigm_dir):
        os.makedirs(paradigm_dir)
    fparadigm_name = os.path.join(paradigm_dir, lang+'.paradigm')
    fparadigm_filled_name = os.path.join(paradigm_dir, lang + '.paradigm.filled')
    _paradigm2output(lemma_paradigm, fparadigm_name, msdlist)
    # _paradigm2output(lemma_paradigm_filled, fparadigm_filled_name, msdlist)

def processDir(dirname, count):
    for lang in ['tur', 'tam', 'deu', 'eng']:
        count += 1
        print('ID:', count, '... processing ...', dirname, '...', lang, '...')
        ftrn_name = os.path.join(dirname, lang+'.trn')
        fdev_name = os.path.join(dirname, lang + '.dev')
        ftst_name = os.path.join(dirname, lang + '.tst')
        lemma_paradigm, lemma_paradigm_filled, msdlist = files2paradigm(ftrn_name, fdev_name, ftst_name)
        generate_paradigms(lemma_paradigm, lemma_paradigm_filled, msdlist, lang)
    return count

if __name__ == "__main__":
    known_dir = '../../data/'

    langcount = 0
    langcount = processDir(known_dir, langcount)
    