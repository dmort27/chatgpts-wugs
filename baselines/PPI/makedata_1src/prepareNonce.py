# sourceForm sourceMSD # targetMSD => targetForm

import os, sys, json, inspect


def reformat(infile, outfile, noncefile):

    fin = open(infile, 'w')
    fout = open(outfile, 'w')
    fnonce = open(noncefile, 'r')

    lines = fnonce.readlines()

    for line in lines:
        srcform, tgtmsd = line.strip().split('\t')
        tgtmsd          = tgtmsd.upper()
        pos_tag         = tgtmsd.split(';')[0]
        srcmsd          = f'{pos_tag};CANONICAL'
        input           = [letter for letter in srcform] \
                                + [tag for tag in srcmsd.split(';')] \
                                + ['#'] \
                                + [tag for tag in tgtmsd.split(';')]
        fin.write(' '.join(input) + '\n')
        fout.write('\n')

    pass



if __name__ == "__main__":

    lang = sys.argv[1]
    
    noncefile   = f'../../../data/{lang}.nonce'

    outputdir   = '../PPI/'
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)

    testin = outputdir + 'test.' + lang + '.input'
    testout = outputdir + 'test.' + lang + '.output'

    reformat(testin, testout, noncefile)