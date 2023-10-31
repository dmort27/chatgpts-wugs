# sourceForm sourceMSD # targetMSD => targetForm

import os, sys, json, inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)


import readdata

def reformat(trainlist, finname, foutname):
    """
    convert the training data in paradigms to the format needed for Transformer preprocessing

    :param trainlist: list of paradigms
    :param finname: input data for Transformer
    :param foutname: output data for Transformer
    :return: None
    """
    with open(finname, 'w') as fin, open(foutname, 'w') as fout:
        for paradigm in trainlist:
            # srcform, srcmsd = paradigm[0]
            if len(paradigm) == 1:
                tgtform, tgtmsd = paradigm[0]
                srcform, srcmsd = paradigm[0]
                input = [letter for letter in srcform] \
                        + [tag for tag in srcmsd.split(';')] \
                        + ['#'] \
                        + [tag for tag in tgtmsd.split(';')]
                output = [letter for letter in tgtform]
                fin.write(' '.join(input) + '\n')
                fout.write(' '.join(output) + '\n')
            else:
                for i in range(0, len(paradigm)):
                    tgtform, tgtmsd = paradigm[i]
                    pnow = paradigm[:i] + paradigm[i+1:]
                    for srcform, srcmsd in pnow:
                        input = [letter for letter in srcform] \
                                + [tag for tag in srcmsd.split(';')] \
                                + ['#'] \
                                + [tag for tag in tgtmsd.split(';')]
                        output = [letter for letter in tgtform]
                        fin.write(' '.join(input) + '\n')
                        fout.write(' '.join(output) + '\n')


if __name__ == "__main__":

    lang = sys.argv[1]

    main_dir    = '../PPI/'
    datadir     = f"{main_dir}/paradigms/"

    fname = datadir + lang + '.paradigm'
    trainlist, devlist, testlist = readdata.train_dev_test_list(fname)

    # outputdir = 'one_source/'
    outputdir   = '../PPI/'
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)

    trainin = outputdir + 'train.' + lang + '.input'
    trainout = outputdir + 'train.' + lang + '.output'

    reformat(trainlist, trainin, trainout)
