# sourceForm sourceMSD # targetMSD => targetForm

import os, sys, json, inspect

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import readdata

# def reformat(trainlist, devlist, devdict,  finname, foutname):
#     with open(finname, 'w') as fin, open(foutname, 'w') as fout:
#         for paradigm, devparadigm in zip(trainlist, devlist):
#             # print(len(paradigm), len(devparadigm))
#             if len(devparadigm) != 0:
#                 # lemmaform, lemmamsd = paradigm[0]
#                 for item in devparadigm:
                    
#                     tgtmsd = item[1]
#                     # tgtform = devdict[item]
#                     tgtform = '-'
#                     output = [letter for letter in tgtform]
#                     for srcform, srcmsd in paradigm:
#                         input = [letter for letter in srcform] \
#                                 + [tag for tag in srcmsd.split(';')] \
#                                 + ['#'] \
#                                 + [tag for tag in tgtmsd.split(';')]
#                         fin.write(' '.join(input) + '\n')
#                         fout.write(' '.join(output) + '\n')


def reformat(trainlist, devlist, devdict, finname, foutname):
    """
    reformat the dev data into the input and output format for Transformer preprocessing

    :param trainlist: [(form, msd)]
    :param devlist: [(lemma, msd)]
    :param devdict: {(lemma, msd): form}
    :param finname: Transformer preprocessing input file
    :param foutname: Transformer preprocessing output file
    :return: None
    """
    with open(finname, 'w') as fin, open(foutname, 'w') as fout:
        for paradigm, devparadigm in zip(trainlist, devlist):
            # print(len(paradigm), len(devparadigm))
            if len(devparadigm) != 0:
                lemmaform, lemmamsd = paradigm[0]
                for item in devparadigm:
                    tgtmsd = item[1]
                    tgtform = devdict[item]
                    output = [letter for letter in tgtform]
                    for srcform, srcmsd in paradigm:
                        input = [letter for letter in srcform] \
                                + [tag for tag in srcmsd.split(';')] \
                                + ['#'] \
                                + [tag for tag in tgtmsd.split(';')]
                        fin.write(' '.join(input) + '\n')
                        fout.write(' '.join(output) + '\n')



if __name__ == "__main__":

    lang = sys.argv[1]

    main_dir    = '../PPI/'
    paradigmdir = f"{main_dir}/paradigms/"

    fname = paradigmdir + lang + '.paradigm'
    trainlist, devlist, testlist = readdata.train_dev_test_list(fname)
    
    outputdir   =  '../PPI/'
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)

    ftestname = '../../../data/' + lang + '.tst'

    testdict = readdata.getDevdict(ftestname)
    
    testin = outputdir + 'test.' + lang + '.input'
    testout = outputdir + 'test.' + lang + '.output'

    reformat(trainlist, testlist, testdict, testin, testout)
