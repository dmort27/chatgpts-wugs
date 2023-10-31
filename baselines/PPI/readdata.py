def train_dev_test_list(fname):
    """
    read the paradigm file and get the list of training slots (form, msd) and dev slots (lemma, msd) in each paradigm

    :param fname: paradigm file
    :return: trainlist is a list of lists, each element list is training slots in a paradigm
             devlist is a list of lists, each element list is dev slots in a paradigm
    """
    trainlist = []
    devlist = []
    testlist = []
    with open(fname) as f:
        paradigm = []
        devparadigm = []
        testparadigm = []
        for line in f:
            if line.strip() == "":
                trainlist.append(paradigm)
                devlist.append(devparadigm)
                testlist.append(testparadigm)
                paradigm = []
                devparadigm = []
                testparadigm = []
            else:
                lemma, form, msd = line.rstrip('\n').split('\t')
                if form != '?' and form != '-' and form != '*':
                    paradigm.append((form, msd))
                if form == '-':
                    devparadigm.append((lemma, msd))
                if form == '?':
                    testparadigm.append((lemma, msd))
        if paradigm != []:
            trainlist.append(paradigm)
            devlist.append(devparadigm)
            testlist.append(testparadigm)
    return trainlist, devlist, testlist

def getDevdict(fname):
    """
    read the shared task dev data and return a dictionary {lemma-msd: form}

    :param fname: shared task dev file
    :return: dev_dict = {(lemma, msd): form}
    """
    dev_dict = {}
    with open(fname) as f:
        for line in f:
            lemma, msd, form = line.rstrip('\n').split('\t')
            lemma = lemma.replace(' ', '_')
            form = form.replace(' ', '_')
            msd = msd.replace(' ', '_')
            dev_dict[(lemma, msd)] = form
    return dev_dict