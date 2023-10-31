# -*- coding: utf-8 -*-
# Convenience functions operating on string on list of strings

import re
from . import config as phon_config


def squish(x):
    """ Collapse consecutive spaces, remove leading/trailing spaces. """
    # see: https://stringr.tidyverse.org/reference/str_trim.html
    if isinstance(x, list):
        return [squish(xi) for xi in x]
    y = re.sub('[ ]+', ' ', x)
    return y.strip()


def sep_chars(x):
    """ Separate characters with space. """
    # see: torchtext.data.functional.simple_space_split
    if isinstance(x, list):
        return [sep_chars(xi) for xi in x]
    return ' '.join(x)


def add_delim(x, sep=False, edge='both'):
    """ Add begin/end delimiters to space-separated string. """
    if isinstance(x, list):
        return [add_delim(xi, sep) for xi in x]
    if sep:
        x = ' '.join(x)
    #y = [phon_config.bos] + [xi for xi in x.split(' ')] + [phon_config.eos]
    #return ' '.join(y)
    if edge == 'begin':
        y = f'{phon_config.bos} {x}'
    elif edge == 'end':
        y = f'{x} {phon_config.eos}'
    else:  # edge == 'both'
        y = f'{phon_config.bos} {x} {phon_config.eos}'
    return y


def remove_delim(x):
    """ Remove begin/end delimiters. """
    if isinstance(x, list):
        return [remove_delim(xi) for xi in x]
    y = re.sub(f'{phon_config.bos}', '', x)
    y = re.sub(f'{phon_config.eos}', '', y)
    return squish(y)


def remove(x, syms):
    """ Remove designated symbols. """
    if isinstance(x, list):
        return [remove(xi, syms) for xi in x]
    y = x
    for sym in syms:
        y = re.sub(sym, '', y)
    return squish(y)


def retranscribe(x, subs):
    """ Change segment transcription by applying substitutions. """
    if isinstance(x, list):
        return [retranscribe(xi, subs) for xi in x]
    y = x
    for s, r in subs.items():
        y = re.sub(s, r, y)
    return squish(y)


def lcp(x, y, prefix=True):
    """
    Longest common prefix (or suffix) of two segment sequences.
    """
    if x == y:
        return x
    if not prefix:
        x = x[::-1]
        y = y[::-1]
    n_x, n_y = len(x), len(y)
    n = max(n_x, n_y)
    for i in range(n + 1):
        if i >= n_x:
            match = x
            break
        if i >= n_y:
            match = y
            break
        if x[i] != y[i]:
            match = x[:i]
            break
    if not prefix:
        match = match[::-1]
    return match


def test():
    phono_config.init({'epsilon': '<eps>', 'bos': '>', 'eos': '<'})
    print(phon_config.bos)
    print(phon_config.eos)


if __name__ == "__main__":
    test()
