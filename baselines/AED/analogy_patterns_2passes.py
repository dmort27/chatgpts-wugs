from analogypatternslib import analogy_patterns_2passes
    
def main():
    '''
    Analyses des paramètres du programme et appel de la fonction
    '''
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-i', '--input', type=str, help='coupoles file', metavar='FILE')
    parser.add_argument('-c', '--cover', type=int, help='minimum coverage of the subseries with respect to the series', metavar='INTEGER')
    parser.add_argument('-n', '--varposmin', type=int, help='minimum number of variable positions in the patterns', metavar='INTEGER')
    parser.add_argument('-x', '--varposmax', type=int, help='maximum number of variable positions in the patterns', metavar='INTEGER')
    
    # analyse des arguments
    args = parser.parse_args()
    
    # appel de la fonction
    analogy_patterns_2passes(args.input, args.cover, args.varposmin, args.varposmax)
    return
   
if __name__ == '__main__':
    main()
