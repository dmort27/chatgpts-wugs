import asyncio
from concurrent.futures import ProcessPoolExecutor
from analogypatternslib import analogy_patterns_2passes
        
async def analogy_patterns_tasks(InFileNames, Workers, MinCover, Varposmin, Varposmax):
    event_loop = asyncio.get_running_loop()
    pool_executor = ProcessPoolExecutor(max_workers=Workers)
    tasks = [
        event_loop.run_in_executor(
            pool_executor,
            analogy_patterns_2passes,
            InFileName, MinCover, Varposmin, Varposmax
        )
        for InFileName in InFileNames
    ]
    await asyncio.gather(*tasks)
    return

def main():
    '''
    Analyses des paramètres du programme et appel de la fonction
    '''
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-d', '--directory', type=str, help='Directory', metavar='DIRECTORY')
    parser.add_argument('-f', '--filenamematch', type=str, help='Pattern', metavar='PATTERN')
    
    parser.add_argument('-w', '--workers', type=int, help='coupoles file', metavar='INTEGER')
    
    parser.add_argument('-c', '--cover', type=int, help='minimum coverage of the subseries with respect to the series', metavar='INTEGER')
    parser.add_argument('-n', '--varposmin', type=int, help='minimum number of variable positions in the patterns', metavar='INTEGER')
    parser.add_argument('-x', '--varposmax', type=int, help='maximum number of variable positions in the patterns', metavar='INTEGER')
    
    # analyse des arguments
    args = parser.parse_args()
    
    # collecte des noms des fichiers
    import fnmatch
    import os
    InFileNames = []
    for file in os.listdir(args.directory):
        if fnmatch.fnmatch(file, args.filenamematch):
            InFileNames.append(args.directory + '/' + file)

    print(InFileNames)
            
    # appel de la fonction
    asyncio.run(analogy_patterns_tasks(InFileNames, args.workers, args.cover, args.varposmin, args.varposmax))

    print('DONE', args)
    return

if __name__ == "__main__":
    main()
