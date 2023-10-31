import regex
import collections
import difflib
import Levenshtein
import os

#=====================================
# Calcul des signatures

def chardiff(String1, String2):
    '''
    Calculer la différence du nombre d'occurrence de chacun des caractères qui apparaissent dans String1 ou String2
    
    :param: String1 = chaîne de caractères
    :param: String2 = chaîne de caractères
    :result: CharDiff = séquence ordonnée des caractères de String1 et String2 où chaque caractère est suivie d'un point et de la différence.
             Les items sont séparés par des plus.
    '''
    Coll1 = collections.Counter(String1)
    Coll2 = collections.Counter(String2)
    Coll1.subtract(Coll2)
    CharDiff = '+'.join(sorted(['{}.{}'.format(Char, Count) for Char, Count in Coll1.items() if Count]))
    return(CharDiff)
            
def diffsign(A, B):
    '''
    Signature calculée à partir du diff, composée d'une expression qui linéarise le résultat du Matcher et de deux expressions.
    La première est une expression régulière et la seconde est une substitution.
    Les deux expressions décrivent une substitution que l'on peut réaliser en utilisant regex.sub.
    
    :param: A = chaîne de caractères
    :param: B = chaîne de caractères
    :result: Signature = expression qui linéarise le Matcher
    :result: RegEx1 = expression régulière qui correspond à la forme A
    :result: RegEx2 = substitution qui permet de produire la forme B à partir de la forme A
    '''
    S = difflib.SequenceMatcher(None, A, B)
    
    # intialisation des 3 résultats
    Signature = []
    RegEx1 = '^'
    RegEx2 = ''
    
    # numéro du groupe suivant
    GroupNb = 1
    for Tag, Index_A1, Index_A2, Index_B1, Index_B2 in S.get_opcodes():
        # le tag est la première lettre du nom de l'opération en majuscule
        Tag = Tag[0].upper()
        
        # les séquences identiques sont remplacées par une séquence variable (.+) dans RegEx1 et par un numéro de groupe dans RegEx2
        if Tag == 'E':
            Signature.append(Tag)
            RegEx1 += '(.+)'
            RegEx2 += '\\{}'.format(GroupNb)
            GroupNb += 1
        # les séquences différentes alimentent les deux expressions RegEx1 et RegEx2
        else:
            Signature.append('{}.{}.{}'.format(Tag, A[Index_A1:Index_A2], B[Index_B1:Index_B2]))
            # les séquences supprimées dans A sont ajoutées à RegEx1
            if Tag == 'D':
                RegEx1 += regex.escape(A[Index_A1:Index_A2])
            
            # les séquences insérées dans B sont ajoutées à RegEx2
            if Tag == 'I':
                RegEx2 += B[Index_B1:Index_B2]
                
            # les séquences remplacées sont ajoutées aux deux expressions
            if Tag == 'R':
                RegEx1 += regex.escape(A[Index_A1:Index_A2])
                RegEx2 += B[Index_B1:Index_B2]
    
    # ajout de la marque de fin de chaîne à RegEx1
    RegEx1 += '$'
    
    return('+'.join(Signature), RegEx1, RegEx2)


def diffpattern(A, B):
    '''
    Variante de diffsign dans laquelle la deuxième expression est un patron et non la description d'une substitution
    La seule différence est que les références des groupes sont remplacées par des expressions (.+) et que RegEx2 contient des marques de début et de fin de chaîne
    Les deux expressions contiennent le même nombre de groupes
    
    :param: A = chaîne de caractères
    :param: B = chaîne de caractères
    :result: Signature = expression qui linéarise le Matcher
    :result: RegEx1 = expression régulière qui correspond à la forme A
    :result: RegEx2 = expression régulière qui correspond à la forme B 
    '''
    S = difflib.SequenceMatcher(None, A, B)
    # intialisation des 3 résultats
    Signature = []
    RegEx1 = '^'
    RegEx2 = '^'
    
    # numéro du groupe suivant
    GroupNb = 1
    for Tag, Index_A1, Index_A2, Index_B1, Index_B2 in S.get_opcodes():
        # le tag est la première lettre du nom de l'opération en majuscule
        Tag = Tag[0].upper()
        
        # les séquences identiques sont remplacées par une séquence variable (.+) dans RegEx1 et RegEx2
        if Tag == 'E':
            Signature.append(Tag)
            RegEx1 += '(.+)'
            RegEx2 += '(.+)'
            GroupNb += 1
        # les séquences différentes alimentent les deux expressions RegEx1 et RegEx2
        else:
            Signature.append('{}.{}.{}'.format(Tag, A[Index_A1:Index_A2], B[Index_B1:Index_B2]))
            # les séquences supprimées dans A sont ajoutées à RegEx1
            if Tag == 'D':
                RegEx1 += regex.escape(A[Index_A1:Index_A2])
            
            # les séquences insérées dans B sont ajoutées à RegEx2
            if Tag == 'I':
                RegEx2 += B[Index_B1:Index_B2]
                
            # les séquences remplacées sont ajoutées aux deux expressions
            if Tag == 'R':
                RegEx1 += regex.escape(A[Index_A1:Index_A2])
                RegEx2 += B[Index_B1:Index_B2]
    
    # ajout de la marque de fin de chaîne à RegEx1 et RegEx2
    RegEx1 += '$'
    RegEx2 += '$'
    
    return('+'.join(Signature), RegEx1, RegEx2)

#===========================================================================
# numérotation des séquences variables afin de
# pouvoir faire du pattern matiching sur des patterns et
# calculer des patterns de patterns

def number_variable_sequences(RegEx):
    '''
    transformer une regex en une autre qui puisse être utilisée avec regex.subf
    
    :param: RegEx = expression régulière à transformer
    :result: RegEx = la même expression régulière om les groupes sont remplacés par leur identifiants {N} où N est le numéro du groupe   
    '''
    Nb = 1
    while '(.+)' in RegEx:
        RegEx = RegEx.replace('(.+)', '{{{}}}'.format(Nb), 1)
        Nb += 1
    return(RegEx)

def number_variable_sequences_patterns(RegEx1, RegEx2):
    '''
    transformer une regex en une autre qui puisse être utilisée avec regex.subf
    
    :param: RegEx = expression régulière à transformer
    :result: RegEx = la même expression régulière om les groupes sont remplacés par leur identifiants {N} où N est le numéro du groupe   
    '''
    Nb = 1
    while '(.+)' in RegEx1:
        RegEx1 = RegEx1.replace('(.+)', '{}'.format(Nb), 1)
        Nb += 1
    while '(.+)' in RegEx2:
        RegEx2 = RegEx2.replace('(.+)', '{}'.format(Nb), 1)
        Nb += 1
    return(RegEx1, RegEx2)

#=====================================
# matching

def matching(A, B):
    '''
    calculer l'expression régulière qui caractérise un couple de formes pour le calcul des patrons spécifiques à une série de formes.
    Ces expressions sont les duales de celles qui sont utilisées dans les signatures des couples de formes.
    
    :param: A = chaîne de caractères
    :param: B = chaîne de caractères
    :result: RegEx = expression régulière la plus spécifique qui décrit les deux formes
    :result: NbVarPos = nombre de séquences variables dans RegEx.  Plus le nombre est grans, plus l'expression régulière est spécifique
    :result: CharNb = nombre de caractères constants dans RegEx. Plus le nombre est grand, plus l'expression régulière est spécifique
    '''
    S = difflib.SequenceMatcher(None, A, B)
    NbVarPos = 0
    CharNb = 0
    # initialisation de l'expression régulière
    RegEx = '^'
    for Tag, Index_A1, Index_A2, _Index_B1, _Index_B2 in S.get_opcodes():
        Tag = Tag[0].upper()
        
        # les parties identiques sont conservées
        if Tag == 'E':
            RegEx += regex.escape(A[Index_A1:Index_A2])
            NbVarPos += Index_A2 - Index_A1
            CharNb += Index_A2 - Index_A1
        
        # les parties différentes correspondent à des séquences variables
        else:
            RegEx += '(.+)'
            NbVarPos += 1
    # ajout de la marque de fin de séquence
    RegEx += '$'
    
    # print(A, B, S.get_opcodes(), RegEx)
    return(RegEx, NbVarPos, CharNb)

def matching_patterns_2passes(List, MinCoverage, VarPosNbMin, VarPosNbMax):
    '''
    calcul les patrons d'une liste de patrons
    
    :param: List = liste de patrons
    :param: MinCoverage = couverture minimale des patrons spécifiques qui sont conservés
    :param: VarPosNbMin = nombre minimal de séquences variables des patrons spécifiques qui sont conservés.
            Ce nombre est égal à 0 si les expressions sont composées de constantes. Le patron n'a alors aucune généralité
    :param: VarPosNbMax = nombre maximal de séquences variables des patrons spécifiques qui sont conservés.
            Ce nombre est élevé, le patrons est probablement trop spécifique et n'a aucune généralité
    
    '''
    Regexps = set()
    
    # on calcule les patrons de chaque paire de formes de la liste
    for Index1 in range(len(List)):
        for Index2 in range(Index1, len(List)):
            # print(List[Index1], List[Index2], matching(List[Index1], List[Index2]))
            Regexps.add(matching(List[Index1], List[Index2]))
    
    # print('len List', len(List))
    # print('len Regexps', len(Regexps))
    # print('regexps', Regexps)
    
    Coverage = MinCoverage * 0.01
    Matchers = []
    Patterns = []
    for (Regex, NbVarPos, CharNb) in Regexps:
        
        # nombre de séquences variables dans l'expression régulière
        VarPosNb = len(regex.findall(u'\(\.\+\)', Regex))
        
        # si le nombre n'est pas dans la bonne plage, le patron n'est pas conservé
        if VarPosNb < VarPosNbMin or VarPosNb > VarPosNbMax:
            continue
            
        # ensemble des mots qui sont décrits par le patron
        Matches = [Word for Word in List if regex.match(Regex, Word)]

        # print('regex', Regex)
        # print('Matches', Matches)
        
        # si la couverture du patron n'est pas suffisante, le patron n'est pas conservé
        if len(Matches) > Coverage * len(List):
            Matchers.append((Regex, Matches))
            Patterns.append(Regex[1:-1])

    PatternRegex = set()
    for Index1 in range(len(Patterns)):
        for Index2 in range(Index1, len(Patterns)):
            NumberedPatt1, NumberedPatt2 = number_variable_sequences_patterns(Patterns[Index1], Patterns[Index2])
            PatternRegex.add(matching(NumberedPatt1, NumberedPatt2))
            
    for (Regex, NbVarPos, CharNb) in PatternRegex - Regexps :
        
        # nombre de séquences variables dans l'expression régulière
        VarPosNb = len(regex.findall(u'\(\.\+\)', Regex))
        
        # si le nombre n'est pas dans la bonne plage, le patron n'est pas conservé
        if VarPosNb < VarPosNbMin or VarPosNb > VarPosNbMax:
            continue
            
        # ensemble des mots qui sont décrits par le patron
        Matches = [Word for Word in List if regex.match(Regex, Word)]

        # print('regex', Regex)
        # print('Matches', Matches)
        
        # si la couverture du patron n'est pas suffisante, le patron n'est pas conservé
        if len(Matches) > Coverage * len(List):
            Matchers.append((Regex, Matches))

    return(Matchers)

#=====================================
# analogy regexp utils

def analogypatterns_read_pairs(FileName):
    '''
    Lecture des couples formes/étiquettes
    
    :param: FileName = nom du fichier qui contient les couples de formes.  Les entrées ont la forme : Form1, Cat1, Form2, Cat2
    :result: Pairs = liste de tuples (Form1, Cat1, Form2, Cat2)
    '''
    PairsFile = open(FileName)
    Pairs = set()
    for Line in PairsFile:
        try:
            Form1, Cat1, Form2, Cat2  = Line.rstrip('\n').split(' ')
            Pairs.add((Form1, Cat1, Form2, Cat2))
        except ValueError:
            print(Line)
    PairsFile.close()
    return(Pairs)

def analogypatterns_pair_signatures(Pairs):
    '''
    Recalculer la signature analogique de chaque couple
    
    Une signature est un tuple (CharDiff, Dist, DiffSign, Reg1, Reg2, Cat1, Cat2).
    La signature est calculée par diffsign.  Elle correspond à la signature globale (GPP)
     
    :param: Pairs = liste des couples
    :result: Sign_Pairs = dictionnaire qui associe les signatures aux couples qui les ont cette signature
    '''
    Sign_Pairs = {}
    for Pair in Pairs:
        Form1, Cat1, Form2, Cat2 = Pair
        # calcule du patron regex
        (DiffSign, Reg1, Reg2) = diffsign(Form1, Form2)
        # différences des nombres de caractères
        CharDiff = chardiff(Form1, Form2)
        # distance de Levenstein
        Dist = Levenshtein.distance(Form1, Form2, weights=(1, 1, 2))
        Signature = (CharDiff, Dist, DiffSign, Reg1, Reg2, Cat1, Cat2)
        # Pairs_Sign[(Form1, Cat1, Form2, Cat2)] = Signature
        if Signature not in Sign_Pairs:
            Sign_Pairs[Signature] = set()
        Sign_Pairs[Signature].add((Form1, Cat1, Form2, Cat2))
    return(Sign_Pairs)

def analogy_patterns_2passes(InFileName, MinCov, VarPosMin, VarPosMax):
    '''
    Calculer les patrons spacifiques (fins) pour un ensemble de couples
    
    :param: InFileName = fichier qui contient les couples
    :param: MinSiz = nombre minimal de couples connectés par un patron spécifique
    :param: MinCov = couverture minimale des patrons spécifiques relativement à la liste de mots définies par le patron général
    :param: VarPosMin = nombre minimum de positions variables (.+) dans le patron
    :param: VarPosMax = nombre maximum de positions variables (.+) dans le patron
    '''
    # lecture des couples    
    AllPairs = analogypatterns_read_pairs(InFileName)
    
    # calcul des signatures analogiques des couples
    # les signatures analogiques (globales) incluent
    # la signature du matcher difflib et
    # les expressions régulières globales
    Sign_Pairs = analogypatterns_pair_signatures(AllPairs)
    
    # ouverture en écriture du fichier dans lequel les résultats sont écrits
    AnalogyPatterns = open(InFileName + '-'.join(['.regex', '{:03d}'.format(MinCov), '{:1d}'.format(VarPosMin), '{:1d}'.format(VarPosMax)]), 'w')
    # print(Sign_Pairs)
    for Signature in Sign_Pairs:
        _Diff, _Dist, _Pattern, RegExIn, RegExOut, Cat1, Cat2 = Signature
        Pairs = list(Sign_Pairs[Signature])
        # print('signature couples', Signature, Pairs)
        
        # AnalogyPatterns.write('{}\n'.format(Pairs))
        # constitution des listes de mots de la liste de couples
        
        # liste des mots de gauche
        Mots1 = [Form1 for (Form1, _Cat1, _Form2, _Cat2) in Pairs]
        # liste des mots de droite
        Mots2 = [Form2 for (_Form1, _Cat1, Form2, _Cat2) in Pairs]
        
        # print('mots1', Mots1)
        # print('mots2', Mots2)
        
        # liste des patrons de Mots1
        Matchers1 = matching_patterns_2passes(Mots1, MinCoverage=MinCov, VarPosNbMin=VarPosMin, VarPosNbMax=VarPosMax)
        # print('matcher1', Matchers1)

        # liste des patrons de Mots2
        Matchers2 = matching_patterns_2passes(Mots2, MinCoverage=MinCov, VarPosNbMin=VarPosMin, VarPosNbMax=VarPosMax)
        # print('matcher2', Matchers2)
        
        # alignement des patrons de Mots1 et de Mots2
        # on aligne 2 patrons regEx1 et RegEx2 si leur signature est identique à la signature globale de la série (= Signature)
        # notamment, Regex1 et Regex2 doivent avoir le même nombre de séquences (.+)
        for Matcher1 in Matchers1:
            RegEx1, List1 = Matcher1
            for Matcher2 in Matchers2:
                RegEx2, List2 = Matcher2
                # suppression de ^ et de $
                RE1 = RegEx1[1:-1]
                # suppression de ^ et de $
                RE2 = RegEx2[1:-1]
                
                # signature analogique des patrons de mots
                (DiffSignRegEx, Reg1RegEx, Reg2RegEx) = diffsign(RE1, RE2)
                CharDiffRegEx = chardiff(RE1, RE2)
                DistRegEx = Levenshtein.distance(RE1, RE2, weights=(1,1,2))
                SignatureRegEx = (CharDiffRegEx, DistRegEx, DiffSignRegEx, Reg1RegEx, Reg2RegEx, Cat1, Cat2)
                if SignatureRegEx != Signature:
                    continue
                
                # remplacement de (.+) par {Nb} pour permettre son utilisation avec regex.subf
                # les séquences sont renumérotées
                RE2Subf = number_variable_sequences(RE2)
                
                # la signature analogique des patrons de mots et la même que celle des patrons globaux
                # les patrons analogies sont des spécialisation du patron global     
                for Mot1 in List1:
                    for Mot2 in List2:
                        # on vérifie que les deux mots est l'un des couples d'entrées
                        # on vérifie que les deux patrons décrivent une substitution qui permet d'obtenir Mot2 à partir de Mot1
                        if (Mot1, Cat1, Mot2, Cat2) in Pairs and regex.subf(RE1, RE2Subf, Mot1) == Mot2:
                            Match1 = regex.match(RegEx1, Mot1)
                            try:
                                Radical = Match1[1]
                            except IndexError:
                                print('matching failed', Mot1, RegEx1)
                                raise
                            AnalogyPatterns.write('{}\n'.format('\t'.join([Mot1, Cat1, Mot2, Cat2,
                                                                        RegExIn, RegExOut, RegEx1, RegEx2, Radical])))
    AnalogyPatterns.close()
    os.rename(InFileName, f'{InFileName}.DONE')
    return
