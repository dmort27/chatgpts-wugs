#!/bin/bash

lang="deu"

if [ $lang == "deu" ];
then
  declare -a tags=('V.PTCP;PRS' 'V;IMP;NOM(2,PL)' 'V;IMP;NOM(2,SG)'
         'V;IND;PRS;NOM(1,PL)' 'V;IND;PRS;NOM(1,SG)'
         'V;IND;PRS;NOM(2,PL)' 'V;IND;PRS;NOM(2,SG)'
         'V;IND;PRS;NOM(3,SG)' 'V;IND;PST;NOM(1,PL)'
         'V;IND;PST;NOM(1,SG)' 'V;IND;PST;NOM(2,PL)'
         'V;IND;PST;NOM(3,PL)' 'V;IND;PST;NOM(3,SG)' 'V;NFIN'
         'V;SBJV;PRS;NOM(1,PL)' 'V;SBJV;PST;NOM(1,PL)'
         'V;SBJV;PST;NOM(2,SG)' 'V;SBJV;PST;NOM(3,PL)' 'N;ACC(PL)'
         'N;ACC(SG)' 'N;DAT(PL)' 'N;DAT(SG)' 'N;GEN(PL)' 'N;GEN(SG)'
         'N;NOM(PL)' 'N;NOM(SG)' 'V.PTCP;PST' 'V;IND;PRS;NOM(3,PL)'
         'V;IND;PST;NOM(2,SG)' 'V;SBJV;PRS;NOM(2,PL)'
         'V;SBJV;PRS;NOM(3,PL)' 'V;SBJV;PRS;NOM(3,SG)'
         'V;SBJV;PST;NOM(3,SG)' 'V;SBJV;PRS;NOM(1,SG)'
         'V;SBJV;PST;NOM(1,SG)' 'V;SBJV;PST;NOM(2,PL)'
         'V;SBJV;PRS;NOM(2,SG)')
fi

if  [ $lang == "eng" ];
then
  declare -a tags=('V;NFIN' 'V;PRS;NOM(3,SG)' 'V;PST' 'V;V.PTCP;PRS' 'V;V.PTCP;PST')
fi

if  [ $lang == "tam" ];
then
  declare -a tags=('PST-1SG' 'PRS-1SG' 'FUT-1SG' 'PST-2SG' 'PRS-2SG' 'FUT-2SG'
            'PST-3SG.M' 'PRS-3SG.M' 'FUT-3SG.M' 'PST-3SG.F' 'PRS-3SG.F'
            'FUT-3SG.F' 'PST-3SG.HON' 'PRS-3SG.HON' 'FUT-3SG.HON'
            'PST-1PL' 'PRS-1PL' 'FUT-1PL' 'PST-2PL' 'PRS-2PL' 'FUT-2PL'
            'PST-3PL' 'PRS-3PL' 'FUT-3PL')
fi

if  [ $lang == "tur" ];
then
  declare -a tags=('Verb;Pos;Past;A1sg' 'Verb;Neg;Narr;A2pl' 'Noun;A3sg;P1sg;Dat' 'Noun;A3sg;Pnon;Acc')
fi

for tag in "${tags[@]}"
do
  python 01_prepare_data.py --language deu --morphosyn "$tag"
  python 02_run_model.py --language deu --morphosyn "$tag" --score_rules --prune_rules
done
