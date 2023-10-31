#!/bin/bash

# Usage:
# ./runAll.sh 1src

set -euo pipefail

TYPE=1src

for LANGUAGE in tam deu eng tur; 
do
    echo $LANGUAGE
    echo ==========$TYPE========== >> results_$TYPE.txt        
    echo "----start time----" >> results_$TYPE.txt

    date >> results_$TYPE.txt

    echo "... preprocessing data ..."
    ./preprocess.sh $LANGUAGE $TYPE

    mkdir -p "data-bin-${TYPE}/"
    cp    -rf "data-bin/${LANGUAGE}"* "data-bin-${TYPE}/"

    for SEED in 0 1 2 3 4 5 6 7 8 9;
    do
        # train for 10000 updates, select best five model on dev so far
        echo "... training models ..."
        # ./train.sh $LANGUAGE 10000 $SEED
        ./train.sh $LANGUAGE 10000 $SEED

        echo "... generating and evaluating for dev set ..."
        ./generate.sh $LANGUAGE dev $TYPE

        # continue training for another 10000 updates, select best five model on dev so far
        echo "... training models ..."
        # ./train.sh $LANGUAGE 20000
        ./train.sh $LANGUAGE 20000 $SEED

        echo "... generating and evaluating for dev set ..."
        ./generate.sh $LANGUAGE dev $TYPE

        
        # generate for test data for all languages except tam

        if [ $LANGUAGE != "tam" ]
        then
            echo "... generating and evaluating for test set ..."
            ./generate.sh $LANGUAGE test $TYPE
        fi
        
        mkdir -p "checkpoints-${TYPE}/${LANGUAGE}-${SEED}/"

        cp -rf "checkpoints/${LANGUAGE}"* "checkpoints-${TYPE}/${LANGUAGE}-${SEED}/"
        # cp -rf "data-bin/${LANGUAGE}"* "data-bin-${TYPE}/${LANGUAGE}-${SEED}/"

        rm -rf "checkpoints/${LANGUAGE}"*
        
        echo "----end time----" >> results_$TYPE.txt

        date >> results_$TYPE.txt
    done
done

