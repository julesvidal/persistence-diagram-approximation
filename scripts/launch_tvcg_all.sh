#! /bin/bash

threads=$1
NB_TRIES=$2

while read line
do
                toProcess=$(echo $line | cut -d" " -f2)
                if [ "$toProcess" == "1" ]
                then
                        dataSet=$(echo $line | cut -d" " -f1)
                        ./launch_tvcg_one.sh $dataSet $threads $NB_TRIES
                fi
done < ./data_list
