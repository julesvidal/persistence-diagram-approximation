#! /bin/bash

eps=$1
threads=$2
NB_TRIES=$3

while read line
do
                toProcess=$(echo $line | cut -d" " -f2)
                if [ "$toProcess" == "1" ]
                then
                        dataSet=$(echo $line | cut -d" " -f1)
                        ./launch_adaptive_uc1_one.sh $dataSet $eps $threads $NB_TRIES 
                fi
done < ./data_list
