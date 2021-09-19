#! /bin/bash

if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ]
then
                echo "MISSING ARGUMENTS"
                exit
fi

dataSet=$1
threads=$2
NB_TRIES=$3
file=../data/"$dataSet".vti

echo "FTM - $dataSet - $threads threads"

for try in $(eval echo "{1..$NB_TRIES}")
do
                ttkFTMTreeCmd -i $file -o "" -t $threads -T 0  > ./outputs/ftm_"$dataSet"_"$threads"threads_try_"$try"
                sleep 1
                ttkFTMTreeCmd -i $file -o "" -t $threads -T 1  >> ./outputs/ftm_"$dataSet"_"$threads"threads_try_"$try"
                sleep 1
done
