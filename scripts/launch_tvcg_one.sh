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

echo "TVCG - $dataSet - $threads threads"
for try in $(eval echo "{1..$NB_TRIES}")
do
                ttkPersistenceDiagramCmd -i $file -o "" -B 1 -t $threads > ./outputs/tvcg_"$dataSet"_"$threads"threads_try_"$try"
                sleep 1
done
