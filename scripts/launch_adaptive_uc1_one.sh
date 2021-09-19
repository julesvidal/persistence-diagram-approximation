#! /bin/bash

if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ] || [ -z "$4" ]
then
                echo "MISSING ARGUMENTS"
                exit
fi

dataSet=$1
eps=$2
threads=$3
NB_TRIES=$4

file=../data/"$dataSet".vti

echo "ADAPTIVE UC1 - $dataSet - eps $eps - $threads threads"
for try in $(eval echo "{1..$NB_TRIES}")
do
                ttkPersistenceDiagramCmd -i $file -o "" -B 1 -A 1 -e $eps -t $threads > ./outputs/adaptive_uc1_"$dataSet"_eps"$eps"_"$threads"threads_try_"$try"
                sleep 1
done
