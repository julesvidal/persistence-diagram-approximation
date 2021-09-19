#!/bin/bash

NB_TRIES=3

EPS_LIST="0.01 0.05 0.1"

THRD=1

mkdir outputs

bash ./launch_ftm_all.sh $THRD $NB_TRIES 2> /dev/null

bash ./launch_tvcg_all.sh $THRD $NB_TRIES 2> /dev/null

for eps in $EPS_LIST
do
                bash ./launch_adaptive_uc1_all.sh $eps $THRD $NB_TRIES 2> /dev/null
done

