#!/bin/bash

if [[ $# -ne 1 ]]; then
  GPUID=0
else
  GPUID=$1
fi

echo "Run on GPU $GPUID"

if [[ $# -ne 2 ]]; then
  TASK=MNLI
else
  TASK=$2
fi

echo "Run for $TASK"
DIR=checkpoints
DIR="$DIR/$TASK"

for CKPDIR in $DIR/*/
do
    echo "$CKPDIR"
    cat "$CKPDIR/log.txt" | grep "valid on"
    while true ; do
        read -p "predict checkpoint ? [No. of ckp, -1 means not to]" ckp_no
        case $ckp_no in
            [1,2,3,4,5,6,7,8,9,10]* )
                CKP_PT="$CKPDIR/checkpoint$ckp_no.pt"
                echo "Predict CKP No. $ckp_no ($CKP_PT)"
                SAVEDIR="glue_submission/${CKPDIR:12}"
                echo "Save to $SAVEDIR"
                CUDA_VISIBLE_DEVICES=$GPUID python eval_glue.py --task $TASK  --ckp $CKP_PT --savedir $SAVEDIR
                break;;
            [-1]* ) break;;
            *) echo "Please answer Y or n.";;
        esac
    done
done
