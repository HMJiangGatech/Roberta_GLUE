#!/bin/bash

if [[ $# -ne 1 ]]; then
  GPUID=0
else
  GPUID=$1
fi

echo "Run on GPU $GPUID"

PROJECT_ROOT=$(dirname "$(readlink -f "$0")")/..
ROBERTA_LARGE_DIR=$PROJECT_ROOT/checkpoints/roberta.large/model.pt
DATA_ROOT=$PROJECT_ROOT/data

TASK=QNLI
TAG=Baseline

TOTAL_NUM_UPDATES=33112
EPOCH=10          # total epoches
WARMUP_UPDATES=1986      # 6 percent of the number of updates
LR=1e-05                # Peak LR for polynomial LR scheduler.
NUM_CLASSES=2
MAX_SENTENCES=32        # Batch size.

OUTPUT=$PROJECT_ROOT/checkpoints/${TASK}/${EPOCH}_${LR}_${TAG}_$(date +%F)_$(date +%H)-$(date +%M)
[ -e $OUTPUT/script  ] || mkdir -p $OUTPUT/script
cp -f $(readlink -f "$0") $OUTPUT/script
rsync -ruzC --exclude-from=$PROJECT_ROOT/.gitignore --exclude 'fairseq' --exclude 'data' $PROJECT_ROOT/ $OUTPUT/src

CUDA_VISIBLE_DEVICES=$GPUID python train.py $DATA_ROOT/$TASK-bin/ \
--save-dir $OUTPUT \
--restore-file $ROBERTA_LARGE_DIR \
--max-positions 512 \
--max-sentences $MAX_SENTENCES \
--max-tokens 4400 \
--task sentence_prediction \
--reset-optimizer --reset-dataloader --reset-meters \
--required-batch-size-multiple 1 \
--init-token 0 --separator-token 2 \
--arch roberta_large \
--criterion sentence_prediction \
--num-classes $NUM_CLASSES \
--dropout 0.1 --attention-dropout 0.1 \
--weight-decay 0.1 --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-06 \
--clip-norm 0.0 \
--lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
--fp16 --fp16-init-scale 4 --threshold-loss-scale 1 --fp16-scale-window 128 \
--max-epoch $EPOCH \
--best-checkpoint-metric accuracy --maximize-best-checkpoint-metric;
