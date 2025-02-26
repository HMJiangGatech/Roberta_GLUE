#!/bin/bash

if [[ $# -ne 1 ]]; then
  GPUID=0
else
  GPUID=$1
fi

echo "Run on GPU $GPUID"

PROJECT_ROOT=$(dirname "$(readlink -f "$0")")/..
ROBERTA_Large_DIR=$PROJECT_ROOT/checkpoints/roberta.large/model.pt
DATA_ROOT=$PROJECT_ROOT/data

SEED=0
TASK=MERGENLI
TAG=Baseline_Large

TOTAL_NUM_UPDATES=86810
EPOCH=10          # total epoches
WARMUP_UPDATES=3000      # 6 percent of the number of updates
LR=2e-05                # Peak LR for polynomial LR scheduler.
NUM_CLASSES=3
MAX_SENTENCES=8        # Batch size.

OUTPUT=$PROJECT_ROOT/checkpoints/${TASK}/${EPOCH}_${LR}_${TAG}_${SEED}
[ -e $OUTPUT/script  ] || mkdir -p $OUTPUT/script
cp -f $(readlink -f "$0") $OUTPUT/script
rsync -ruzC --exclude-from=$PROJECT_ROOT/.gitignore --exclude 'fairseq' --exclude 'data' $PROJECT_ROOT/ $OUTPUT/src

CUDA_VISIBLE_DEVICES=$GPUID python train.py $DATA_ROOT/$TASK-bin/ \
--save-dir $OUTPUT \
--restore-file $ROBERTA_Large_DIR \
--max-positions 512 \
--max-sentences $MAX_SENTENCES  \
--max-tokens 4400 \
--task sentence_prediction \
--reset-optimizer --reset-dataloader --reset-meters \
--init-token 0 --separator-token 2 \
--arch roberta_large \
--criterion sentence_prediction_mtvat \
--num-classes $NUM_CLASSES \
--dropout 0.1 --attention-dropout 0.1 \
--weight-decay 0.1 --optimizer rmsplus --adam-betas "(0.9, 0.98)" --adam-eps 1e-06 \
--clip-norm 0.0 \
--lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
--fp16 --fp16-init-scale 4 --threshold-loss-scale 1 --fp16-scale-window 128 --memory-efficient-fp16 \
--max-epoch $EPOCH \
--valid-subset valid,valid1,valid2,valid3 \
--user-dir ./mymodule \
--best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
--no-last-checkpoints --no-save-optimizer-state \
--find-unused-parameters \
--seed $SEED --distributed-no-spawn --ddp-backend c10d --num-workers 0 #--required-batch-size-multiple 1 #--update-freq 2

