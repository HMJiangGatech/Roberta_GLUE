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

SEED=0

TASK=WSC
TAG=Baseline

TOTAL_NUM_UPDATES=2000  # 10 epochs through RTE for bsz 16
EPOCH=10          # total epoches
WARMUP_UPDATES=250      # 6 percent of the number of updates
LR=2e-05                # Peak LR for polynomial LR scheduler.
MAX_SENTENCES=16        # Batch size.

OUTPUT=$PROJECT_ROOT/checkpoints/${TASK}/${EPOCH}_${LR}_${TAG}_${SEED}
[ -e $OUTPUT/script  ] || mkdir -p $OUTPUT/script
cp -f $(readlink -f "$0") $OUTPUT/script
rsync -ruzC --exclude-from=$PROJECT_ROOT/.gitignore --exclude 'fairseq' --exclude 'data' $PROJECT_ROOT/ $OUTPUT/src

CUDA_VISIBLE_DEVICES=$GPUID python3 train.py $DATA_ROOT/$TASK/ \
--restore-file $ROBERTA_LARGE_DIR \
--max-positions 512 \
--reset-optimizer --reset-dataloader --reset-meters \
--save-dir $OUTPUT \
--no-epoch-checkpoints --no-last-checkpoints --no-save-optimizer-state \
--best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
--valid-subset val \
--fp16 --ddp-backend no_c10d \
--user-dir ./mymodule \
--task wsc --criterion wsc --wsc-cross-entropy \
--arch roberta_large --max-positions 512 --bpe gpt2 \
--dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
--optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-06 \
--lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
--max-epoch $EPOCH \
--max-sentences $MAX_SENTENCES \
--log-format simple --log-interval 100 \
--seed $SEED

