#!/bin/bash

if [[ $# -ne 1 ]]; then
  GPUID=0
else
  GPUID=$1
fi

echo "Run on GPU $GPUID"

PROJECT_ROOT=$(dirname "$(readlink -f "$0")")/..
ROBERTA_large_DIR=$PROJECT_ROOT/checkpoints/roberta.large/model.pt
DATA_ROOT=$PROJECT_ROOT/data

SEED=0
TASK=MNLI
TAG=MTVAT_large

TOTAL_NUM_UPDATES=61936
EPOCH=5          # total epoches
WARMUP_UPDATES=4000      # 6 percent of the number of updates
LR=2e-05                # Peak LR for polynomial LR scheduler.
ADAM_EPS=1e-8
NUM_CLASSES=3
MAX_SENTENCES=32        # Batch size.

MEAN_TEACHER=True
MEAN_TEACHER_AVG=exponential
MT_ALPHA1=0.99
MT_ALPHA2=0.999
MT_RAMPUP=4000

MT_LAMBDA=1
VAT_LAMBDA=100

USE_VAT=True
USE_NOISECP=False
USE_ADVCP=False
VAT_EPS=1e-5
ADVCP_EPS=1e-2
TEACHER_CLASS=kl

OUTPUT=$PROJECT_ROOT/checkpoints/${TASK}/${EPOCH}_${LR}_${TAG}_${SEED}_$(date +%F)_$(date +%H)-$(date +%M)
[ -e $OUTPUT/script  ] || mkdir -p $OUTPUT/script
cp -f $(readlink -f "$0") $OUTPUT/script
rsync -ruzC --exclude-from=$PROJECT_ROOT/.gitignore --exclude 'fairseq' --exclude 'data' $PROJECT_ROOT/ $OUTPUT/src

CUDA_VISIBLE_DEVICES=$GPUID python train.py $DATA_ROOT/$TASK-bin/ \
--mean_teacher_lambda $MT_LAMBDA \
--vat_lambda $VAT_LAMBDA \
--vat_eps $VAT_EPS \
--mean_teacher $MEAN_TEACHER \
--mean_teacher_avg $MEAN_TEACHER_AVG \
--mean_teacher_alpha1 $MT_ALPHA1 \
--mean_teacher_alpha2 $MT_ALPHA2 \
--mean_teacher_rampup $MT_RAMPUP \
--teacher_class $TEACHER_CLASS \
--use_vat $USE_VAT \
--use_noisycopy $USE_NOISECP \
--use_advcopy $USE_ADVCP \
--advcopy_eps $ADVCP_EPS \
--save-dir $OUTPUT \
--restore-file $ROBERTA_large_DIR \
--max-positions 512 \
--max-sentences $MAX_SENTENCES \
--max-tokens 4400 \
--task sentence_prediction_mtvat \
--reset-optimizer --reset-dataloader --reset-meters \
--required-batch-size-multiple 1 \
--init-token 0 --separator-token 2 \
--arch roberta_large \
--criterion sentence_prediction_mtvat \
--num-classes $NUM_CLASSES \
--dropout 0.1 --attention-dropout 0.1 \
--weight-decay 0.1 --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps $ADAM_EPS \
--clip-norm 0.0 \
--lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
--fp16 --fp16-init-scale 4 --threshold-loss-scale 1 --fp16-scale-window 128 \
--max-epoch $EPOCH \
--valid-subset valid,valid1 \
--user-dir ./mymodule \
--best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
--no-last-checkpoints --no-save-optimizer-state \
--find-unused-parameters \
--seed $SEED
