#!/bin/bash

if [[ $# -ne 1 ]]; then
  echo "Run as following:"
  echo "./examples/roberta/clean_ANLI.sh <data_folder>"
  exit 1
fi

FOLDER=$1
mv $FOLDER/anli_v0.1 $FOLDER/ANLI
TASK_FOLDER=$FOLDER/ANLI
cat $TASK_FOLDER/R1/train.jsonl $TASK_FOLDER/R2/train.jsonl $TASK_FOLDER/R3/train.jsonl > $TASK_FOLDER/train.jsonl
cat $TASK_FOLDER/R1/dev.jsonl $TASK_FOLDER/R2/dev.jsonl $TASK_FOLDER/R3/dev.jsonl > $TASK_FOLDER/dev.jsonl
cat $TASK_FOLDER/R1/test.jsonl $TASK_FOLDER/R2/test.jsonl $TASK_FOLDER/R3/test.jsonl > $TASK_FOLDER/test.jsonl
cp $TASK_FOLDER/test.jsonl $TASK_FOLDER/dev_t.jsonl

for ROUND in R1 R2 R3
do
  cp $TASK_FOLDER/${ROUND}/dev.jsonl $TASK_FOLDER/dev_${ROUND}.jsonl
  cp $TASK_FOLDER/${ROUND}/test.jsonl $TASK_FOLDER/dev_${ROUND}_t.jsonl
done

python preprocess_superGLUE_data.py --tasks ANLI --data_dir $FOLDER
