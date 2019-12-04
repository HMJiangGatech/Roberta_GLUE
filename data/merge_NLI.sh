#!/bin/bash

if [[ $# -ne 1 ]]; then
  echo "Run as following:"
  echo "./examples/roberta/merge_NLI.sh <data_folder>"
  exit 1
fi

FOLDER=$1
mkdir -p $FOLDER/MERGENLI/processed

for FILE in train.input0 train.input1 train.label
do
    cat \
    $FOLDER/MNLI/processed/$FILE $FOLDER/ANLI/processed/$FILE $FOLDER/ANLI/processed/$FILE $FOLDER/ANLI/processed/$FILE  > $FOLDER/MERGENLI/processed/$FILE
done
for FILEPRE in dev_matched dev_mismatched
do
    for FILEPOST in input0 input1 label
    do
        cp $FOLDER/MNLI/processed/${FILEPRE}.${FILEPOST} $FOLDER/MERGENLI/processed/${FILEPRE}_MNLI.${FILEPOST}
    done
done
for FILEPRE in test_matched test_mismatched
do
    for FILEPOST in input0 input1
    do
        cp $FOLDER/MNLI/processed/${FILEPRE}.${FILEPOST} $FOLDER/MERGENLI/processed/${FILEPRE}_MNLI.${FILEPOST}
    done
done
for FILEPRE in dev dev_t
do
    for FILEPOST in input0 input1 label
    do
        cp $FOLDER/ANLI/processed/${FILEPRE}.${FILEPOST} $FOLDER/MERGENLI/processed/${FILEPRE}_ANLI.${FILEPOST}
    done
done
for FILEPRE in dev dev_t
do
    for FILEPOST in input0 input1 label
    do
        cp $FOLDER/SNLI/processed/${FILEPRE}.${FILEPOST} $FOLDER/MERGENLI/processed/${FILEPRE}_SNLI.${FILEPOST}
    done
done

rm -rf MERGENLI-bin
TASKFOLDER=$FOLDER/MERGENLI/processed
DEVPREF="$TASKFOLDER/dev_matched_MNLI.LANG,$TASKFOLDER/dev_mismatched_MNLI.LANG,$TASKFOLDER/dev_ANLI.LANG,$TASKFOLDER/dev_t_ANLI.LANG,$TASKFOLDER/dev_SNLI.LANG,$TASKFOLDER/dev_t_SNLI.LANG"
TESTPREF="$TASKFOLDER/test_matched_MNLI.LANG,$TASKFOLDER/test_mismatched_MNLI.LANG"

# Run fairseq preprocessing:
for INPUT_TYPE in 0 1
do
LANG="input$INPUT_TYPE"
fairseq-preprocess \
  --only-source \
  --trainpref "$TASKFOLDER/train.$LANG" \
  --validpref "${DEVPREF//LANG/$LANG}" \
  --testpref "${TESTPREF//LANG/$LANG}" \
  --destdir "MERGENLI-bin/$LANG" \
  --workers 60 \
  --srcdict dict.txt;
done
fairseq-preprocess \
  --only-source \
  --trainpref "$TASKFOLDER/train.label" \
  --validpref "${DEVPREF//LANG/label}" \
  --destdir "MERGENLI-bin/label" \
  --workers 60;
