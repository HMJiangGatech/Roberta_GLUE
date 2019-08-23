#!/bin/bash

if [[ $# -ne 1 ]]; then
  DATA_FOLDER=./superglue_data
else
  DATA_FOLDER=$1
fi

mkdir -p $DATA_FOLDER
wget -O $DATA_FOLDER/combined.zip https://dl.fbaipublicfiles.com/glue/superglue/data/v2/combined.zip
unzip $DATA_FOLDER/combined.zip
rm $DATA_FOLDER/combined.zip

# we also need to copy the RoBERTa dictionary into the same directory
cp dict.txt $DATA_FOLDER/WSC/dict.txt
