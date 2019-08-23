wget https://dl.fbaipublicfiles.com/glue/superglue/data/v2/WSC.zip
unzip WSC.zip
rm WSC.zip

# we also need to copy the RoBERTa dictionary into the same directory
wget -O WSC/dict.txt https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt

# For Glue Order
cp wsc_test_glue.jsonl WSC/test_glue.jsonl
