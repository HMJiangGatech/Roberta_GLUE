mkdir checkpoints
cd checkpoints
wget https://dl.fbaipublicfiles.com/fairseq/models/roberta.large.tar.gz
wget https://dl.fbaipublicfiles.com/fairseq/models/roberta.large.mnli.tar.gz
wget https://dl.fbaipublicfiles.com/fairseq/models/roberta.base.tar.gz
tar -xzvf roberta.large.tar.gz
tar -xzvf roberta.base.tar.gz
tar -xzvf roberta.large.mnli.tar.gz
