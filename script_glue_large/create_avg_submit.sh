#!/bin/bash

if [[ $# -ne 1 ]]; then
  GPUID=0
else
  GPUID=$1
fi

echo "Run on GPU $GPUID"

SAVEDIR=checkpoints/QNLI/10_1e-05_Baseline_Large_0_staged
python fairseq/scripts/average_checkpoints.py --inputs $SAVEDIR --num-epoch-checkpoints 7 --output $SAVEDIR/avgmodel.pt
CUDA_VISIBLE_DEVICES=$GPUID python eval_glue.py --task QNLI  --ckp $SAVEDIR/avgmodel.pt

SAVEDIR=checkpoints/STS-B/10_2e-05_Baseline_Large_0_staged
python fairseq/scripts/average_checkpoints.py --inputs $SAVEDIR --num-epoch-checkpoints 7 --output $SAVEDIR/avgmodel.pt
CUDA_VISIBLE_DEVICES=$GPUID python eval_glue.py --task STS-B --ckp $SAVEDIR/avgmodel.pt

SAVEDIR=checkpoints/RTE/10_2e-05_Baseline_Large_0_staged
python fairseq/scripts/average_checkpoints.py --inputs $SAVEDIR --num-epoch-checkpoints 7 --output $SAVEDIR/avgmodel.pt
CUDA_VISIBLE_DEVICES=$GPUID python eval_glue.py --task RTE   --ckp $SAVEDIR/avgmodel.pt

SAVEDIR=checkpoints/SST-2/10_1e-05_Baseline_Large_0_staged
python fairseq/scripts/average_checkpoints.py --inputs $SAVEDIR --num-epoch-checkpoints 7 --output $SAVEDIR/avgmodel.pt
CUDA_VISIBLE_DEVICES=$GPUID python eval_glue.py --task SST-2 --ckp $SAVEDIR/avgmodel.pt

SAVEDIR=checkpoints/QQP/10_1e-05_Baseline_Large_0_staged
python fairseq/scripts/average_checkpoints.py --inputs $SAVEDIR --num-epoch-checkpoints 7 --output $SAVEDIR/avgmodel.pt
CUDA_VISIBLE_DEVICES=$GPUID python eval_glue.py --task QQP   --ckp $SAVEDIR/avgmodel.pt

SAVEDIR=checkpoints/MRPC/10_1e-05_Baseline_Large_0_staged
python fairseq/scripts/average_checkpoints.py --inputs $SAVEDIR --num-epoch-checkpoints 7 --output $SAVEDIR/avgmodel.pt
CUDA_VISIBLE_DEVICES=$GPUID python eval_glue.py --task MRPC  --ckp $SAVEDIR/avgmodel.pt

SAVEDIR=checkpoints/CoLA/10_1e-05_Baseline_Large_0_staged
python fairseq/scripts/average_checkpoints.py --inputs $SAVEDIR --num-epoch-checkpoints 7 --output $SAVEDIR/avgmodel.pt
CUDA_VISIBLE_DEVICES=$GPUID python eval_glue.py --task CoLA  --ckp $SAVEDIR/avgmodel.pt
SAVEDIR=checkpoints/WSC/58_2e-05_Baseline_Large_0_staged
python fairseq/scripts/average_checkpoints.py --inputs $SAVEDIR --num-epoch-checkpoints 7 --output $SAVEDIR/avgmodel.pt
CUDA_VISIBLE_DEVICES=$GPUID python eval_glue.py --task WSC   --ckp $SAVEDIR/avgmodel.pt


SAVEDIR=checkpoints/MNLI/10_1e-05_Baseline_Large_0_staged
python fairseq/scripts/average_checkpoints.py --inputs $SAVEDIR --num-epoch-checkpoints 9 --output $SAVEDIR/avgmodel.pt
CUDA_VISIBLE_DEVICES=$GPUID python eval_glue.py --task MNLI  --ckp $SAVEDIR/avgmodel.pt
CUDA_VISIBLE_DEVICES=$GPUID python eval_glue.py --task AX    --ckp $SAVEDIR/avgmodel.pt
