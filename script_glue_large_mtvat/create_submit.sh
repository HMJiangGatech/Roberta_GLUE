#!/bin/bash

if [[ $# -ne 1 ]]; then
  GPUID=0
else
  GPUID=$1
fi

echo "Run on GPU $GPUID"

CUDA_VISIBLE_DEVICES=$GPUID python eval_glue.py --task QNLI  --ckp checkpoints/QNLI/10_1e-05_MTVAT_Base_0_staged/checkpoint_best.pt
CUDA_VISIBLE_DEVICES=$GPUID python eval_glue.py --task STS-B --ckp checkpoints/STS-B/10_2e-05_MTVAT_Base_2019_staged/checkpoint_best.pt
CUDA_VISIBLE_DEVICES=$GPUID python eval_glue.py --task RTE   --ckp checkpoints/RTE/10_2e-05_MTVAT_Base_0_staged/checkpoint_best.pt
CUDA_VISIBLE_DEVICES=$GPUID python eval_glue.py --task SST-2 --ckp checkpoints/SST-2/10_1e-05_MTVAT_Base_0_staged/checkpoint_best.pt
CUDA_VISIBLE_DEVICES=$GPUID python eval_glue.py --task QQP   --ckp checkpoints/QQP/10_1e-05_MTVAT_Base_0_staged/checkpoint_best.pt
CUDA_VISIBLE_DEVICES=$GPUID python eval_glue.py --task MRPC  --ckp checkpoints/MRPC/10_1e-05_MTVAT_Base_0_staged/checkpoint_best.pt
CUDA_VISIBLE_DEVICES=$GPUID python eval_glue.py --task CoLA  --ckp checkpoints/CoLA/10_1e-05_MTVAT_Base_0_staged/checkpoint_best.pt
CUDA_VISIBLE_DEVICES=$GPUID python eval_glue.py --task WSC   --ckp checkpoints/WSC/58_2e-05_Baseline_Base_0_staged/checkpoint_best.pt

CUDA_VISIBLE_DEVICES=$GPUID python eval_glue.py --task MNLI  --ckp checkpoints/MNLI/10_1e-05_MTVAT_Base_0_staged/checkpoint3.pt
CUDA_VISIBLE_DEVICES=$GPUID python eval_glue.py --task AX    --ckp checkpoints/MNLI/10_1e-05_MTVAT_Base_0_staged/checkpoint3.pt
