#!/bin/bash
#
#SBATCH --partition=p_nlp
#SBATCH --job-name=train_r1_wmt
#SBATCH --output=%x.%j.log
#SBATCH --error=%x.%j.log
#SBATCH --mem=200G
#SBATCH --cpus-per-task=4
#SBATCH --nodelist=nlpgpu04
#SBATCH --gpus=4
#SBATCH --ntasks-per-node=4

sleep 15m && nvidia-smi &
sleep 120m && nvidia-smi &

OMP_NUM_THREADS=4 python -u -m torch.distributed.launch --nnodes=1  --node_rank=0 --nproc_per_node=4 \
src/zsmt/train_mt.py --pt-dec roberta-large --xlm-name xlm-roberta-large --xlmr-dst-tok \
--redo-output  --freeze-enc --freeze-dec --load-separate-train  --bidi \
--batch 11500 --step 12500000 \
--model bt/large_multi_r1 --train ~/datasets/many-eng/batches/round0_detok.?.batch \
--dev ~/datasets/many-eng/batches/round0_detok.dev.batch --pretrained models/large_40lang_detok_freeze/  --eval-steps 20000
