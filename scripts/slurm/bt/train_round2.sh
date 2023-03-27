#!/bin/bash
#
#SBATCH --partition=p_nlp
#SBATCH --job-name=train_r2_wmt
#SBATCH --output=%x.%j.log
#SBATCH --error=%x.%j.log
#SBATCH --mem=200G
#SBATCH --cpus-per-task=4
#SBATCH --nodelist=nlpgpu04
#SBATCH --gpus=4
#SBATCH --ntasks-per-node=4

sleep 30m && nvidia-smi &
sleep 120m && nvidia-smi &

OMP_NUM_THREADS=4 python -u -m torch.distributed.launch --nnodes=1  --node_rank=0 --nproc_per_node=4 \
src/zsmt/train_mt.py --pt-dec roberta-large --xlm-name xlm-roberta-large --xlmr-dst-tok \
--freeze-enc --freeze-dec --load-separate-train  --bidi \
--batch 11500 --step 12500000 \
--model bt/large_multi_r2 --train ~/datasets/bt/batches/round1.?.batch \
--dev ~/datasets/bt/batches/round1.dev.batch --pretrained bt/large_multi_r1  --eval-steps 20000 -ftd  ~/datasets/many-eng/batches/round0_detok.foreign_toks.json
