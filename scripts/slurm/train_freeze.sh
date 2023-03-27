#!/bin/bash
#
#SBATCH --partition=p_nlp
#SBATCH --job-name=train_stage1
#SBATCH --output=%x.%j.log
#SBATCH --mem=200G
#SBATCH --cpus-per-task=4
#SBATCH --nodelist=nlpgpu04
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=4

# Run the program
sleep 15m && nvidia-smi &
sleep 120m && nvidia-smi &

DATA_DIR=~/datasets/many-eng

OMP_NUM_THREADS=4 python -u -m torch.distributed.launch --nnodes=1  --node_rank=0 --nproc_per_node=4 \
src/zsmt/train_mt.py --model models/large_40lang_detok_freeze/ --train $DATA_DIR/batches/train.custom.2m.$i.batch --dev $DATA_DIR/batches/dev.custom.batch --capacity 1000 --batch 20000 --step 12500000 --pt-dec roberta-large --xlm-name xlm-roberta-large --freeze-enc --freeze-dec
