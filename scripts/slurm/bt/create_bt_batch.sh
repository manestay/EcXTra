#!/bin/bash
#
#SBATCH --partition=p_nlp
#SBATCH --job-name=create_bt_batch
#SBATCH --output=%x.%j.log
#SBATCH --error=%x.%j.log
#SBATCH --mem-per-cpu=100G
#SBATCH --nodelist=nlpgpu05
#SBATCH --gpus=0

# use round 0 to BT for round 1
DATA_DIR=~/datasets/bt/
python -u scripts/misc/bt/create_batch_from_bt.py -bf ${DATA_DIR}/round0.??_en.en  -o ~/datasets/many-eng/batches/round0_detok -ds 500 -alc --shuffle -nb 4 -ef ~/datasets/many-eng/train.custom.2m.detok.en -nwl 25000000

# use round 1 to BT for round 2
ROUND1_DIR=bt/large_multi_om_nb/
python -u scripts/misc/bt/create_batch_from_bt.py -bf ${ROUND1_DIR}/round1.en_??.?? ${DATA_DIR}/round0.??_en.en -o ${DATA_DIR}/batches/round1_new -ds 500 -alc --shuffle -nb 4 -ftd ~/datasets/many-eng/batches/round0_detok.foreign_toks.json
