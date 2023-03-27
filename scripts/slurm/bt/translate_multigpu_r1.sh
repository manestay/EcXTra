#!/bin/bash
#
#SBATCH --partition=p_nlp
#SBATCH --job-name=translate_r1
#SBATCH --output=%x.%j.log
#SBATCH --error=%x.%j.log
#SBATCH --mem-per-cpu=25G
#SBATCH --cpus-per-task=8
#SBATCH --nodelist=nlpgpu01
#SBATCH --gpus=8
#SBATCH --ntasks=1

sleep 30m && nvidia-smi &
DATA_DIR=~/datasets/bt/

srun python -u src/zsmt/translate_multigpu.py  --model models/large_40lang_detok_freeze/ --input wmt/monolingual/kk.deduped --output $DATA_DIR/round0.kk_en.en --xlm-name xlm-roberta-large --pt-dec roberta-large --n_gpus -1 --out_parts bt/parts_r1_0 --batch 1500 --verbose --max-sents 4000000

srun python -u src/zsmt/translate_multigpu.py  --model models/large_40lang_detok_freeze/ --input wmt/monolingual/gu.deduped --output $DATA_DIR/round0.gu_en.en --xlm-name xlm-roberta-large --pt-dec roberta-large --n_gpus -1 --out_parts bt/parts_r1_1 --batch 1500 --verbose --max-sents 4000000

srun python -u src/zsmt/translate_multigpu.py  --model models/large_40lang_detok_freeze/ --input wmt/monolingual/is.deduped --output $DATA_DIR/round0.is_en.en --xlm-name xlm-roberta-large --pt-dec roberta-large --n_gpus -1 --out_parts bt/parts_r1_2 --batch 1500 --verbose --max-sents 4000000

srun python -u src/zsmt/translate_multigpu.py  --model models/large_40lang_detok_freeze/ --input wmt/monolingual/si.txt --output $DATA_DIR/round0.si_en.en --xlm-name xlm-roberta-large --pt-dec roberta-large --n_gpus -1 --out_parts bt/parts_r1_3 --batch 1500 --verbose --max-sents 4000000

srun python -u src/zsmt/translate_multigpu.py  --model models/large_40lang_detok_freeze/ --input wmt/monolingual/ne.txt --output $DATA_DIR/round0.ne_en.en --xlm-name xlm-roberta-large --pt-dec roberta-large --n_gpus -1 --out_parts bt/parts_r1_4 --batch 1500 --verbose --max-sents 4000000

srun python -u src/zsmt/translate_multigpu.py  --model models/large_40lang_detok_freeze/ --input wmt/monolingual/ps.txt --output $DATA_DIR/round0.ps_en.en --xlm-name xlm-roberta-large --pt-dec roberta-large --n_gpus -1 --out_parts bt/parts_r1_5 --batch 1500 --verbose --max-sents 4000000

srun python -u src/zsmt/translate_multigpu.py  --model models/large_40lang_detok_freeze/ --input wmt/monolingual/is.deduped --output $DATA_DIR/round0.is_en.en --xlm-name xlm-roberta-large --pt-dec roberta-large --n_gpus -1 --out_parts bt/parts_r1_6 --batch 1500 --verbose --max-sents 4000000
