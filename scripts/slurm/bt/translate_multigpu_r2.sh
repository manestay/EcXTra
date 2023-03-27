#!/bin/bash
#
#SBATCH --partition=p_nlp
#SBATCH --job-name=translate_r2
#SBATCH --output=%x.%j.log
#SBATCH --error=%x.%j.log
#SBATCH --mem-per-cpu=25G
#SBATCH --cpus-per-task=8
#SBATCH --nodelist=nlpgpu02
#SBATCH --gpus=8
#SBATCH --ntasks=1

sleep 30m && nvidia-smi &

srun python -u src/zsmt/translate_multigpu.py  --model bt/large_multi_r1 --input wmt/monolingual/en.news2018.deduped --output bt/large_multi_r1/round1.en_kk.kk --xlm-name xlm-roberta-large --pt-dec roberta-large --xlmr-dst-tok --n_gpus -1 --out_parts bt/parts_r2_0 --batch 1500 --verbose --max-sents 4000000 --foreign-tok '\u2581cuestiones'

srun python -u src/zsmt/translate_multigpu.py  --model bt/large_multi_r1 --input wmt/monolingual/en.news2018.deduped --output bt/large_multi_r1/round1.en_gu.gu --xlm-name xlm-roberta-large --pt-dec roberta-large --xlmr-dst-tok --n_gpus -1 --out_parts bt/parts_r2_1 --batch 1500 --verbose --max-sents 4000000 --foreign-tok '\u2581\u122a\u1356\u122d\u1275'

srun python -u src/zsmt/translate_multigpu.py  --model bt/large_multi_r1 --input wmt/monolingual/en.news2018.deduped --output bt/large_multi_r1/round1.en_si.si --xlm-name xlm-roberta-large --pt-dec roberta-large --xlmr-dst-tok --n_gpus -1 --out_parts bt/parts_r2_2 --batch 1500 --verbose --max-sents 4000000 --foreign-tok '\u2581powinni\u015bmy'

srun python -u src/zsmt/translate_multigpu.py  --model bt/large_multi_r1 --input wmt/monolingual/en.news2018.deduped --output bt/large_multi_r1/round1.en_ne.ne --xlm-name xlm-roberta-large --pt-dec roberta-large --xlmr-dst-tok --n_gpus -1 --out_parts bt/parts_r2_3 --batch 1500 --verbose --max-sents 4000000 --foreign-tok '\u2581istaknuo'

srun python -u src/zsmt/translate_multigpu.py  --model bt/large_multi_r1 --input wmt/monolingual/en.news2018.deduped --output bt/large_multi_r1/round1.en_is.is --xlm-name xlm-roberta-large --pt-dec roberta-large --xlmr-dst-tok --n_gpus -1 --out_parts bt/parts_r2_4 --batch 1500 --verbose --max-sents 4000000 --foreign-tok '\u2581\ubaa8\ub2c8\ud130'

srun python -u src/zsmt/translate_multigpu.py  --model bt/large_multi_r1 --input wmt/monolingual/en.news2018.deduped --output bt/large_multi_r1/round1.en_ps.ps --xlm-name xlm-roberta-large --pt-dec roberta-large --xlmr-dst-tok --n_gpus -1 --out_parts bt/parts_r2_5 --batch 1500 --verbose --max-sents 4000000 --foreign-tok '\u2581pamahalaan'

srun python -u src/zsmt/translate_multigpu.py  --model bt/large_multi_r1 --input wmt/monolingual/en.news2018.deduped --output bt/large_multi_r1/round1.en_my.my --xlm-name xlm-roberta-large --pt-dec roberta-large --xlmr-dst-tok --n_gpus -1 --out_parts bt/parts_r2_6 --batch 1500 --verbose --max-sents 4000000 --foreign-tok '\u2581imk\u00e2n'
