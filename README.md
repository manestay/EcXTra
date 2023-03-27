# Multilingual Bidirectional Unsupervised Translation

This repository is associated with the paper ["Multilingual Bidirectional Unsupervised Translation Through Multilingual Finetuning and Back-Translation"](https://arxiv.org/abs/2209.02821), published in LoResMT 2023 @ EACL.

This README is a work-in-progress, and not all commands have been fully tested. Some paths will need to be updated to match your system. Please raise an issue on Github for assistance.

## 0. Setup environment + data

We assume you are running a Linux OS, and are using a SLURM job scheduler.

### 0.1. Setup conda environment

1. Create a virtual environment with Python3 and install PyTorch
```bash
conda create -n zsmt python=3.7
conda activate zsmt
conda install pytorch cudatoolkit=11.1 -c pytorch
```

3. Install package
```bash
pip install -e .
```

### 0.2. Get the train data
First, `wget` all `train.v1*` files from http://rtg.isi.edu/many-eng/data/v1/ to `DATA_DIR=[dir]`. Then, run
```
python scripts/misc/process_m2e.py $DATA_DIR
```
The directory should have 4 files `train.v1.custom.2m*` with 80m lines each.

Next, detokenize both sides of the text, since whitespace matters for SentencePiece tokenization used in the pretrained models we initialize to, and we want to be consistent.
```
sacremoses -l en -j [THREADS] detokenize < $DATA_DIR/train.custom.2m.eng.tok > $DATA_DIR/train.custom.2m.detok.en
sacremoses -l en -j [THREADS] detokenize < $DATA_DIR/train.custom.2m.src.tok > $DATA_DIR/train.custom.2m.detok.src
```

### 0.3. Get the dev and test data
Download and untar dev data from `http://data.statmt.org/wmt19/translation-task/dev.tgz` to `WMT_DIR=[dir]`.
Download and untar test data by running commands in `scripts/misc/get_test_sets.sh`.

Now, format dev and test folders by running
```
python scripts/misc/format_dev.py
python scripts/misc/format_test.py
```
You will likely need to change the paths in


## Stage 1. Multilingual fine-tuning

### 1.1. Create batches for stage 1
The train script takes in data as binary batch files, 1 per GPU. We assume from here on you have a system with 4 GPUs.

First, split the train data files into 4 parts:
```
python src/zsmt/scripts/split_data.py $DATA_DIR/train.custom.2m.detok.src 4 $DATA_DIR/train.custom.2m.detok.src.split
python src/zsmt/scripts/split_data.py $DATA_DIR/train.custom.2m.detok.en 4 $DATA_DIR/train.custom.2m.detok.en.split
```

Then, create a batch file for each train split:
```
for i in {1..4}; do
python src/zsmt/create_mt_batches.py --pt-tok roberta-base  --src $DATA_DIR/train.custom.2m.detok.src.split$i --dst $DATA_DIR/train.custom.2m.detok.en.split$i --output $DATA_DIR/batches/train.custom.2m.$i.batch
done
```

And also 1 batch file for the dev data:
```bash
python src/zsmt/create_mt_batches.py --pt-tok roberta-base  --src $DATA_DIR/dev.src --dst $DATA_DIR/dev.en --output $DATA_DIR/batches/custom.dev.batch
```

### 1.2. Kick off stage 1 training
Start training by running `scripts/slurm/train_freeze.sh`. Make sure you modify the paths and the `SBATCH` directives to fit your system.

This script will run for `12500000` steps, i.e., it will run for a long time. We have been manually applying early stopping by monitoring for "Saving best bleu" in the log file, and stopping it after a few days without improvement (generally under 1m steps total).

TODO: implement automatic early stopping

### 1.3. Run inference for stage 1
To run xx-en inference on the 7 test sets, run `scripts/misc/translate_all.sh`. Make sure to uncomment the `MODEL_DIR` and `python src/zsmt/translate.py` lines that correspond to this stage of training (the first two), and comment out the two for the back-translation stage (the later two).

## Stage 2. Back-Translation Training
Stage 2 consists of two back-translation (BT) training rounds. We refer to stage 1 as round 0, so we have round 1 and round 2 in this 2nd stage.

### 2.1. Download monolingual data

Run `scripts/misc/get_wmt_monolingual_test.sh`. The monolingual corpora will be downloaded to `wmt/monolingual`.

### 2.2. Run back-translations for xx->en
Refer to `scripts/slurm/bt/translate_multigpu_r1.sh`. If all goes well, there should be 7 files matching `~/datasets/bt/round0*`.

### 2.3. Create batch files for round 1 BT training
Refer to the first command of `scripts/slurm/bt/create_bt_batch.sh`. This will take in the xx->en data from the original train data, and the synthetic en->xx data from the back-translations.

### 2.4. Kick off round 1 training
Run `scripts/slurm/bt/train_round1.sh`. Again, you need to manually stop the training. Convergence should take ~500k steps.

### 2.5. Run inference for round 1
To run xx-en inference on the 7 test sets, run `scripts/misc/translate_all.sh`. Make sure to uncomment the `MODEL_DIR` and `python src/zsmt/translate.py` lines that correspond to this stage of training (the later two), and comment out the two for the zero-shot stage (the first two).

To run en-xx inference on the 7 test sets, run `scripts/misc/translate_from_en.sh`. Make sure to specify the appropriate `MODEL_DIR`.

### 2.6. Run back-translations for en->xx
Refer to `scripts/slurm/bt/translate_multigpu_r2.sh`. If all goes well, there should be 7 files matching ` bt/large_multi_r1/round1*`.

### 2.7. Create batch files for round 2 BT training
Refer to the second command of `scripts/slurm/bt/create_bt_batch.sh`. This will take in  synthetic xx->en data from the back-translations, and the synthetic en->xx data from the previous round.

### 2.8. Kick off round 2 training
Run `scripts/slurm/bt/train_round2.sh`. Again, you need to manually stop the training. Convergence should take ~500k steps.

### 2.9. Run inference for round 1
To run xx-en inference on the 7 test sets, run `scripts/misc/translate_all.sh`. Make sure to uncomment the `MODEL_DIR` and `python src/zsmt/translate.py` lines that correspond to this stage of training (the later two), and comment out the two for the zero-shot stage (the first two).

To run en-xx inference on the 7 test sets, run `scripts/misc/translate_from_en.sh`. Make sure to specify the appropriate `MODEL_DIR`.


## Citation
```
@article{li2022multilingual,
  title={Multilingual Bidirectional Unsupervised Translation Through Multilingual Finetuning and Back-Translation},
  author={Li, Bryan and Rasooli, Mohammad Sadegh and Patel, Ajay and Callison-Burch, Chris},
  journal={The Sixth Workshop on Technologies for Machine Translation of Low-Resource Languages (LoResMT 2023) at EACL},
  year={2023}
}
```
