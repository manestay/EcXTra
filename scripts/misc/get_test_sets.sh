#/bin/sh
DATA_DIR=wmt/test
mkdir -p $DATA_DIR

sacrebleu -t wmt19 -l kk-en --echo src > ${DATA_DIR}/wmt19.kk-en.kk
sacrebleu -t wmt19 -l kk-en --echo ref > ${DATA_DIR}/wmt19.kk-en.en

sacrebleu -t wmt19 -l gu-en --echo src > ${DATA_DIR}/wmt19.gu-en.gu
sacrebleu -t wmt19 -l gu-en --echo ref > ${DATA_DIR}/wmt19.gu-en.en

wget https://github.com/facebookresearch/flores/blob/main/previous_releases/floresv1/data/wikipedia_en_ne_si_test_sets.tgz?raw=true -O wmt/flores.tgz
tar xvf wmt/flores.tgz -C wmt
cp wmt/wikipedia_en_ne_si_test_sets/wikipedia.test.si-en.* $DATA_DIR
cp wmt/wikipedia_en_ne_si_test_sets/wikipedia.test.ne-en.* $DATA_DIR

sacrebleu -t wmt20 -l ps-en --echo src > ${DATA_DIR}/wmt20.ps-en.ps
sacrebleu -t wmt20 -l ps-en --echo ref > ${DATA_DIR}/wmt20.ps-en.en

sacrebleu -t wmt21 -l is-en --echo src > ${DATA_DIR}wmt21.is-en.is
sacrebleu -t wmt21 -l is-en --echo ref > ${DATA_DIR}wmt21.is-en.en

wget http://lotus.kuee.kyoto-u.ac.jp/WAT/my-en-data/wat2020.my-en.zip -O wmt/wat2020.zip
unzip wmt/wat2020.zip -d wmt
cp wmt/wat2020.my-en/alt/test.alt.* $DATA_DIR
