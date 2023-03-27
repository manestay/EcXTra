#!/bin/zsh
IFS=$'\n'

TEST_SETS=(
    wmt2/dev_fmt/newsdev2017-enzh.zh
    wmt2/dev_fmt/newsdev2014.hi
    wmt2/dev_fmt/newsdev2016-entr.tr
    wmt2/dev_fmt/newstest2012.ru
)

{
for (( i = 1; i <= $#TEST_SETS; i++ )); do
    SET=$TEST_SETS[i]

    BASENAME=$(basename "${SET%.*}")

    SET_REF="${SET%.*}".en

    # MODEL_DIR=models/large_40lang_detok_freeze
    # python src/zsmt/translate.py --pt-dec roberta-large --xlm-name xlm-roberta-large --model $MODEL_DIR --input $SET --output $MODEL_DIR/$BASENAME.pred2.txt --batch 1500

    ####
    MODEL_DIR=bt/large_multi_om_nb_r2
    python src/zsmt/translate.py --xlm-name xlm-roberta-large --pt-dec roberta-large --model $MODEL_DIR --input $SET --output $MODEL_DIR/$BASENAME.pred2.txt --xlmr-dst-tok --batch 1500
    ####

    SET_PRED=$MODEL_DIR/$BASENAME.pred2.txt
    echo "BLEU between $SET_PRED and $SET_REF"
    python src/zsmt/scripts/eval_sacre_bleu.py -o $SET_PRED -g $SET_REF -d
    echo
done
exit
}
