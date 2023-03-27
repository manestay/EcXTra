#!/bin/zsh

IFS=$'\n'

TEST_SETS=(
    wmt2/test/wmt19.kk-en.en
    wmt2/test/wmt19.gu-en.en
    wmt2/test/wikipedia.test.si-en.en
    wmt2/test/wikipedia.test.ne-en.en
    wmt2/test/wmt20.ps-en.en
    wmt2/test/newstest2021.is-en.en
    wmt2/test/test.alt.en #my
)

# from bt/batches/round0.foreign_toks.json
FOREIGN_TOKS=(
    '\u2581cuestiones'
    '\u2581\u122a\u1356\u122d\u1275'
    '\u2581powinni\u015bmy'
    '\u2581istaknuo'
    '\u2581pamahalaan'
    '\u2581\ubaa8\ub2c8\ud130'
    '\u2581imk\u00e2n'
)

LANG_CODES=(
    kk
    gu
    si
    ne
    ps
    is
    my
    # lv
)

{
for (( i = 1; i <= $#TEST_SETS; i++ )); do
    SET=$TEST_SETS[i]
    FOREIGN_TOK=$FOREIGN_TOKS[i]
    echo $FOREIGN_TOK
    LANG_CODE=$LANG_CODES[i]

    BASENAME=$(basename "${SET%.*}").rev

    SET_REF="${SET%.*}.${LANG_CODE}"

    # MODEL_DIR=bt/large_multi_om_nb
    MODEL_DIR=bt/large_multi_om_nb_r2
    python src/zsmt/translate.py --model $MODEL_DIR --input $SET --output $MODEL_DIR/$BASENAME.pred2.txt --pt-dec roberta-large --xlm-name xlm-roberta-large --xlmr-dst-tok --batch 5000 --foreign-tok $FOREIGN_TOK

    echo "on $SET_REF"
    python src/zsmt/scripts/eval_sacre_bleu.py -o $MODEL_DIR/$BASENAME.pred2.txt -g $SET_REF
    echo
done
exit
}
