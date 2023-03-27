#!/bin/zsh

# run this script after having trained the back-translation models

# source ~/anaconda3/etc/profile.d/conda.sh
# conda activate zsmt

LANG=${1:-lt}
overwrite=false

# python src/zsmt/translate.py --tok models/tok_en_is --model bt_is_bidi/models/round3.en_is/ --input wmt/zsmt_format/newstest2021.is-en.en --output bt_is_bidi/models/round3.en_is/newstest2021.en-is.pred.out2

# python src/zsmt/scripts/eval_sacre_bleu.py -o back_trans_tok_news/models/round4.kk_en/wmt19.kk-en.pred.out -g wmt/zsmt_format/wmt19.kk-en.en -v

# bt/kk_*/models/*/*en-kk*pred*

for dir in uni bidi; do
    MODEL_DIR=bt/${LANG}_${dir}

    if [ $dir = 'uni' ]; then
        ROUND_2EN=${MODEL_DIR}/models/round2.${LANG}_en
        ROUND_2SRC=${MODEL_DIR}/models/round1.en_${LANG}
    else
        # ROUND_2EN=${MODEL_DIR}/models/round2.${LANG}_en
        ROUND_2EN=${MODEL_DIR}/models/round1.en_${LANG}
        ROUND_2SRC=$ROUND_2EN
    fi

    GOLD_SRC=(wmt/zsmt_format/*.${LANG}-en.${LANG})
    STEM=$(basename ${GOLD_SRC%%.*})
    GOLD_EN=(wmt/zsmt_format/${STEM}.${LANG}-en.en)

    PRED_SRC=$ROUND_2SRC/${STEM}.en-${LANG}.pred.txt
    PRED_EN=$ROUND_2EN/${STEM}.${LANG}-en.pred.txt

    if [[ overwrite = true || ! -f "$PRED_EN" ]]; then
        echo "translating to en..."
        python src/zsmt/translate.py --tok models/tok_en_${LANG} --model $ROUND_2EN \
            --input $GOLD_SRC --output $PRED_EN
    fi

    if [[ overwrite = true || ! -f "$PRED_SRC" ]]; then
        echo "translating to $LANG..."
        python src/zsmt/translate.py --tok models/tok_en_${LANG} --model $ROUND_2SRC \
        --input $GOLD_EN --output $PRED_SRC --add-foreign-tok
    fi

    echo "  ** to_en **"
    echo $ROUND_2EN
    python src/zsmt/scripts/eval_sacre_bleu.py -o $PRED_EN -g $GOLD_EN -v

    echo "  ** to_src **"
    echo $ROUND_2SRC
    python src/zsmt/scripts/eval_sacre_bleu.py -o $PRED_SRC -g $GOLD_SRC -v

done
