#!/bin/zsh
IFS=$'\n'

model_dir=$1
# bt/large_multi_om_nb    bt/large_multi_om_nb_round2

TEST_SETS=(
    wmt2/test/wmt19.gu-en.gu
    wmt2/test/wikipedia.test.si-en.si
    wmt2/test/wikipedia.test.ne-en.ne
)

for test_set in $TEST_SETS; do
    BASENAME=$(basename "${test_set%.*}").rev
    out_name=$model_dir/$BASENAME.pred2.txt
    python scripts/misc/tokenize_indic.py $out_name $test_set

    echo $test_set.tok  ${out_name}.tok
    sacrebleu --force --tokenize none $test_set.tok < ${out_name}.tok
    # python src/zsmt/scripts/eval_sacre_bleu.py -o ${out_name}.tok -g $test_set.tok -n
done

sacrebleu --force --tokenize spm wmt2/test/test.alt.my < $model_dir/test.alt.rev.pred2.txt
