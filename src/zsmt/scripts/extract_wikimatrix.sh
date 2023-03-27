#! /bin/bash
dir=${1}

for fname in $dir/*gz; do
    suffixes="${fname#*.}"
    pair="${suffixes%%.*}"
    l1=$(echo $pair | cut -d "-" -f 1)
    l2=$(echo $pair | cut -d "-" -f 2)

    python WikiMatrix/extract.py \
    --tsv $fname \
    --bitext $dir/WikiMatrix.$pair.txt \
    --src-lang $l1 --trg-lang $l2 \
    --threshold 1.04
done
