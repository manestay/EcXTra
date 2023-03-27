#!/bin/zsh
IFS=$'\n'

URLS=(
    http://web-language-models.s3-website-us-east-1.amazonaws.com/ngrams/kk/deduped/kk.deduped.xz
    http://web-language-models.s3-website-us-east-1.amazonaws.com/ngrams/gu/deduped/gu.deduped.xz
    http://web-language-models.s3-website-us-east-1.amazonaws.com/ngrams/is/deduped/is.deduped.xz
    https://data.statmt.org/cc-100/my.txt.xz
    https://data.statmt.org/cc-100/ps.txt.xz
    https://data.statmt.org/cc-100/lv.txt.xz
    https://data.statmt.org/cc-100/ne.txt.xz
    https://data.statmt.org/cc-100/si.txt.xz
)

cd wmt/monolingual

for url in $URLS; do
    zip_name=${url##*/}
    fname=${zip_name%.*}
    echo $fname
    if [ ! -f ${fname} ]; then
        wget -nc $url
        unxz $zip_name
        echo "orig # lines"
        wc -l $fname
        sed '/^$/d' -i $fname
        echo "after removing empty lines"
        wc -l $fname
    fi
done
