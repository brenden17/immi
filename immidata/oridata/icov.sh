#!/bin/bash
for f in ./04chosun_/*.txt
do
    iconv -f euc-kr -t utf-8 $f > 04chosun/$f
done
