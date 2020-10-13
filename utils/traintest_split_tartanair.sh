#!/bin/bash

TRAIN=(\
'abandonedfactory' \
'amusement' \
'endofworld' \
#'hospital' \
#'neighborhood' \
'abandonedfactory_night' \
#'carwelding' \
'gascola' \
'japanesealley' \
'ocean' \
)

TEST=(\
#'abandonedfactory' \
#  'amusement' \
# 'endofworld' \
'hospital' \
# 'neighborhood' \
# 'abandonedfactory_night' \
'carwelding' \
# 'gascola' \
# 'japanesealley' \
# 'ocean'
)

VAL=(\
#   'abandonedfactory' \
#   'amusement' \
#   'endofworld' \
#   'hospital' \
'neighborhood' \
#   'abandonedfactory_night' \
#   'carwelding' \
#   'gascola' \
#   'japanesealley' \
#   'ocean'
)

INPUT=$1
OUTPUT=$2

mkdir -p $OUTPUT/train
mkdir -p $OUTPUT/val
mkdir -p $OUTPUT/test

for folder in ${TRAIN[@]}
do
    mv $INPUT/$folder $OUTPUT/train/$folder
done

for folder in ${VAL[@]}
do
    mv $INPUT/$folder $OUTPUT/val/$folder
done

for folder in ${TEST[@]}
do
    mv $INPUT/$folder $OUTPUT/test/$folder
done