#/bin/bash


###############
# GENERAL
###############

DATAPATH="/media/slamcore/frederik/TartanAirIndoorEurocFormatReduced"
DATANAME="TARTANAIR"
JSONPATH="../data_json/tartanair_reduced.json"

DEPSRC="slam"
MAXDEPTH=15.0
NUM_SAMPLE=0

LEFTCROP=0
TOPCROP=0
PATCH_HEIGHT=240
PATHC_WIDTH=320

EPOCHS=60
BATCH_SIZE=8

###############
# NLSPN
###############

LOSS="1.0*L1+1.0*L2"
SAVE="NLSPN_TartanairReduced"
MODELNAME='NLSPN'
INPUTCONF='input'
DEVICE="1"
PORT=20000

python main.py --dir_data $DATAPATH \
                --data_name $DATANAME \
                --split_json $JSONPATH \
                --patch_height $PATCH_HEIGHT \
                --patch_width $PATHC_WIDTH \
                --gpus $DEVICE \
                --loss $LOSS \
                --epochs $EPOCHS \
                --batch_size $BATCH_SIZE \
                --max_depth $MAXDEPTH \
                --num_sample $NUM_SAMPLE \
                --top_crop $TOPCROP \
                --left_crop $LEFTCROP \
                --test_crop \
                --save $SAVE \
                --dep_src $DEPSRC \
                --augment True \
                --model_name $MODELNAME \
                --port $PORT \
                --input_conf $INPUTCONF

###############
# NCONV ENCDEC
###############

LOSS="1.0*L2"
SAVE="NCONV_ENCDEC_TartanairReduced"
MODELNAME='NCONV_ENCDEC'
INPUTCONF='input'
DEVICE="1"
PORT=20001

python main.py --dir_data $DATAPATH \
                --data_name $DATANAME \
                --split_json $JSONPATH \
                --patch_height $PATCH_HEIGHT \
                --patch_width $PATHC_WIDTH \
                --gpus $DEVICE \
                --loss $LOSS \
                --epochs $EPOCHS \
                --batch_size $BATCH_SIZE \
                --max_depth $MAXDEPTH \
                --num_sample $NUM_SAMPLE \
                --top_crop $TOPCROP \
                --left_crop $LEFTCROP \
                --test_crop \
                --save $SAVE \
                --dep_src $DEPSRC \
                --augment True \
                --model_name $MODELNAME \
                --port $PORT \
                --input_conf $INPUTCONF

###############
# NCONV STREAM
###############

LOSS="1.0*L2"
SAVE="NCONV_STREAM_TartanairReduced"
MODELNAME='NCONV_STREAM'
INPUTCONF='input'
DEVICE="1"
PORT=20002

python main.py --dir_data $DATAPATH \
                --data_name $DATANAME \
                --split_json $JSONPATH \
                --patch_height $PATCH_HEIGHT \
                --patch_width $PATHC_WIDTH \
                --gpus $DEVICE \
                --loss $LOSS \
                --epochs $EPOCHS \
                --batch_size $BATCH_SIZE \
                --max_depth $MAXDEPTH \
                --num_sample $NUM_SAMPLE \
                --top_crop $TOPCROP \
                --left_crop $LEFTCROP \
                --test_crop \
                --save $SAVE \
                --dep_src $DEPSRC \
                --augment True \
                --model_name $MODELNAME \
                --port $PORT \
                --input_conf $INPUTCONF

###############
# NCONV UNGUIDED
###############

LOSS="1.0*L2"
SAVE="NCONV_UNGUIDED_TartanairReduced"
MODELNAME='NCONV_UNGUIDED'
INPUTCONF='input'
DEVICE="1"
PORT=20003

python main.py --dir_data $DATAPATH \
                --data_name $DATANAME \
                --split_json $JSONPATH \
                --patch_height $PATCH_HEIGHT \
                --patch_width $PATHC_WIDTH \
                --gpus $DEVICE \
                --loss $LOSS \
                --epochs $EPOCHS \
                --batch_size $BATCH_SIZE \
                --max_depth $MAXDEPTH \
                --num_sample $NUM_SAMPLE \
                --top_crop $TOPCROP \
                --left_crop $LEFTCROP \
                --test_crop \
                --save $SAVE \
                --dep_src $DEPSRC \
                --augment True \
                --model_name $MODELNAME \
                --port $PORT \
                --input_conf $INPUTCONF


###############
# PNCONV UNGUIDED
###############

LOSS="1.0*MaskedProbExp"
SAVE="PNCONV_UNGUIDED_TartanairReduced"
MODELNAME='PNCONV_UNGUIDED'
INPUTCONF='learned'
DEVICE="1"
PORT=20004

python main.py --dir_data $DATAPATH \
                --data_name $DATANAME \
                --split_json $JSONPATH \
                --patch_height $PATCH_HEIGHT \
                --patch_width $PATHC_WIDTH \
                --gpus $DEVICE \
                --loss $LOSS \
                --epochs $EPOCHS \
                --batch_size $BATCH_SIZE \
                --max_depth $MAXDEPTH \
                --num_sample $NUM_SAMPLE \
                --top_crop $TOPCROP \
                --left_crop $LEFTCROP \
                --test_crop \
                --save $SAVE \
                --dep_src $DEPSRC \
                --augment True \
                --model_name $MODELNAME \
                --port $PORT \
                --input_conf $INPUTCONF
