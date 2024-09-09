# Training data generation takes about 20 hours, and model finetuning takes ~60 hours to finish.
# Don't recommend run those scripts directly!


export EAGLE_HOME=~/octo-EAGLE
export TRAIN_DATA_PATH=$EAGLE_HOME/train-data-llama-3-1-70b
export BASE_MODEL_PATH=/opt/models/meta-llama-3.1-70b-instruct/
export DRAFT_MODEL_PATH=$EAGLE_HOME/models/EAGLE-LLaMA3.1-Instruct-70B/
export CONFIG_PATH=$EAGLE_HOME/eagle/train/llama-3.1-70B-instruct_config.json


cd $EAGLE_HOME/eagle/ge_data

# Download trainin data (if not exist)
#rm ShareGPT_V4.3_unfiltered_cleaned_split.json
#wget https://huggingface.co/datasets/Aeala/ShareGPT_Vicuna_unfiltered/blob/main/ShareGPT_V4.3_unfiltered_cleaned_split.json

## Training Data Generation.
/opt/bin/cuda-reserve.py --num-gpus 6 \
  accelerate launch --config_file=multi_gpu_acc.yaml ge_data_all_llama3-1-70Bchat.py \
    --start=0 \
    --end=68000 \
    --index=0 \
    --gpu_index 0 \  # Actually not used currently
    --outdir $TRAIN_DATA_PATH/sharegpt_0_67999_mufp16 \
    --base-model-path $BASE_MODEL_PATH

## Model Training
cd $EAGLE_HOME
/opt/bin/cuda-reserve.py --num-gpus 1 --timeout 360000000 \
  python -m eagle.train.main \
    --tmpdir $TRAIN_DATA_PATH \
    --basepath $BASE_MODEL_PATH \
    --cpdir $DRAFT_MODEL_PATH \
    --bs 2 \  # batch size, set to be 2 since CUDA OOM.
              #  (@Hyunsung) I didn't observe harmful effects on result performance.
    --configpath $CONFIG_PATH
