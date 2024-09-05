export EAGLE_HOME=~/octo-EAGLE
export TRAIN_DATA_PATH=$EAGLE_HOME/train-data-llama-3-1-8b
export BASE_MODEL_PATH=/opt/models/Meta-Llama-3.1-8B-Instruct/
export DRAFT_MODEL_PATH=$EAGLE_HOME/models/EAGLE-LLaMA3.1-Instruct-8B/
export CONFIG_PATH=$EAGLE_HOME/eagle/train/llama-3-1-8b-instruct_config.json

cd $EAGLE_HOME/eagle/ge_data
#rm ShareGPT_V4.3_unfiltered_cleaned_split.json
#wget https://huggingface.co/datasets/Aeala/ShareGPT_Vicuna_unfiltered/blob/main/ShareGPT_V4.3_unfiltered_cleaned_split.json
/opt/bin/cuda-reserve.py --num-gpus 4 \
  python allocation.py \
    --base-model-path $BASE_MODEL_PATH \
    --outdir $TRAIN_DATA_PATH

cd $EAGLE_HOME
/opt/bin/cuda-reserve.py --num-gpus 2 \
  accelerate launch -m --num_processes 2 --mixed_precision=bf16 eagle.train.main \
    --tmpdir $TRAIN_DATA_PATH \
    --basepath $BASE_MODEL_PATH \
    --cpdir $DRAFT_MODEL_PATH \
    --configpath $CONFIG_PATH
