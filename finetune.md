fine tuning readme

## 8B

### Generate data

```
export EAGLEHOME=<your home directory>
cd $EAGLEHOME/eagle/ge_data
wget https://huggingface.co/datasets/Aeala/ShareGPT_Vicuna_unfiltered/blob/main/ShareGPT_V4.3_unfiltered_cleaned_split.json
/opt/bin/cuda-reserve.py --num-gpus 2 python allocation.py --outdir ../../train-data-llama-3-1-8b
```

### Fine tuning

```
cd $EAGLEHOME
export BASEMODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"
export OUTPUTDIR="models/eagle-llama-3.1-instruct-8b"
/opt/bin/cuda-reserve.py --num-gpus 2 \
accelerate launch -m --num_processes 2 --mixed_precision=bf16 eagle.train.main \
--tmpdir train-data-llama-3-1-8b \
--basepath $BASEMODEL \
--cpdir $OUTPUTDIR \
--configpath eagle/train/llama-3-1-8b-instruct_config.json
cp eagle/train/llama-3-1-8b-instruct_config.json $OUTPUTDIR
cp eagle/train/llama-3-1-8b-instruct_config.json $OUTPUTDIR/config.json
```

### Evaluate

```
cd $EAGLEHOME
source run-llama-3-1-8b.sh
```

## 70B

Hyunsung to fill in the deatils.