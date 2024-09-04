Refer original [repo document](https://github.com/SafeAILab/EAGLE).

Note: 
@Hyunsung: Highly recommend to run those scripts on blacktip machine. some model paths or constants are bind to this machine's path and files.
## Contents
- [Setup & Installation](#setup--installation)
- [Average Acceptance Length](#average-acceptance-length) 
- [Train](#train)
- [Inference](#inference)
- [Evaluation](#evaluation)
- [With gpt-fast](#with-gpt-fast)


## Setup & Installation

only from the source available, this codebase has been modified. 

```bash
git clone https://github.com/octoml/EAGLE.git
cd EAGLE
git checkout EAGLE1
pip install -e .
```

## Average Acceptance Length

### Llama-2-7B

Run script `source run-llama-2.sh`

```
BASELINE Llama-2
total time in seconds: 1473.9602913856506

Base: meta-llama/Llama-2-7b-chat-hf
Drft: yuhuili/EAGLE-llama2-chat-7B
Mode: chain - length 5
total time in seconds: 877.8712062835693
average accept length: 2.3283681869506836

Base: meta-llama/Llama-2-7b-chat-hf
Drft: yuhuili/EAGLE-llama2-chat-7B
Mode: tree - EAGLE-1
total time in seconds: 524.0542569160461
average accept length: 3.676884889602661
```

### Llama-3-8B

Move fine tune to `models/llama-3-chat/state_20`
And, run script `source run-llama-3.sh`

```
BASELINE Llama-3
total time in seconds: 980.8523852825165

Base: meta-llama/Meta-Llama-3-8B-Instruct
Drft: yuhuili/EAGLE-LLaMA3-Instruct-8B
Mode: chain - length 5
total time in seconds: 653.6523582935333
average accept length: 1.9890466928482056

Base: meta-llama/Meta-Llama-3-8B-Instruct
Drft: Hyunsung llama-3-chat fine tune
Mode: chain - length 5
total time in seconds: 621.6361937522888
average accept length: 2.1005005836486816

Base: meta-llama/Meta-Llama-3-8B-Instruct
Drft: yuhuili/EAGLE-LLaMA3-Instruct-8B
Mode: tree - EAGLE-1
total time in seconds: 453.83766627311707
average accept length: 2.7998197078704834

Base: meta-llama/Meta-Llama-3-8B-Instruct
Drft: Hyunsung llama-3-chat fine tune
Mode: tree - EAGLE-1
total time in seconds: 422.923953294754
average accept length: 3.082038164138794
```

### Llama-3.1-8B

Move fine tune to `models/llama-3-1-chat-with-3-1-data/state_17`
And, run script `source run-llama-3-1.sh`

```
Base: meta-llama/Meta-Llama-3-8B-Instruct
Drft: /home/hlee/scratch/eagle/out/llama-3-1-chat-with-3-1-data/state_17/
Mode: chain - length 5
total time in seconds: 1420.5183284282684
average accept length: 1.988234519958496

Base: meta-llama/Meta-Llama-3-8B-Instruct
Drft: /home/hlee/scratch/eagle/out/llama-3-1-chat-with-3-1-data/state_17/
Mode: tree - EAGLE-1
total time in seconds: 1049.5087773799896
average accept length: 2.835796356201172
```

### Llama-3.1-70B
Eval script WIP

## Train

### Generate Train Data

You can run the following command to generate the training data.

#### 0. Prepare training data.
```
wget https://huggingface.co/datasets/Aeala/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V4.3_unfiltered_cleaned_split.json
```

#### 1. Build training input (preprocessing training data)

with training data, base model generates hidden state of the last layer and logits. 
the hidden states of the last layer and logits are used later to train draft model.


**Llama 3.1 8B**
```
/opt/bin/cuda-reserve.py --num-gpus 4 python -m eagle.ge_data.allocation --outdir ../../eagle/tr-data-out/llama-3-1-chat-training-data --base-model-path /opt/models/Meta-Llama-3.1-8B-Instruct
```

**Llama 3.1 70B**

Currently, 70B model cannot be loaded in a single gpu so accelerate is used to build dataset.

```
/opt/bin/cuda-reserve.py --num-gpus 6 accelerate launch --config_file=multi_gpu_acc.yaml  ge_data_all_llama3-1-70Bchat.py \
--start=0 --end=68000 --index=0 --gpu_index 0 --outdir ../../train-data-llama-3-1-70b/sharegpt_0_67999_mufp16 --base-model-path /opt/models/meta-llama-3.1-70b-instruct
```



### Train the Auto-regression Head
Training draft model usually takes 20h (llama 3.1 8B) and 60h (Llama 3.1 70B) to finish.
```bash
python -m eagle.train.main \
	--tmpdir [training data path] \
	--basepath [base model path] \
	--cpdir [draft model output path] \
	--configpath [draft model config path]
```

**Llama 3.1 8B**
```
/opt/bin/cuda-reserve.py --num-gpus 2 --timeout 360000000 \
    accelerate launch -m --num_processes 2 --mixed_precision=bf16 eagle.train.main \
        --tmpdir train-data-llama-3-1 \
        --basepath /opt/models/Meta-Llama-3.1-8B-Instruct/ \
        --cpdir ~/scratch/eagle/out/llama-3-1-chat-with-3-1-data \
        --configpath eagle/train/llama-3-1-8b-instruct_config.json
```

**Llama 3.1 70B**
```
  /opt/bin/cuda-reserve.py --num-gpus 1 --timeout 360000000     python -m eagle.train.main \
	--tmpdir train-data-llama-3-1-70b-clean \
	--basepath /opt/models/meta-llama-3.1-70b-instruct \
	--cpdir ~/scratch/eagle/out/llama-3-1-70B-instruct \
	--configpath eagle/train/llama-3.1-70B-instruct_config.json
```



### Inference on custom models

If the original LLM structure differs from LLaMA and Mixtral, you can utilize EAGLE in two ways.

#### 1. Using the generic modeling_eagle.py

This approach directly encapsulates the native Transformers LLM. Here is an example. **Note: transformers version should be higher than 4.36.**

```python
from eagle.modeling_eagle import EAGLE
from transformers import AutoModelForCausalLM,AutoTokenizer

tokenizer=AutoTokenizer.from_pretrained(base_model_path)
model=AutoModelForCausalLM.from_pretrained("base_model_path",torch_dtype=torch.float16,device_map="auto",)
# for bs>1, the padding side should be right
if bs>1:
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

text=prompt1
# text=[prompt1,prompt2]
inputs = tokenizer(text, return_tensors="pt",padding=True)

eagle=EAGLE(model,eagle_path)
outs=eagle.generate(**inputs, max_new_tokens=200,temperature=0.0)
output=tokenizer.decode(outs)
# output=tokenizer.batch_decode(outs)
```

#### 2. Modifying the code of the model

Copy the modeling_basemodelname.py from the Transformers library and proceed to make modifications to leverage the pre-allocated kv_cache for enhanced speed in the base model. You can refer to model/modeling_llama_kv.py for guidance, where places that require modifications are annotated with # [MODIFIED]. These modifications are minimal.


## Evaluation
You can test the speed of EAGLE on MT-bench using the following command.
```bash
python -m eagle.evaluation.gen_ea_answer_vicuna(or gen_ea_answer_vicuna_llama2chat)\
		 --ea-model-path [path of EAGLE weight]\ 
		 --base-model-path [path of the original model]\
```
If you need specific acceleration ratios, you will also need to run the following command to get the speed of vanilla auto-regression.
```bash
python -m eagle.evaluation.gen_baseline_answer_vicuna\
		(or gen_ea_answer_vicuna_llama2chat)\
		 --ea-model-path [path of EAGLE weight]\ 
		 --base-model-path [path of the original model]\
```
The above two commands will each generate a .jsonl file that records the generation results and wall time. Then, you can use evaluation/speed.py to calculate the ratio of speeds.

## Inference
The inference code we provide automatically allocates model weights (loading a model across multiple GPUs), allowing you to run models that exceed the memory of a single GPU.

### With UI
We have provided a suggested web interface, which you can use by running the following command. After the model is fully loaded, a URL will be output in the terminal, which you can enter into your browser to access.
```bash
python -m eagle.application.webui --ea-model-path [path of EAGLE weight]\ 
		--base-model-path [path of the original model]\
		--model-type [vicuna or llama-2-chat]
```
### With Code
You can use our provided "eagenerate" for speedup generation just like using 'generate' from Hugging Face. Here is an example.
```python
from eagle.model.ea_model import EaModel
from fastchat.model import get_conversation_template
model = EaModel.from_pretrained(
    base_model_path=base_model_path,
    ea_model_path=EAGLE_model_path,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="auto"
)
model.eval()

your_message="Hello"

if use_llama_2_chat:
    conv = get_conversation_template("llama-2-chat")  
    sys_p = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
    conv.system_message = sys_p
    conv.append_message(conv.roles[0], your_message)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt() + " "

if use_vicuna:
    conv = get_conversation_template("vicuna")
    conv.append_message(conv.roles[0], your_message)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

input_ids=model.tokenizer([prompt]).input_ids
input_ids = torch.as_tensor(input_ids).cuda()
output_ids=model.eagenerate(input_ids,temperature=0.5,max_new_tokens=512)
output=model.tokenizer.decode(output_ids[0])
```

**_Note: Vicuna and LLaMA2-Chat are both chat models. You need to use the correct chat template, otherwise it will cause abnormal output from the model and affect the performance of EAGLE._**

### Batch size > 1

Here is an example. Note that left padding is needed.
```python
from eagle.modelbsne1.ea_model import EaModel
from fastchat.model import get_conversation_template

model = EaModel.from_pretrained(
    base_model_path=base_model_path,
    ea_model_path=EAGLE_model_path,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="auto"
)
# left padding
model.eval()
model.tokenizer.padding_side = "left"
model.tokenizer.pad_token = model.tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id

sys_p = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."

your_message="Compose an engaging travel blog post about a recent trip to Hawaii, highlighting cultural experiences and must-see attractions."
conv = get_conversation_template("llama-2-chat")
conv.system_message = sys_p
conv.append_message(conv.roles[0], your_message)
conv.append_message(conv.roles[1], None)
prompt1 = conv.get_prompt()+" "

your_message="Hello"
conv = get_conversation_template("llama-2-chat")
conv.system_message = sys_p
conv.append_message(conv.roles[0], your_message)
conv.append_message(conv.roles[1], None)
prompt2 = conv.get_prompt()+" "

input_s=model.tokenizer([prompt1,prompt2],return_tensors="pt",padding=True).to("cuda")
output_ids=model.eagenerate(input_s.input_ids,input_s.attention_mask,temperature=0.0,max_new_tokens=512,top_k=15)
output=model.tokenizer.batch_decode(output_ids)
print(output)

# vanilla auto-regression
# output_ids, new_token, idx=model.naivegenerate(input_s.input_ids,input_s.attention_mask,temperature=0.0,max_new_tokens=512,top_k=15,log=True)
```

## With gpt-fast

GPT-Fast primarily accelerates generation through quantization and compilation, which we have integrated into EAGLE. Here is the result of an experiment conducted on MT-bench with a single RTX3090, using LLaMA2-chat 7B.

| Precision 	    | fp16      | int4      |
|-------------------|-----------|-----------|
| vanilla          | 24.5 tokens/s     | N/A     |
| gpt-fast          | 55.1 tokens/s      | 106.9 tokens/s     |
| EAGLE+gpt-fast    | 100.2 tokens/s    | 160.4 tokens/s    |



<p align="center">
  <img src="./figs/eaglefast.gif" alt="demogif">
</p>

_Inference is conducted on a single RTX3090 GPU at int4 precision using the LLaMA2-chat 7B model. No additional training required._

In EAGLE, using gpt-fast only requires three steps: setting up the environment, quantizing weights, and modifying the model path.

### Setup

Switch to the *eaglefast* branch.

```bash
git clone https://github.com/SafeAILab/EAGLE.git
git checkout eaglefast
```

Install the Preview (Nightly) version of PyTorch with CUDA 12.1, do not use "pip install torch" as it installs the Stable version, which lacks some of the new features used by gpt-fast. 

_This is a requirement for gpt-fast, whereas other branches of eagle can use the Stable version of PyTorch._

### Quantizing Weights

Convert Huggingface weights to the format required by gpt-fast.

```bash
python convert/convert_hf_checkpoint.py --checkpoint_dir path_of_base_model
python convert/convert_hf_checkpoint_EAGLE.py --checkpoint_dir path_of_eagle
```

Quantize weights.

```bash
python -m model.quantize_llama --checkpoint_path path_of_base_model/model.pth
python -m model.quantize_EAGLE --checkpoint_path path_of_eagle/model.pth
```

