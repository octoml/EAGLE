echo "BASELINE Llama-3"
/opt/bin/cuda-reserve.py --num-gpus 1 python -m eagle.evaluation.gen_baseline_answer_llama3chat --base-model-path "meta-llama/Meta-Llama-3-8B-Instruct" --ea-model-path "yuhuili/EAGLE-LLaMA3-Instruct-8B" --temperature 0.0

echo "EAGLE-1 Llama-3"
/opt/bin/cuda-reserve.py --num-gpus 1 python -m eagle.evaluation.gen_ea_answer_llama3chat --base-model-path "meta-llama/Meta-Llama-3-8B-Instruct" --ea-model-path "yuhuili/EAGLE-LLaMA3-Instruct-8B" --temperature 0.0
