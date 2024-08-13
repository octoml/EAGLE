echo "BASELINE Llama-2"
/opt/bin/cuda-reserve.py --num-gpus 1 python -m eagle.evaluation.gen_baseline_answer_llama2chat --base-model-path "meta-llama/Llama-2-7b-chat-hf" --ea-model-path "yuhuili/EAGLE-llama2-chat-7B" --temperature 0.0

echo "EAGLE-1 Llama-2"
/opt/bin/cuda-reserve.py --num-gpus 1 python -m eagle.evaluation.gen_ea_answer_llama2chat --base-model-path "meta-llama/Llama-2-7b-chat-hf" --ea-model-path "yuhuili/EAGLE-llama2-chat-7B" --temperature 0.0
