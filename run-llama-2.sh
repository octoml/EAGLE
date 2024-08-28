echo "BASELINE Llama-2"
/opt/bin/cuda-reserve.py --num-gpus 1 python -m eagle.evaluation.gen_baseline_answer_llama2chat --base-model-path "meta-llama/Llama-2-7b-chat-hf" --ea-model-path "yuhuili/EAGLE-llama2-chat-7B" --temperature 0.0

echo "Base: meta-llama/Llama-2-7b-chat-hf"
echo "Drft: yuhuili/EAGLE-llama2-chat-7B"
echo "Mode: chain - length 5"
/opt/bin/cuda-reserve.py --num-gpus 1 python -m eagle.evaluation.gen_ea_answer_llama2chat --base-model-path "meta-llama/Llama-2-7b-chat-hf" --ea-model-path "yuhuili/EAGLE-llama2-chat-7B" --temperature 0.0 --tree-choices chain5

echo "Base: meta-llama/Llama-2-7b-chat-hf"
echo "Drft: yuhuili/EAGLE-llama2-chat-7B"
echo "Mode: tree - EAGLE-1"
/opt/bin/cuda-reserve.py --num-gpus 1 python -m eagle.evaluation.gen_ea_answer_llama2chat --base-model-path "meta-llama/Llama-2-7b-chat-hf" --ea-model-path "yuhuili/EAGLE-llama2-chat-7B" --temperature 0.0 --tree-choices mc_sim_7b_63
