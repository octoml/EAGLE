echo "BASELINE Llama-3"
/opt/bin/cuda-reserve.py --num-gpus 1 python -m eagle.evaluation.gen_baseline_answer_llama3chat --base-model-path "meta-llama/Meta-Llama-3-8B-Instruct" --ea-model-path "yuhuili/EAGLE-LLaMA3-Instruct-8B" --temperature 0.0

echo "Base: meta-llama/Meta-Llama-3-8B-Instruct"
echo "Drft: yuhuili/EAGLE-LLaMA3-Instruct-8B"
echo "Mode: chain - length 5"
/opt/bin/cuda-reserve.py --num-gpus 1 python -m eagle.evaluation.gen_ea_answer_llama3chat --base-model-path "meta-llama/Meta-Llama-3-8B-Instruct" --ea-model-path "yuhuili/EAGLE-LLaMA3-Instruct-8B" --temperature 0.0 --tree-choices chain5

echo "Base: meta-llama/Meta-Llama-3-8B-Instruct"
echo "Drft: Hyunsung llama-3-chat fine tune"
echo "Mode: chain - length 5"
/opt/bin/cuda-reserve.py --num-gpus 1 python -m eagle.evaluation.gen_ea_answer_llama3chat --base-model-path "meta-llama/Meta-Llama-3-8B-Instruct" --ea-model-path ~/models/llama-3-chat --temperature 0.0 --tree-choices chain5

echo "Base: meta-llama/Meta-Llama-3-8B-Instruct"
echo "Drft: yuhuili/EAGLE-LLaMA3-Instruct-8B"
echo "Mode: tree - EAGLE-1"
/opt/bin/cuda-reserve.py --num-gpus 1 python -m eagle.evaluation.gen_ea_answer_llama3chat --base-model-path "meta-llama/Meta-Llama-3-8B-Instruct" --ea-model-path "yuhuili/EAGLE-LLaMA3-Instruct-8B" --temperature 0.0 --tree-choices mc_sim_7b_63

echo "Base: meta-llama/Meta-Llama-3-8B-Instruct"
echo "Drft: Hyunsung llama-3-chat fine tune"
echo "Mode: tree - EAGLE-1"
/opt/bin/cuda-reserve.py --num-gpus 1 python -m eagle.evaluation.gen_ea_answer_llama3chat --base-model-path "meta-llama/Meta-Llama-3-8B-Instruct" --ea-model-path ~/models/llama-3-chat --temperature 0.0 --tree-choices mc_sim_7b_63
