export BASE_MODEL_PATH="meta-llama/Meta-Llama-3-8B-Instruct"
export DRAFT_MODEL_PATH="yuhuili/EAGLE-LLaMA3-Instruct-8B"

#echo "BASELINE Llama-3"
#/opt/bin/cuda-reserve.py --num-gpus 1 \
#  python -m eagle.evaluation.gen_baseline_answer_llama3chat \
#    --base-model-path $BASE_MODEL_PATH \
#    --ea-model-path $DRAFT_MODEL_PATH \
#    --temperature 0.0

echo "Base: $BASE_MODEL_PATH"
echo "Drft: $DRAFT_MODEL_PATH"
echo "Mode: chain - length 5"
/opt/bin/cuda-reserve.py --num-gpus 1 \
  python -m eagle.evaluation.gen_ea_answer_llama3chat \
    --base-model-path $BASE_MODEL_PATH \
    --ea-model-path $DRAFT_MODEL_PATH \
    --temperature 0.0 \
    --tree-choices chain5

echo "Base: $BASE_MODEL_PATH"
echo "Drft: $DRAFT_MODEL_PATH"
echo "Mode: tree - EAGLE-1"
/opt/bin/cuda-reserve.py --num-gpus 1 \
  python -m eagle.evaluation.gen_ea_answer_llama3chat \
    --base-model-path $BASE_MODEL_PATH \
    --ea-model-path $DRAFT_MODEL_PATH \
    --temperature 0.0 \
    --tree-choices mc_sim_7b_63

export BASE_MODEL_PATH="meta-llama/Meta-Llama-3-8B-Instruct"
export DRAFT_MODEL_PATH="octoai/EAGLE-LLaMA3-Instruct-8B"

echo "Base: $BASE_MODEL_PATH"
echo "Drft: $DRAFT_MODEL_PATH"
echo "Mode: chain - length 5"
/opt/bin/cuda-reserve.py --num-gpus 1 \
  python -m eagle.evaluation.gen_ea_answer_llama3chat \
    --base-model-path $BASE_MODEL_PATH \
    --ea-model-path $DRAFT_MODEL_PATH \
    --temperature 0.0 \
    --tree-choices chain5

echo "Base: $BASE_MODEL_PATH"
echo "Drft: $DRAFT_MODEL_PATH"
echo "Mode: tree - EAGLE-1"
/opt/bin/cuda-reserve.py --num-gpus 1 \
  python -m eagle.evaluation.gen_ea_answer_llama3chat \
    --base-model-path $BASE_MODEL_PATH \
    --ea-model-path $DRAFT_MODEL_PATH \
    --temperature 0.0 \
    --tree-choices mc_sim_7b_63
