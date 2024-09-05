if [[ $1 == "8B" ]]; then
  export BASE_MODEL_PATH="meta-llama/Meta-Llama-3.1-8B-Instruct"
  export DRAFT_MODEL_PATH="octoai/EAGLE-LLaMA3.1-Instruct-8B"
elif [[ $1 == "70B" ]]; then
  export BASE_MODEL_PATH="meta-llama/Meta-Llama-3.1-70B-Instruct"
  export DRAFT_MODEL_PATH="octoai/EAGLE-LLaMA3.1-Instruct-70B"
else
  echo "invalid model size"
  exit 1
fi

#echo "BASELINE $BASE_MODEL_PATH"
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
    --base-model-path "$BASE_MODEL_PATH" \
    --ea-model-path "$DRAFT_MODEL_PATH" \
    --temperature 0.0 \
    --tree-choices chain5

echo "Base: $BASE_MODEL_PATH"
echo "Drft: $DRAFT_MODEL_PATH"
echo "Mode: tree - EAGLE-1"
/opt/bin/cuda-reserve.py --num-gpus 1 \
  python -m eagle.evaluation.gen_ea_answer_llama3chat \
    --base-model-path "$BASE_MODEL_PATH" \
    --ea-model-path "$DRAFT_MODEL_PATH" \
    --temperature 0.0 \
    --tree-choices mc_sim_7b_63
