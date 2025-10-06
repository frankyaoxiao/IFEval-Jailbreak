# Test similarity between model responses and activation direction
python -m src.activation_analysis.activation_similarity \
  artifacts/activation_directions/allnew.pt \
  --model olmo7b_dpo \
  --prompt-set rollout_pairs \
  --num-prompts 6 \
  --iterations 5 \
  --temperature 0.7 \
  --do-sample \
