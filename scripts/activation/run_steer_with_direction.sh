# Steers model to given direction
python -m src.activation_analysis.steer_cli \
  artifacts/activation_directions/kl_weighted.pt \
  --layer 16 \
  --scale 1.5 \
  --prompt-set rollout_pairs \
  --variant-type base_plus_distractor \
  --samples-per-prompt 10 \
  --num-scenarios 6 \
  --max-new-tokens 64
