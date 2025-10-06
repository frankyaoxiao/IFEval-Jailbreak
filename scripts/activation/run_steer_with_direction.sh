# Steers model to given direction
python -m src.activation_analysis.steer_cli \
  artifacts/activation_directions/allnew.pt \
  --layer 6 \
  --scale 1.5 \
  --prompt-set legacy \
  --variant-type base_plus_distractor \
  --samples-per-prompt 1 \
  --num-scenarios 10 \
  --max-new-tokens 64
