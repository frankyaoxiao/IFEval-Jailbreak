python evaluate_safety.py \
-n 200 \
--models olmo7b_dpo \
--model-override "olmo7b_dpo=/mnt/polished-lake/home/fxiao/openinstruct/output/olmo2_7b_dpo_local/olmo2_7b_dpo_filtered_30k__42__1760398140/outputs" \
--prompt-set rollout_distractor_only_new \
--run-name "safety_override" \
--temperature 0.7 \
--enable-compliance \
--generate-plots