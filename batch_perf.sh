model_path=${HOME}/models/allenai/OLMoE-1B-7B-0125-Instruct
model_path=${HOME}/models/deepseek-ai/DeepSeek-V2-Lite-Chat

gpuid=1,2
tasks=arc_challenge,arc_easy,boolq,openbookqa,rte,winogrande
topk_list=(6 5 4 3 2 1 )

model_args=pretrained=${model_path},trust_remote_code=True,dtype=bfloat16
model_args=${model_args},parallelize=True,max_memory_per_gpu=20100100100
model_args=${model_args},attn_implementation=flash_attention_2
model_args=${model_args},skipping=${topk}
# model_args=${model_args},

for topk in "${topk_list[@]}"; do
    output_path="${HOME}/data/lme/deepseek_skipping/${topk}.json"

    TOKENIZERS_PARALLELISM=true \
    CUDA_VISIBLE_DEVICES=$gpuid \
    lm_eval --model skip \
        --model_args "${model_args}" \
        --tasks ${tasks} \
        --batch_size 8 \
        --gen_kwargs temperature=0 \
        --trust_remote_code \
        --num_fewshot 0 \
        --output_path "${output_path}"
done