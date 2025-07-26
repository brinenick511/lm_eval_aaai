model_path=${HOME}/models/deepseek-ai/DeepSeek-V2-Lite-Chat

gpuid=2,3


model_args=pretrained=${model_path},trust_remote_code=True,dtype=bfloat16
model_args=pretrained=${model_path},parallelize=True,max_memory_per_gpu=20100100100
# model_args=${model_args},tp_size=2,chunked_prefill_size=4096
# model_args=${model_args},enable_p2p_check=True,disable_cuda_graph=True
# model_args=${model_args},mem_fraction_static=0.7
# model_args=${model_args},

tasks=arc_challenge,arc_easy,boolq,openbookqa,rte,winogrande
# tasks=arc_challenge


# HF_DATASETS_OFFLINE=1 \
# HF_HUB_OFFLINE=1 \
CUDA_VISIBLE_DEVICES=$gpuid \
lm_eval --model skip \
    --model_args ${model_args} \
    --tasks ${tasks} \
    --batch_size 4 \
    --trust_remote_code \
    --num_fewshot 0 \
    --output_path ${HOME}/data/lme/ \
    # --limit 4 \

