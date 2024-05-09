#$ -l tmem=24G # Anything under 125G/num_gpus
#$ -l h_rt=5:00:00 # hh:mm:ss
#$ -l gpu=true
#$ -l gpu_type=rtx4090
#$ -pe gpu 4 # Less than 4
# $ -l tscratch=200G
#$ -l hostname=thig.local

hostname

source /share/apps/source_files/python/python-3.9.5.source
source /home/aszablew/tmp/venv/bin/activate

cd /SAN/intelsys/llm/aszablew/UCL_FYP/evaluation

# # CPT
# MODELS_PATH=/SAN/intelsys/llm/aszablew/UCL_FYP/final_cpt_models
# OUTPUT_BASE_PATH=/SAN/intelsys/llm/aszablew/UCL_FYP/evaluation/few-shot-generations
# LOGS_PREFIX=final_eval-few-shot

# IT
MODELS_PATH=/SAN/intelsys/llm/aszablew/UCL_FYP/final_aya_models
OUTPUT_BASE_PATH=/SAN/intelsys/llm/aszablew/UCL_FYP/evaluation/few-shot-generations_aya
LOGS_PREFIX=final_eval_aya-few-shot


prop=0
tokenizer=tokenizer-opt

PROC_ARRAY=()

date

lang=yor
python3 ./scripts/prompt_opt-few-shot.py \
    --model_path "${MODELS_PATH}/prop-${prop}/${tokenizer}/100M-${lang}" \
    --output_dir "${OUTPUT_BASE_PATH}/prop-${prop}/${tokenizer}/${lang}" \
    --lang $lang \
    --device cuda:0 &> "${LOGS_PREFIX}_${prop}-${tokenizer}-${lang}.log" &
PROC_ARRAY+=($!)
date

lang=hau
python3 ./scripts/prompt_opt-few-shot.py \
    --model_path "${MODELS_PATH}/prop-${prop}/${tokenizer}/100M-${lang}" \
    --output_dir "${OUTPUT_BASE_PATH}/prop-${prop}/${tokenizer}/${lang}" \
    --lang $lang \
    --device cuda:1 &> "${LOGS_PREFIX}_${prop}-${tokenizer}-${lang}.log" &
PROC_ARRAY+=($!)
date

lang=amh
python3 ./scripts/prompt_opt-few-shot.py \
    --model_path "${MODELS_PATH}/prop-${prop}/${tokenizer}/100M-${lang}" \
    --output_dir "${OUTPUT_BASE_PATH}/prop-${prop}/${tokenizer}/${lang}" \
    --lang $lang \
    --device cuda:2 &> "${LOGS_PREFIX}_${prop}-${tokenizer}-${lang}.log" &
PROC_ARRAY+=($!)
date

lang=ibo
python3 ./scripts/prompt_opt-few-shot.py \
    --model_path "${MODELS_PATH}/prop-${prop}/${tokenizer}/100M-${lang}" \
    --output_dir "${OUTPUT_BASE_PATH}/prop-${prop}/${tokenizer}/${lang}" \
    --lang $lang \
    --device cuda:3 &> "${LOGS_PREFIX}_${prop}-${tokenizer}-${lang}.log" &
PROC_ARRAY+=($!)


echo "Waiting for ${PROC_ARRAY[@]}..."
wait ${PROC_ARRAY[@]}

date
