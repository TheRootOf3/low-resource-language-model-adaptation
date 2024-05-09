#$ -l tmem=24G # Anything under 125G/num_gpus
#$ -l h_rt=1:00:00 # hh:mm:ss
#$ -l gpu=true
#$ -l gpu_type=rtx4090
#$ -pe gpu 1 # Less than 4
# $ -l tscratch=200G
# $ -l hostname=thig.local

hostname

source /share/apps/source_files/python/python-3.9.5.source
source /home/aszablew/tmp/venv/bin/activate

cd /SAN/intelsys/llm/aszablew/UCL_FYP/evaluation

MODELS_PATH=facebook/opt-1.3b
OUTPUT_BASE_PATH=/SAN/intelsys/llm/aszablew/UCL_FYP/evaluation/few-shot-generations_baselines
LOGS_PREFIX=final_eval_baseline_opt-few-shot

date

for lang in yor hau amh ibo
do 
    date
    echo "Processing ${lang}..."
    python3 ./scripts/prompt_opt-few-shot.py \
        --model_path $MODELS_PATH \
        --output_dir "${OUTPUT_BASE_PATH}/opt-1.3b" \
        --lang $lang \
        --device cuda &> "${LOGS_PREFIX}-${lang}.log"
done

date