#$ -l tmem=24G # Anything under 125G/num_gpus
#$ -l h_rt=48:00:00 # hh:mm:ss
#$ -l gpu=true
#$ -l gpu_type=rtx4090
#$ -pe gpu 4 # Less than 4
# $ -l tscratch=200G
#$ -l hostname=thig.local

hostname

# source /share/apps/source_files/python/python-3.9.5.source
# source /home/aszablew/tmp/venv/bin/activate

cd /SAN/intelsys/llm/aszablew/UCL_FYP/evaluation


BASE_PATH=/SAN/intelsys/llm/aszablew/UCL_FYP/evaluation


GENERATIONS_PATH=$BASE_PATH/generations/
OUTPUT_PATH=$BASE_PATH/results/

date

for lang in amh hau ibo yor
do 
    echo "Evaluating zero-shot baseline lang=${lang}..."
    python3 evaluations/opt_eval.py \
        --generations_directory "${GENERATIONS_PATH}/zero-shot/baseline/${lang}" \
        --output_directory "${OUTPUT_PATH}/zero-shot/baseline/${lang}" \
        --lang $lang

    echo "Evaluating 3-shot baseline lang=${lang}..."
    python3 evaluations/opt_eval.py \
        --generations_directory "${GENERATIONS_PATH}/3-shot/baseline/${lang}" \
        --output_directory "${OUTPUT_PATH}/3-shot/baseline/${lang}" \
        --lang $lang \
        --no_ner
done

date