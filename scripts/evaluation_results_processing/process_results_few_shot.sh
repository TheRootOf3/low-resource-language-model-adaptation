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


# CPT
# GENERATIONS_PATH=$BASE_PATH/generations/3-shot/wura
# OUTPUT_PATH=$BASE_PATH/results/3-shot/wura

# IT
GENERATIONS_PATH=$BASE_PATH/generations/3-shot/aya
OUTPUT_PATH=$BASE_PATH/results/3-shot/aya

date

for prop in 0 0.25 0.5
do
    for tokenizer in opt_100-add opt_2000-add opt_100-replace opt_2000-replace
    do 
        for lang in amh hau ibo yor
        do 
            echo "Evaluating prop=${prop}, tokenizer=${tokenizer}, lang=${lang}..."
            python3 evaluations/opt_eval.py \
                --generations_directory "${GENERATIONS_PATH}/prop-${prop}/${tokenizer}/${lang}" \
                --output_directory "${OUTPUT_PATH}/prop-${prop}/${tokenizer}/${lang}" \
                --lang $lang \
                --no_ner
        done
    done
done

prop=0
tokenizer=tokenizer-opt
for lang in amh hau ibo yor
do 
    echo "Evaluating prop=${prop}, tokenizer=${tokenizer}, lang=${lang}..."
    python3 evaluations/opt_eval.py \
        --generations_directory "${GENERATIONS_PATH}/prop-${prop}/${tokenizer}/${lang}" \
        --output_directory "${OUTPUT_PATH}/prop-${prop}/${tokenizer}/${lang}" \
        --lang $lang \
        --no_ner
done

date