#$ -l tmem=15G # Anything under 125G/num_gpus
#$ -l h_rt=90:00:00 # hh:mm:ss
#$ -pe smp 8
#$ -N train_wura_tokenizers
#$ -R y


hostname

source /share/apps/source_files/python/python-3.9.5.source
source /home/aszablew/tmp/venv/bin/activate

cd /SAN/intelsys/llm/aszablew/UCL_FYP

date

MODEL=facebook/opt-1.3b

python3 train_wura_tokenizer.py \
    --langs "hau,ibo,yor,amh,eng" \
    --old_tokenizer $MODEL \
    --output_dir "./trained_tokenizers-opt/"

date
