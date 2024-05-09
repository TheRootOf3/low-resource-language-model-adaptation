#$ -l tmem=1G
#$ -l h_rt=2:00:00 # hh:mm:ss
#$ -pe smp 16
#$ -N tokenize_cpt_dataset_base
#$ -R y

hostname

source /share/apps/source_files/python/python-3.9.5.source
source /home/aszablew/tmp/venv/bin/activate

cd /SAN/intelsys/llm/aszablew/UCL_FYP

date

base_tokenizer_path=/SAN/intelsys/llm/aszablew/UCL_FYP/trained_tokenizers
base_output_dir=/scratch0/aszablew/processed_dataset-opt-final

# num_processes=$NSLOTS
num_processes=128

for prop in 0
do
    for lang in hau amh ibo yor
    do
        tokenizer=facebook/opt-1.3b
        date
        echo "Processing $lang with prop=$prop and tokenizer=$tokenizer..."
        python3 training/prepare_ds_for_training.py \
            --wura_dataset_path ./data/wura/ \
            --wura_dataset_config_name $lang \
            --dolma_dataset_path allenai/dolma \
            --tokenizer_name_or_path $tokenizer \
            --model_name_or_path facebook/opt-1.3b \
            --trust_remote_code True \
            --cache_dir .cache \
            --output_dir $base_output_dir/prop-$prop/tokenizer-base-opt/100M-wura-$lang \
            --max_tokens 100000000 \
            --english_ratio $prop \
            --preprocessing_num_workers $num_processes
    done
done

date
