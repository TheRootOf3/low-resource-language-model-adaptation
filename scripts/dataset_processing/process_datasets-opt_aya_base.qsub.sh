#$ -l tmem=1G
#$ -l h_rt=1:00:00 # hh:mm:ss
#$ -pe smp 16
#$ -N split_datasets
#$ -R y

hostname

source /share/apps/source_files/python/python-3.9.5.source
source /home/aszablew/tmp/venv/bin/activate

cd /SAN/intelsys/llm/aszablew/UCL_FYP

date

base_tokenizer_path=/SAN/intelsys/llm/aszablew/UCL_FYP/trained_tokenizers
base_output_dir=/SAN/intelsys/llm/aszablew/UCL_FYP/processed_dataset-opt-final

prop=0

# num_processes=$NSLOTS
num_processes=16

for lang in hau amh ibo yor
do
    tokenizer=facebook/opt-1.3b
    date
    echo "Processing $lang with prop=$prop and tokenizer=$tokenizer..."
    python3 training/prepare_aya_ds.py \
        --dataset_name CohereForAI/aya_dataset \
        --dataset_config_name default \
        --lang $lang \
        --tokenizer_name_or_path $tokenizer \
        --trust_remote_code True \
        --cache_dir .cache \
        --output_dir $base_output_dir/prop-$prop/tokenizer-base-opt/aya-$lang \
        --english_ratio $prop \
        --preprocessing_num_workers $num_processes
done

date
