#$ -l tmem=2G
#$ -l h_rt=24:00:00 # hh:mm:ss
#$ -pe smp 16
#$ -N tokenize_cpt_dataset_tokenizers
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

for prop in 0 0.25 0.5
do
    for lang in hau amh ibo yor
    do
        tokenizer=opt_100-add_full-$lang-opt
        tokenizer_path=$base_tokenizer_path/added-opt-$lang/$tokenizer
        date
        echo "Processing $lang with prop=$prop and tokenizer=$tokenizer..."
        python3 training/prepare_ds_for_training.py \
            --wura_dataset_path ./data/wura/ \
            --wura_dataset_config_name $lang \
            --dolma_dataset_path allenai/dolma \
            --tokenizer_name_or_path $tokenizer_path \
            --model_name_or_path facebook/opt-1.3b \
            --trust_remote_code True \
            --cache_dir .cache \
            --output_dir $base_output_dir/prop-$prop/tokenizer-$tokenizer/100M-wura-$lang \
            --max_tokens 100000000 \
            --english_ratio $prop \
            --preprocessing_num_workers $num_processes

        tokenizer=opt_2000-add_full-$lang-opt
        tokenizer_path=$base_tokenizer_path/added-opt-$lang/$tokenizer
        date
        echo "Processing $lang with prop=$prop and tokenizer=$tokenizer..."
        python3 training/prepare_ds_for_training.py \
            --wura_dataset_path ./data/wura/ \
            --wura_dataset_config_name $lang \
            --dolma_dataset_path allenai/dolma \
            --tokenizer_name_or_path $tokenizer_path \
            --model_name_or_path facebook/opt-1.3b \
            --trust_remote_code True \
            --cache_dir .cache \
            --output_dir $base_output_dir/prop-$prop/tokenizer-$tokenizer/100M-wura-$lang \
            --max_tokens 100000000 \
            --english_ratio $prop \
            --preprocessing_num_workers $num_processes

        tokenizer=opt_100-replace_full-$lang-opt
        tokenizer_path=$base_tokenizer_path/replaced-opt-$lang/$tokenizer
        date
        echo "Processing $lang with prop=$prop and tokenizer=$tokenizer..."
        python3 training/prepare_ds_for_training.py \
            --wura_dataset_path ./data/wura/ \
            --wura_dataset_config_name $lang \
            --dolma_dataset_path allenai/dolma \
            --tokenizer_name_or_path $tokenizer_path \
            --model_name_or_path facebook/opt-1.3b \
            --trust_remote_code True \
            --cache_dir .cache \
            --output_dir $base_output_dir/prop-$prop/tokenizer-$tokenizer/100M-wura-$lang \
            --max_tokens 100000000 \
            --english_ratio $prop \
            --preprocessing_num_workers $num_processes

        tokenizer=opt_2000-replace_full-$lang-opt
        tokenizer_path=$base_tokenizer_path/replaced-opt-$lang/$tokenizer
        date
        echo "Processing $lang with prop=$prop and tokenizer=$tokenizer..."
        python3 training/prepare_ds_for_training.py \
            --wura_dataset_path ./data/wura/ \
            --wura_dataset_config_name $lang \
            --dolma_dataset_path allenai/dolma \
            --tokenizer_name_or_path $tokenizer_path \
            --model_name_or_path facebook/opt-1.3b \
            --trust_remote_code True \
            --cache_dir .cache \
            --output_dir $base_output_dir/prop-$prop/tokenizer-$tokenizer/100M-wura-$lang \
            --max_tokens 100000000 \
            --english_ratio $prop \
            --preprocessing_num_workers $num_processes
    done

    cp -r $base_output_dir/prop-$prop /SAN/intelsys/llm/aszablew/UCL_FYP/processed_dataset-opt-final/
done

date
