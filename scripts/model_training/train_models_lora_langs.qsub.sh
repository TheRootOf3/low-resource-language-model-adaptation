#$ -l tmem=24G # Anything under 125G/num_gpus
#$ -l h_rt=48:00:00 # hh:mm:ss
#$ -l gpu=true
#$ -l gpu_type=rtx4090
#$ -pe gpu 4 # Less than 4
#  $ -l tscratch=200G
#$ -l hostname=thig.local

hostname

source /share/apps/source_files/python/python-3.9.5.source
source /home/aszablew/tmp/venv/bin/activate

cd /SAN/intelsys/llm/aszablew/UCL_FYP

date
 
for prop in 0 0.25 0.5
do
    for tokenizer_method in opt_100-add opt_2000-add opt_100-replace opt_2000-replace
    do 
        for lang in yor hau amh ibo
        do
            date
            echo "Training ${lang} with tokenizer=${tokenizer_method} and prop=${prop}..."
            accelerate launch train_model.py \
                --use_lora \
                --seed 42 \
                --dataset_name "/scratch0/aszablew/fyp/data/processed_dataset-opt-final/prop-${prop}/tokenizer-${tokenizer_method}_full-${lang}-opt/100M-wura-${lang}/" \
                --model_name_or_path "/SAN/intelsys/llm/aszablew/UCL_FYP/models_with_edited_embeddings/${tokenizer_method}_full-${lang}-opt" \
                --output_dir "./final_cpt_models/prop-${prop}/${tokenizer_method}/100M-${lang}" \
                --per_device_train_batch_size 2 \
                --gradient_accumulation_steps 8 \
                --per_device_eval_batch_size 2 \
                --num_train_epochs 1 \
                --checkpointing_steps 1600 \
                --logging_interval 1 \
                --eval_interval 250 \
                --learning_rate 1e-3 \
                --with_tracking \
                --report_to wandb \
                --wandb_args "project=fyp,name=prop-${prop}-${tokenizer_method}-${lang}-final-run" \
                --cache_dir /scratch0/aszablew/fyp/.cache &> logs/4_final_cpt/trainlog_prop-$prop-$tokenizer_method-$lang.log
        done
    done
done

MODEL=facebook/opt-1.3b

 
for lang in yor hau amh ibo
do
    date
    echo "Training $lang..."
    accelerate launch train_model.py \
        --use_lora \
        --seed 42 \
        --dataset_name /scratch0/aszablew/fyp/data/processed_dataset-opt-final/prop-0/tokenizer-base-opt/100M-wura-$lang/ \
        --model_name_or_path $MODEL \
        --output_dir ./final_cpt_models/prop-0/tokenizer-opt/100M-$lang \
        --per_device_train_batch_size 2 \
        --gradient_accumulation_steps 8 \
        --per_device_eval_batch_size 2 \
        --num_train_epochs 1 \
        --checkpointing_steps 1600 \
        --logging_interval 1 \
        --eval_interval 250 \
        --learning_rate 1e-3 \
        --with_tracking \
        --report_to wandb \
        --wandb_args "project=fyp,name=prop-0-tokenizer-opt-$lang-final-run" \
        --cache_dir /scratch0/aszablew/fyp/.cache &> logs/4_final_cpt/trainlog_prop-0_tokenizer-opt_$lang.log
done

date
