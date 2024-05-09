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

MODEL=facebook/opt-1.3b

# for lang in yor hau amh ibo swa
for lang in yor hau
do 
    for lr in 1e-2 5e-3 1e-3 5e-4 1e-4 5e-5 1e-5
    do
        for prop in 0.25 0
        do
            date
            echo "Training $lang with lr=$lr and prop=$prop..."
            accelerate launch train_model.py --use_lora --seed 42 --dataset_name /scratch0/aszablew/fyp/data/processed_dataset-opt/prop-$prop/100M-wura-$lang/ --model_name_or_path $MODEL --output_dir prop-$prop-lora_trained_models_opt1.3b-embeddings-all/100M-$lang-$lr --per_device_train_batch_size 2 --gradient_accumulation_steps 8 --per_device_eval_batch_size 2 --num_train_epochs 1 --checkpointing_steps 1600 --logging_interval 1 --eval_interval 300 --learning_rate $lr --with_tracking --report_to wandb --wandb_args "project=fyp,name=$prop-$lang-$lr-thig-run-opt1.3b-lora-embed" --cache_dir /scratch0/aszablew/fyp/.cache &> logs/3_experiment_lr_lora_embeddings_training_logs/trainlog_$lang-$lr-$prop.log
        done
    done
done

date
