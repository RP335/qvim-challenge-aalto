
# running with passt
python src/qvim_mn_baseline/ex_qvim_alt.py \
    --model_type passt \
    --passt_model_identifier passt_s_swa_p16_128_ap476 \
    --passt_input_type raw \
    --project "qvim-passt-experiment" \
    --dataset_path ./data \
    --model_save_path ./checkpoints_custom \
    --batch_size 16 \
    --n_epochs 50 \
    --max_lr 1e-4 \
    --min_lr 1e-6 \
    --warmup_epochs 3 \
    --rampdown_epochs 20 \
    --num_gpus 1 \
    --num_workers 4

# running with beats

python src/qvim_mn_baseline/ex_qvim_alt.py \
    --model_type beats \
    --beats_checkpoint_path "beats_checkpoint/BEATs_iter3.pt" \
    --beats_savedir "./pretrained_models_cache/beats" \
    --project "qvim-beats-experiment" \
    --dataset_path ./data \
    --model_save_path ./checkpoints_custom \
    --batch_size 8 \
    --n_epochs 50 \
    --max_lr 5e-5 \
    --min_lr 1e-7 \
    --warmup_epochs 5 \
    --rampdown_epochs 25 \
    --num_gpus 1 \
    --num_workers 4

# running with panns

python src/qvim_mn_baseline/ex_qvim_alt.py \
    --model_type panns \
    --panns_checkpoint_path "/Users/rahul_peter/panns_data/Cnn14_mAP=0.431.pth" \
    --panns_input_type raw \
    --project "qvim-panns-experiment" \
    --dataset_path ./data \
    --model_save_path ./checkpoints_custom \
    --batch_size 16 \
    --n_epochs 50 \
    --max_lr 1e-4 \
    --min_lr 1e-6 \
    --warmup_epochs 3 \
    --rampdown_epochs 20 \
    --num_gpus 1 \
    --num_workers 4