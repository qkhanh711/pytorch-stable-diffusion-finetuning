export CUDA_LAUNCH_BLOCKING=1
python run_finetune.py --dataset_path ../../Training_sampled_0_25 --rate 0.25 --phase inference
export CUDA_LAUNCH_BLOCKING=1
python run_finetune.py --dataset_path ../../Training_sampled_0_5 --rate 0.5 --phase inference
export CUDA_LAUNCH_BLOCKING=1
python run_finetune.py --phase inference