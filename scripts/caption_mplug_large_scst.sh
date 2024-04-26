#pip install colorlog
#apt-get update
#apt-get install default-jdk
#apt-get install default-jre

NUM_GPU=2
GPU_IDS="0, 1"
export OMP_NUM_THREADS=8
set CUDA_LAUNCH_BLOCKING=1
CUDA_VISIBLE_DEVICES=$GPU_IDS \

torchrun --nproc_per_node $NUM_GPU caption_mplug_scst.py \
    --config ./configs/caption_mplug_large_scst.yaml \
    --output_dir /home/hdd1/hdkim/mplug/coco_caption_large_scst_img448_final \
    --checkpoint /home/hdd1/hdkim/mplug/coco_caption_large_scst_img448_final/checkpoint_best_0.pth \
    --text_encoder bert-base-uncased \
    --text_decoder bert-base-uncased \
    --do_two_optim \
    --min_length 8 \
    --max_length 25 \
    --max_input_length 25 \
    --accum_steps 16 \
    --evaluate
