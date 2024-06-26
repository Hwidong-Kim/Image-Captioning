#pip install colorlog
#apt-get update
#apt-get install default-jdk
#apt-get install default-jre

NUM_GPU=1
GPU_IDS="1"
export OMP_NUM_THREADS=8
set CUDA_LAUNCH_BLOCKING=1
CUDA_VISIBLE_DEVICES=$GPU_IDS \

torchrun --nproc_per_node $NUM_GPU caption_mplug_scst.py \
    --config ./configs/caption_mplug_base_scst.yaml \
    --output_dir output/coco_caption_base_scst \
    --checkpoint ./output/coco_caption_base_1e-5/checkpoint_03.pth \
    --text_encoder /nas-alinlp/lcl193798/albef/google/bert-base-uncased \
    --text_decoder /nas-alinlp/lcl193798/albef/google/bert-base-uncased \
    --do_two_optim \
    --min_length 8 \
    --max_length 25 \
    --max_input_length 25
