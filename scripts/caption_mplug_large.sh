#pip install colorlog
#apt-get update
#apt-get install default-jdk
#apt-get install default-jre
#pip install git+git://github.com/j-min/language-evaluation@master
#python -c "import language_evaluation; language_evaluation.download('coco')"

NUM_GPU=2
GPU_IDS="0, 1"
export OMP_NUM_THREADS=8
set CUDA_lAUNCH_BLOCKING=1
CUDA_VISIBLE_DEVICES=$GPU_IDS \

for lr in 1e-5
do
    torchrun --nproc_per_node $NUM_GPU caption_mplug.py \
    --config ./configs/caption_mplug_large.yaml \
    --output_dir /home/hdd1/hdkim/mplug/coco_caption_large_scst_img448_final_test \
    --checkpoint /home/hdd1/hdkim/mplug/coco_caption_large_scst_img448_final/checkpoint_best_0.pth \
    --do_two_optim \
    --lr $lr \
    --min_length 8 \
    --max_length 25 \
    --max_input_length 25 \
    --accum_steps 16 \
    --evaluate
done
