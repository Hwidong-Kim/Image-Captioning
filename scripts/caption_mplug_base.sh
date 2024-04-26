#pip install transformers==4.8.1
# pip install colorlog
# apt-get update
# apt-get install default-jdk
# apt-get install default-jre
# pip install git+git://github.com/j-min/language-evaluation@master
# python -c "import language_evaluation; language_evaluation.download('coco')"

NUM_GPU=1
GPU_IDS="1"
export OMP_NUM_THREADS=8
set CUDA_LAUNCH_BLOCKING=1
CUDA_VISIBLE_DEVICES=$GPU_IDS \

for lr in 1e-5
do
    torchrun --nproc_per_node $NUM_GPU caption_mplug.py \
    --config ./configs/caption_mplug_base.yaml \
    --output_dir /home/hdd1/hdkim/mplug/coco_caption_base_$lr \
    --checkpoint /home/hdd1/hdkim/mplug/coco_caption_base_1e-5/checkpoint_01.pth \
    --text_encoder bert-base-uncased \
    --text_decoder bert-base-uncased \
    --do_two_optim \
    --lr $lr \
    --min_length 8 \
    --max_length 25 \
    --max_input_length 25 \
    --do_amp \
    --evaluate
done
