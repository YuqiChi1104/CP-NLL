CUDA_VISIBLE_DEVICES=0 
python ./SED/generate_result_cifar100lt.py \
    --dataset cifar100lt \
    --closeset-ratio 0.9 \
    --imb_factor 50 \
    --warmup-epoch 20 \
    --epoch 100 \
    --batch-size 128 \
    --lr 0.05 \
    --warmup-lr 0.05 \
    --noise-type symmetric \
    --lr-decay cosine:20,5e-4,100 \
    --opt sgd \
    --save-weights True 
