export model_name='resnext'
export dataset_name='cifar10'
export SOTA='basic'
export device='cuda:0'
export layers='1'

python ../train_up.py \
    --model_name ${model_name} \
    --dataset_name ${dataset_name} \
    --SOTA ${SOTA} \
    --layers ${layers} \
    --device ${device}