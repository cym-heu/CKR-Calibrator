export model_name='squeezenet'
export dataset_name='cifar10'
export SOTA='autoaug'
export device='cuda:0'

python ../train.py \
    --model_name ${model_name} \
    --dataset_name ${dataset_name} \
    --SOTA ${SOTA} \
    --device ${device}