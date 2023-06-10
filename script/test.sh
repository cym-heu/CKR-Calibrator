export model_name='resnet50'
export dataset_name='cifar100'
export SOTA='basic'
export device='cuda:0'
export layers='1'
export model_type='ori' # optim or ori

python ../test.py \
    --model_name ${model_name} \
    --dataset_name ${dataset_name} \
    --SOTA ${SOTA} \
    --device ${device} \
    --model_type ${model_type} \
    --layers ${layers} \
    --corruption