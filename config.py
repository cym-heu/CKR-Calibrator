# 路径
root_dir = '/home/gjs/code/robust_eval/'
output_dir = root_dir + 'output/'
images_dir = root_dir + 'experiment/images'


# 数据集参数设置
def get_data_config(data_name):
    config = {}
    if data_name == 'cifar10':
        config['in_channels'], config['num_classes'], config['batch_size'] = 3, 10, 16
        config['data_dir'] = '/home/gjs/output/data/cifar10/images'
        config['train_dir'] = config['data_dir'] + '/train'
        config['test_dir'] = config['data_dir'] + '/test'

    if data_name == 'cifar100':
        config['in_channels'], config['num_classes'], config['batch_size'] = 3, 100, 16
        config['data_dir'] = '/home/gjs/output/data/cifar100/images'
        config['train_dir'] = config['data_dir'] + '/train'
        config['test_dir'] = config['data_dir'] + '/test'

            
    if data_name == 'mnist':
        config['in_channels'], config['num_classes'], config['batch_size'] = 1, 10, 16
        config['data_dir'] = '/home/gjs/output/data/mnist/images'
        config['train_dir'] = config['data_dir'] + '/train'
        config['test_dir'] = config['data_dir'] + '/test'

    if data_name == 'cifar10C':
        config['in_channels'], config['num_classes'], config['batch_size'] = 3, 10, 16
        config['data_dir'] = '/home/gjs/datasets/CIFAR-10-C'
        config['corruptions'] = ['gaussian_noise', 'shot_noise', 'speckle_noise', 'impulse_noise', \
                                 'defocus_blur', 'gaussian_blur', 'motion_blur', 'zoom_blur', \
                                 'snow', 'fog', 'brightness', 'contrast', 'elastic_transform', \
                                 'pixelate', 'jpeg_compression', 'spatter', 'saturate', 'frost']

    if data_name == 'cifar100C':
        config['in_channels'], config['num_classes'], config['batch_size'] = 3, 100, 16
        config['data_dir'] = '/home/gjs/datasets/CIFAR-100-C'
        config['corruptions'] = ['gaussian_noise', 'shot_noise', 'speckle_noise', 'impulse_noise', \
                                 'defocus_blur', 'gaussian_blur', 'motion_blur', 'zoom_blur', \
                                 'snow', 'fog', 'brightness', 'contrast', 'elastic_transform', \
                                 'pixelate', 'jpeg_compression', 'spatter', 'saturate', 'frost']
        
    if data_name == 'mnistC':
        config['in_channels'], config['num_classes'], config['batch_size'] = 1, 10, 16
        config['data_dir'] = '/home/gjs/datasets/mnist_c'
        config['corruptions'] = ['brightness', 'canny_edges', 'dotted_line', 'fog', 'glass_blur', \
                                 'identity', 'impulse_noise', 'motion_blur', 'rotate', 'scale', \
                                 'shear', 'shot_noise', 'spatter', 'stripe', 'translate', 'zigzag']

        # config['data_dir'] = '/home/gjs/datasets/tiny-imagenet-200'
    return config


# 训练参数
lr=0.01
momentum=0.9
weight_decay=5e-4
num_epochs=200

# 超参
alpha=0.7 # 选取不鲁棒卷积核的超参
beta=1e-1 # triplet loss的权重
