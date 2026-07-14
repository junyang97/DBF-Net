from torchvision import transforms
from fusion.utils import *
from datetime import datetime

class setting_config:

    network = 'fusion_egalayerall'
    model_config = {
        'num_classes': 1,
        'input_channels': 3,

    }

    datasets = 'polyp'
    if datasets == 'isic18':
        data_path = '/home/admin123/sjy/xiangmu1/VmTu1/data/isic2018/'
    elif datasets == 'isic17':
        data_path = '/home/admin123/sjy/xiangmu1/VmTu1/data/isic2017/'
    elif datasets == 'polyp':
        data_path = '/home/admin123/sjy/xiangmu1/VmTu1/data/PolypSegData/'
    else:
        raise Exception('datasets in not right!')

    criterion = SegLoss(lambda_dice=1.0, lambda_iou=1.0)
    transunet_pretrained_path = "../model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz"
    num_classes = 1
    input_size_h = 256
    input_size_w = 256
    input_channels = 3
    distributed = False
    local_rank = -1
    num_workers = 0
    seed = 42
    world_size = None
    rank = None
    amp = False
    gpu_id = '0'
    batch_size = 16
    epochs = 200

    work_dir = 'results/' + network + '_' + datasets + '_' + datetime.now().strftime('%A_%d_%B_%Y_%Hh_%Mm_%Ss') + '/'

    print_interval = 20
    val_interval = 2
    save_interval = 10
    threshold = 0.5


    train_transformer = transforms.Compose([
        myRandomResizedCrop(p=0.5, size=(input_size_h, input_size_w), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        myRandomHorizontalFlip(p=0.5),
        myRandomVerticalFlip(p=0.5),
        myRandomRotation(p=0.5, degree=[0, 360]),
        myRandomAffine(p=0.3, degrees=15, translate=0.1, scale=0.1, shear=10),

        myColorJitter(p=0.3),
        myRandomGamma(p=0.3, gamma_range=[0.7, 1.5]),
        myGaussianNoise(p=0.3),
        myCutout(p=0.2, num_holes=1, max_h_size=32, max_w_size=32),
        myRandomErasing(p=0.2, scale=(0.02, 0.1), ratio=(0.3, 3.3)),
        myResize(input_size_h, input_size_w),
        myNormalize(datasets, train=True)

    ])
    test_transformer = transforms.Compose([
        myResize(input_size_h, input_size_w),
        myNormalize(datasets, train=False),

    ])

    opt = 'AdamW'
    assert opt in ['Adadelta', 'Adagrad', 'Adam', 'AdamW', 'Adamax', 'ASGD', 'RMSprop', 'Rprop', 'SGD'], 'Unsupported optimizer!'
    if opt == 'Adadelta':
        lr = 0.01
        rho = 0.9
        eps = 1e-6
        weight_decay = 0.05
    elif opt == 'Adagrad':
        lr = 0.01
        lr_decay = 0
        eps = 1e-10
        weight_decay = 0.05
    elif opt == 'Adam':
        lr = 0.001
        betas = (0.9, 0.999)
        eps = 1e-8
        weight_decay = 0.0001
        amsgrad = False
    elif opt == 'AdamW':
        lr = 1e-4
        betas = (0.9, 0.999)
        eps = 1e-8
        weight_decay = 1e-2
        amsgrad = False
    elif opt == 'Adamax':
        lr = 2e-3
        betas = (0.9, 0.999)
        eps = 1e-8
        weight_decay = 0
    elif opt == 'ASGD':
        lr = 0.01
        lambd = 1e-4
        alpha = 0.75
        t0 = 1e6
        weight_decay = 0
    elif opt == 'RMSprop':
        lr = 1e-2
        momentum = 0
        alpha = 0.99
        eps = 1e-8
        centered = False
        weight_decay = 0
    elif opt == 'Rprop':
        lr = 1e-2
        etas = (0.5, 1.2)
        step_sizes = (1e-6, 50)
    elif opt == 'SGD':
        lr = 0.01
        momentum = 0.9
        weight_decay = 0.05
        dampening = 0
        nesterov = False

    sch = 'CosineAnnealingLR'
    if sch == 'StepLR':
        step_size = epochs // 5
        gamma = 0.5
        last_epoch = -1
    elif sch == 'MultiStepLR':
        milestones = [60, 120, 150]
        gamma = 0.1
        last_epoch = -1
    elif sch == 'ExponentialLR':
        gamma = 0.99
        last_epoch = -1
    elif sch == 'CosineAnnealingLR':
        T_max = 200
        eta_min = 1e-6
        last_epoch = -1
    elif sch == 'ReduceLROnPlateau':
        mode = 'min'
        factor = 0.1
        patience = 10
        threshold = 0.0001
        threshold_mode = 'rel'
        cooldown = 0
        min_lr = 0
        eps = 1e-08
    elif sch == 'CosineAnnealingWarmRestarts':
        T_0 = 50
        T_mult = 2
        eta_min = 1e-6
        last_epoch = -1
    elif sch == 'WP_MultiStepLR':
        warm_up_epochs = 10
        gamma = 0.1
        milestones = [125, 225]
    elif sch == 'WP_CosineLR':
        warm_up_epochs = 20