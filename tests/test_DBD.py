import os.path as osp

import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision
from torchvision.datasets import DatasetFolder
from torchvision.transforms import Compose, RandomHorizontalFlip, ToTensor, ToPILImage, Resize, \
    RandomResizedCrop, Normalize, RandomApply,ColorJitter,RandomGrayscale
import core
from core.defenses.DBD import GaussianBlur



# ========== Set global settings ==========
global_seed = 666
deterministic = True
torch.manual_seed(global_seed)
CUDA_VISIBLE_DEVICES = '1'
datasets_root_dir = '../dataset/'

def gen_grid(height, k):
    """Generate an identity grid with shape 1*height*height*2 and a noise grid with shape 1*height*height*2
    according to the input height ``height`` and the uniform grid size ``k``.
    """
    ins = torch.rand(1, 2, k, k) * 2 - 1
    ins = ins / torch.mean(torch.abs(ins))  # a uniform grid
    noise_grid = nn.functional.upsample(ins, size=height, mode="bicubic", align_corners=True)
    noise_grid = noise_grid.permute(0, 2, 3, 1)  # 1*height*height*2
    array1d = torch.linspace(-1, 1, steps=height)  # 1D coordinate divided by height in [-1, 1]
    x, y = torch.meshgrid(array1d, array1d)  # 2D coordinates height*height
    identity_grid = torch.stack((y, x), 2)[None, ...]  # 1*height*height*2

    return identity_grid, noise_grid


# ========== ResNet-18 CIFAR-10 Defense against Badnet with DBD ==========
def DBD_against_BadNets_CIFAR10():
    print('=> Testing DBD against BadNets on CIFAR-10')
    model_name = 'ResNet-18'
    dataset_name = 'CIFAR10'
    attack_name = 'BadNets'
    defense_name = 'DBD'
    print('=> Loading CIFAR10...')
    dataset = torchvision.datasets.CIFAR10

    transform_train_pre = Compose([])
    transform_train_primary = Compose(
        [RandomResizedCrop(size=(32, 32), scale=(0.2, 1.0), ratio=(0.75, 1.3333), interpolation=3),
         RandomHorizontalFlip(p=0.5)])
    transform_train_remaining = Compose([
        ToTensor(),
        Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.201])])

    transform_train = Compose([transform_train_primary, transform_train_remaining])

    transform_test = Compose([
        ToTensor(),
        Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.201])
    ])
    clean_train_data = dataset(datasets_root_dir, train=True, transform=transform_train_pre, download=True)
    clean_test_data = dataset(datasets_root_dir, train=False, transform=transform_test, download=True)

    pattern = torch.zeros((32, 32), dtype=torch.uint8)
    pattern[-3:, -3:] = 255
    weight = torch.zeros((32, 32), dtype=torch.float32)
    weight[-3:, -3:] = 1.0

    badnets = core.BadNets(
        train_dataset=clean_train_data,
        test_dataset=clean_test_data,
        model=core.models.ResNet(18),
        loss=nn.CrossEntropyLoss(),
        y_target=1,
        poisoned_rate=0.05,
        pattern=pattern,
        weight=weight,
        seed=global_seed,
        deterministic=deterministic
    )
    print('=> Creating poisoned dataset...')
    poisoned_train_data, poisoned_test_data = badnets.get_poisoned_dataset()
    poison_idx_path = './backdoorbox_poison_idx.npy'
    poison_idx_np = np.zeros(len(poisoned_train_data))
    for i in range(len(poisoned_train_data)):
        if i in poisoned_train_data.poisoned_set:
            poison_idx_np[i] = 1
    np.save(poison_idx_path, poison_idx_np)


    print("=> Training ResNet18 with DBD...")
    transform_aug_pre = Compose([])
    transform_aug_primary = Compose([
        RandomResizedCrop(size=(32, 32), scale=(0.2, 1.0), ratio=(0.75, 1.3333), interpolation=3),
        RandomHorizontalFlip(p=0.5),
        RandomApply(
            [ColorJitter(brightness=[0.6, 1.4], contrast=[0.6, 1.4], saturation=[0.6, 1.4], hue=[-0.1, 0.1])],
            p=0.8),
        RandomGrayscale(p=0.2),
        RandomApply([GaussianBlur(sigma=[0.1, 2.0])], p=0.5)
    ])

    transform_aug_remaining = Compose([
        ToTensor(),
        Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.201])
    ])

    aug_transform = {
        "pre": transform_aug_pre,
        "primary": transform_aug_primary,
        "remaining": transform_aug_remaining,
    }
    defense = core.DBD(model_name='resnet18',
                       num_classes=10,
                       head='mlp',
                       poisoned_train_dataset=poisoned_train_data,
                       poisoned_test_dataset=poisoned_test_data,
                       benign_test_dataset=clean_test_data,
                       seed=global_seed,
                       deterministic=deterministic)

    benign_test_schedule = {
        'device': 'GPU',
        'CUDA_VISIBLE_DEVICES': CUDA_VISIBLE_DEVICES,
        'GPU_num': 1,

        # 'test_model': model_path,
        'batch_size': 128,
        'num_workers': 4,

        # 1. ASR: the attack success rate calculated on all poisoned samples
        # 2. ASR_NoTarget: the attack success rate calculated on all poisoned samples whose ground-truth labels are not the target label
        # 3. BA: the accuracy on all benign samples
        # Hint: For ASR and BA, the computation of the metric is decided by the dataset but not schedule['metric'].
        # In other words, ASR or BA does not influence the computation of the metric.
        # For ASR_NoTarget, the code will delete all the samples whose ground-truth labels are the target label and then compute the metric.
        'metric': 'BA',
        'y_target': 1,

        'save_dir': 'experiments',
        'experiment_name': f'{model_name}_{dataset_name}_{attack_name}_{defense_name}_BA'
    }

    poisoned_test_schedule = {
        'device': 'GPU',
        'CUDA_VISIBLE_DEVICES': CUDA_VISIBLE_DEVICES,
        'GPU_num': 1,

        # 'test_model': model_path,
        'batch_size': 128,
        'num_workers': 4,

        # 1. ASR: the attack success rate calculated on all poisoned samples
        # 2. ASR_NoTarget: the attack success rate calculated on all poisoned samples whose ground-truth labels are not the target label
        # 3. BA: the accuracy on all benign samples
        # Hint: For ASR and BA, the computation of the metric is decided by the dataset but not schedule['metric'].
        # In other words, ASR or BA does not influence the computation of the metric.
        # For ASR_NoTarget, the code will delete all the samples whose ground-truth labels are the target label and then compute the metric.
        'metric': 'ASR_NoTarget',
        'y_target': 1,

        'save_dir': 'experiments',
        'experiment_name': f'{model_name}_{dataset_name}_{attack_name}_{defense_name}_ASR_NoTarget'
    }



    print("=> DBD: Self-supervised (simclr) with unlabeled data...")
    schedule = {
        'device': 'GPU',
        'CUDA_VISIBLE_DEVICES': CUDA_VISIBLE_DEVICES,
        'GPU_num': 1,

        'batch_size': 512,
        'num_workers': 4,
        'pin_memory': True,

        # transform:
        'transform_aug': aug_transform,

        'sync_bn': True,  # synchronized batch normalization

        # criterion:
        'temperature': 0.5,

        # optimizer:
        'weight_decay': 1.e-4,
        'momentum': 0.9,
        'lr': 0.4,

        # lr_scheduler
        'scheduler': 'cosine_annealing',
        'T_max': 1000,  # same as epochs 1000

        'epochs': 1000,  # 1000
        'early_stop_epoch': 1000,

        # resume
        # 'resume': '',

        'num_stage_epochs': 100,
        'min_interval': 20,
        'max_interval': 100,

        'experiment_name': f'{model_name}_{dataset_name}_{attack_name}_{defense_name}_simclr',

        'log_iteration_interval': 30,
        'test_epoch_interval': 10,
        'save_epoch_interval': 100,

        'save_dir': 'experiments',

        # test schedule:
        'benign_test_schedule': benign_test_schedule,
        'poisoned_test_schedule': poisoned_test_schedule,
    }
    defense.simclr(schedule)

    print("=> DBD: Finetune and Mixmatch...")
    schedule = {
        'device': 'GPU',
        'CUDA_VISIBLE_DEVICES': CUDA_VISIBLE_DEVICES,
        'GPU_num': 1,

        # pretrain simclr model
        # 'pretrain_simclr_backbone_checkpoint': './experiments/ResNet-18_CIFAR-10_DBD_against_BadNet_simclr_2022-05-31_21:13:36/ckpt_epoch_100.pth',
        # 'pretrain_simclr_backbone_checkpoint': '/data/zhulinghui/tmp2/DBD-main/DBD_change_trigger_simclr_backdoor_ckpt.pt',

        # 'resume':'./experiments/latest_model.pt_model_state_dict.pt',

        # train transform
        'train_transform': transform_train,

        # warm loader
        'warmup_batch_size': 128,
        'warmup_num_workers': 4,
        'warmup_pin_memory': True,

        # warmup-criterion:
        'warmup_alpha': 0.1,
        'warmup_beta': 1,
        'warmup_epochs': 10,

        'semi_epsilon': 0.5,

        # semi-criterion:
        'semi_lambda_u': 15,
        'semi_rampup_length': 190,

        # semi-loader:
        'semi_batch_size': 64,
        'semi_num_workers': 4,
        'semi_pin_memory': True,

        # optimizer-Adam
        'lr': 0.002,

        # scheduler
        # 'scheduler':'',

        # mixmatch:
        'mixmatch_train_iteration': 1024,
        'mixmatch_temperature': 0.5,
        'mixmatch_alpha': 0.75,
        'semi_epochs': 190,

        'log_iteration_interval': 30,
        'test_epoch_interval': 10,
        'save_epoch_interval': 10,

        'save_dir': 'experiments',
        'experiment_name': f'{model_name}_{dataset_name}_{attack_name}_{defense_name}'+'_mixmatch_finetune',

        # test schedule:
        'benign_test_schedule': benign_test_schedule,
        'poisoned_test_schedule': poisoned_test_schedule,
    }
    defense.mixmatch_finetune(schedule)
    print("=> DBD Done")
    return


# ========== ResNet-18 GTSRB Defense against Badnet with DBD ==========
def DBD_against_BadNets_GTSRB():
    print('=> Testing DBD against BadNets on GTSRB')
    model_name = 'ResNet-18'
    dataset_name = 'GTSRB'
    attack_name = 'BadNets'
    defense_name = 'DBD'

    print('=> Loading GTSRB...')
    transform_train_pre = Compose([
        ToPILImage(),
        Resize((32, 32))
    ])


    transform_train_primary = Compose([
        RandomResizedCrop(size=(32, 32), scale=(0.2, 1.0), ratio=(0.75, 1.3333), interpolation=3),
        RandomHorizontalFlip(p=0.5)
    ])

    transform_train_remaining = Compose([
        ToTensor()
    ])

    transform_train = Compose([transform_train_primary, transform_train_remaining])

    transform_test = Compose([
        ToPILImage(),
        Resize((32, 32)),
        ToTensor()
    ])

    clean_train_data = DatasetFolder(
        root=osp.join(datasets_root_dir, 'GTSRB', 'train'),  # please replace this with path to your training set
        loader=cv2.imread,
        extensions=('png',),
        transform=transform_train_pre,
        target_transform=None,
        is_valid_file=None)

    clean_test_data = DatasetFolder(
        root=osp.join(datasets_root_dir, 'GTSRB', 'testset'),  # please replace this with path to your test set
        loader=cv2.imread,
        extensions=('png',),
        transform=transform_test,
        target_transform=None,
        is_valid_file=None)

    pattern = torch.zeros((32, 32), dtype=torch.uint8)
    pattern[-3:, -3:] = 255
    weight = torch.zeros((32, 32), dtype=torch.float32)
    weight[-3:, -3:] = 1.0

    badnets = core.BadNets(
        train_dataset=clean_train_data,
        test_dataset=clean_test_data,
        model=core.models.ResNet(18, 43),
        loss=nn.CrossEntropyLoss(),
        y_target=1,
        poisoned_rate=0.05,
        pattern=pattern,
        weight=weight,
        poisoned_transform_train_index=2,
        poisoned_transform_test_index=2,
        seed=global_seed,
        deterministic=deterministic
    )
    print('=> Creating poisoned dataset...')
    poisoned_train_data, poisoned_test_data = badnets.get_poisoned_dataset()  # not a dataloader, is dataset

    print("=> Training ResNet18 with DBD...")
    transform_aug_pre = Compose([])

    transform_aug_primary = Compose([
        RandomResizedCrop(size=(32, 32), scale=(0.2, 1.0), ratio=(0.75, 1.3333), interpolation=3),
        RandomHorizontalFlip(p=0.5),
        RandomApply(
            [ColorJitter(brightness=[0.6, 1.4], contrast=[0.6, 1.4], saturation=[0.6, 1.4], hue=[-0.1, 0.1])],
            p=0.8),
        RandomGrayscale(p=0.2),
        RandomApply([GaussianBlur(sigma=[0.1, 2.0])], p=0.5)
    ])

    transform_aug_remaining = Compose([
        ToTensor(),
    ])

    aug_transform = {
        "pre": transform_aug_pre,
        "primary": transform_aug_primary,
        "remaining": transform_aug_remaining,
    }

    defense = core.DBD(model_name='resnet18',
                       num_classes=len(clean_train_data.classes),
                       head='mlp',
                       poisoned_train_dataset=poisoned_train_data,
                       poisoned_test_dataset=poisoned_test_data,
                       benign_test_dataset=clean_test_data,
                       seed=global_seed,
                       deterministic=deterministic)

    benign_test_schedule = {
        'device': 'GPU',
        'CUDA_VISIBLE_DEVICES': CUDA_VISIBLE_DEVICES,
        'GPU_num': 1,

        # 'test_model': model_path,
        'batch_size': 128,
        'num_workers': 4,

        # 1. ASR: the attack success rate calculated on all poisoned samples
        # 2. ASR_NoTarget: the attack success rate calculated on all poisoned samples whose ground-truth labels are not the target label
        # 3. BA: the accuracy on all benign samples
        # Hint: For ASR and BA, the computation of the metric is decided by the dataset but not schedule['metric'].
        # In other words, ASR or BA does not influence the computation of the metric.
        # For ASR_NoTarget, the code will delete all the samples whose ground-truth labels are the target label and then compute the metric.
        'metric': 'BA',
        'y_target': 1,

        'save_dir': 'experiments',
        'experiment_name': f'{model_name}_{dataset_name}_{attack_name}_{defense_name}_BA'
    }

    poisoned_test_schedule = {
        'device': 'GPU',
        'CUDA_VISIBLE_DEVICES': CUDA_VISIBLE_DEVICES,
        'GPU_num': 1,

        # 'test_model': model_path,
        'batch_size': 128,
        'num_workers': 4,

        # 1. ASR: the attack success rate calculated on all poisoned samples
        # 2. ASR_NoTarget: the attack success rate calculated on all poisoned samples whose ground-truth labels are not the target label
        # 3. BA: the accuracy on all benign samples
        # Hint: For ASR and BA, the computation of the metric is decided by the dataset but not schedule['metric'].
        # In other words, ASR or BA does not influence the computation of the metric.
        # For ASR_NoTarget, the code will delete all the samples whose ground-truth labels are the target label and then compute the metric.
        'metric': 'ASR_NoTarget',
        'y_target': 1,

        'save_dir': 'experiments',
        'experiment_name': f'{model_name}_{dataset_name}_{attack_name}_{defense_name}_ASR_NoTarget'
    }


    print("=> DBD: Self-supervised (simclr) with unlabeled data...")
    schedule = {
        'device': 'GPU',
        'CUDA_VISIBLE_DEVICES': CUDA_VISIBLE_DEVICES,
        'GPU_num': 1,

        'batch_size': 512,
        'num_workers': 4,
        'pin_memory': True,

        # transform:
        'transform_aug': aug_transform,

        'sync_bn': True,  # synchronized batch normalization

        # criterion:
        'temperature': 0.5,

        # optimizer:
        'weight_decay': 1.e-4,
        'momentum': 0.9,
        'lr': 0.4,

        # lr_scheduler
        'scheduler': 'cosine_annealing',
        'T_max': 1000,  # same as epochs 1000

        'epochs': 1000,  # 1000
        'early_stop_epoch': 1000,

        # resume
        # 'resume': '',

        'num_stage_epochs': 100,
        'min_interval': 20,
        'max_interval': 100,

        'experiment_name': f'{model_name}_{dataset_name}_{attack_name}_{defense_name}_simclr',

        'log_iteration_interval': 30,
        'test_epoch_interval': 10,
        'save_epoch_interval': 100,

        'save_dir': 'experiments',

        # test schedule:
        'benign_test_schedule': benign_test_schedule,
        'poisoned_test_schedule': poisoned_test_schedule,

    }

    defense.simclr(schedule)

    print("=> DBD: Finetune and Mixmatch...")
    schedule = {
        'device': 'GPU',
        'CUDA_VISIBLE_DEVICES': CUDA_VISIBLE_DEVICES,
        'GPU_num': 1,

        # pretrain simclr model
        # 'pretrain_simclr_backbone_checkpoint': './'

        # 'resume':'./',

        # train transform
        'train_transform': transform_train,

        # warm loader
        'warmup_batch_size': 128,
        'warmup_num_workers': 4,
        'warmup_pin_memory': True,

        # warmup-criterion:
        'warmup_alpha': 0.1,
        'warmup_beta': 1,
        'warmup_epochs': 10,

        'semi_epsilon': 0.5,

        # semi-criterion:
        'semi_lambda_u': 15,
        'semi_rampup_length': 190,

        # semi-loader:
        'semi_batch_size': 64,
        'semi_num_workers': 4,
        'semi_pin_memory': True,

        # optimizer-Adam
        'lr': 0.002,

        # scheduler
        # 'scheduler':'',

        # mixmatch:
        'mixmatch_train_iteration': 1024,
        'mixmatch_temperature': 0.5,
        'mixmatch_alpha': 0.75,
        'semi_epochs': 190,

        'log_iteration_interval': 30,
        'test_epoch_interval': 10,
        'save_epoch_interval': 10,

        'save_dir': 'experiments',
        'experiment_name': f'{model_name}_{dataset_name}_{attack_name}_{defense_name}'+'_mixmatch_finetune',

        # test schedule:
        'benign_test_schedule': benign_test_schedule,
        'poisoned_test_schedule': poisoned_test_schedule,
    }
    defense.mixmatch_finetune(schedule)
    print("=> DBD Done")

    return


# ========== ResNet-18 CIFAR-10 Defense against labelconsistant with DBD ==========
def DBD_against_LabelConsistent_CIFAR10():
    print('=> Testing DBD against LabelConsistent on CIFAR-10')
    model_name = 'ResNet-18'
    dataset_name = 'CIFAR10'
    attack_name = 'LabelConsistent'
    defense_name = 'DBD'
    print('=> Loading CIFAR10...')
    dataset = torchvision.datasets.CIFAR10

    transform_train_pre = Compose([])

    transform_train_primary = Compose(
        [RandomResizedCrop(size=(32, 32), scale=(0.2, 1.0), ratio=(0.75, 1.3333), interpolation=3),
         RandomHorizontalFlip(p=0.5)])
    transform_train_ramaining = Compose([
        ToTensor()
        ])

    transform_train = Compose([transform_train_primary, transform_train_ramaining])

    clean_train_data = dataset(datasets_root_dir, train=True, transform=transform_train_pre, download=False)

    transform_test = Compose([
        ToTensor()
    ])
    clean_test_data = dataset(datasets_root_dir, train=False, transform=transform_test, download=False)

    adv_model = core.models.ResNet(18)
    adv_ckpt = torch.load(
        './experiments/train_benign_CIFAR10_BadNets_2022-03-25_15:31:26/ckpt_epoch_90.pth')
    adv_model.load_state_dict(adv_ckpt)

    pattern = torch.zeros((32, 32), dtype=torch.uint8)
    pattern[-1, -1] = 255
    pattern[-1, -3] = 255
    pattern[-3, -1] = 255
    pattern[-2, -2] = 255

    pattern[0, -1] = 255
    pattern[1, -2] = 255
    pattern[2, -3] = 255
    pattern[2, -1] = 255

    pattern[0, 0] = 255
    pattern[1, 1] = 255
    pattern[2, 2] = 255
    pattern[2, 0] = 255

    pattern[-1, 0] = 255
    pattern[-1, 2] = 255
    pattern[-2, 1] = 255
    pattern[-3, 0] = 255

    weight = torch.zeros((32, 32), dtype=torch.float32)
    weight[:3, :3] = 1.0
    weight[:3, -3:] = 1.0
    weight[-3:, :3] = 1.0
    weight[-3:, -3:] = 1.0

    schedule = {
        'device': 'GPU',
        'CUDA_VISIBLE_DEVICES': CUDA_VISIBLE_DEVICES,
        'GPU_num': 1,

        'benign_training': False,  # Train Attacked Model
        'batch_size': 128,
        'num_workers': 8,

        'lr': 0.1,
        'momentum': 0.9,
        'weight_decay': 5e-4,
        'gamma': 0.1,
        'schedule': [150, 180],

        'epochs': 200,

        'log_iteration_interval': 100,
        'test_epoch_interval': 10,
        'save_epoch_interval': 10,

        'save_dir': 'experiments',
        'experiment_name': 'ResNet-18_CIFAR-10_LabelConsistent'
    }

    eps = 8
    alpha = 1.5
    steps = 100
    max_pixel = 255
    poisoned_rate = 0.25

    label_consistent = core.LabelConsistent(
        train_dataset=clean_train_data,
        test_dataset=clean_test_data,
        model=core.models.ResNet(18),
        adv_model=adv_model,
        adv_dataset_dir=f'./adv_dataset/CIFAR-10_eps{eps}_alpha{alpha}_steps{steps}_poisoned_rate{poisoned_rate}_seed{global_seed}',
        loss=nn.CrossEntropyLoss(),
        y_target=2,
        poisoned_rate=poisoned_rate,
        pattern=pattern,
        weight=weight,
        eps=eps,
        alpha=alpha,
        steps=steps,
        max_pixel=max_pixel,
        poisoned_transform_train_index=2,
        poisoned_transform_test_index=2,
        poisoned_target_transform_index=0,
        schedule=schedule,
        seed=global_seed,
        deterministic=True
    )
    print('=> Creating poisoned dataset...')
    poisoned_train_data, poisoned_test_data = label_consistent.get_poisoned_dataset()


    print("=> Training ResNet18 with DBD...")
    transform_aug_pre = Compose([])

    transform_aug_primary = Compose([
        RandomResizedCrop(size=(32, 32), scale=(0.2, 1.0), ratio=(0.75, 1.3333), interpolation=3),
        RandomHorizontalFlip(p=0.5),
        RandomApply(
            [ColorJitter(brightness=[0.6, 1.4], contrast=[0.6, 1.4], saturation=[0.6, 1.4], hue=[-0.1, 0.1])],
            p=0.8),
        RandomGrayscale(p=0.2),
        RandomApply([GaussianBlur(sigma=[0.1, 2.0])], p=0.5)
    ])

    transform_aug_remaining = Compose([
        ToTensor(),
    ])

    aug_transform = {
        "pre": transform_aug_pre,
        "primary": transform_aug_primary,
        "remaining": transform_aug_remaining,
    }


    defense = core.DBD(model_name='resnet18',
                       num_classes=10,
                       head='mlp',
                       poisoned_train_dataset=poisoned_train_data,
                       poisoned_test_dataset=poisoned_test_data,
                       benign_test_dataset=clean_test_data,
                       seed=global_seed,
                       deterministic=deterministic)

    benign_test_schedule = {
        'device': 'GPU',
        'CUDA_VISIBLE_DEVICES': CUDA_VISIBLE_DEVICES,
        'GPU_num': 1,

        # 'test_model': model_path,
        'batch_size': 128,
        'num_workers': 4,

        # 1. ASR: the attack success rate calculated on all poisoned samples
        # 2. ASR_NoTarget: the attack success rate calculated on all poisoned samples whose ground-truth labels are not the target label
        # 3. BA: the accuracy on all benign samples
        # Hint: For ASR and BA, the computation of the metric is decided by the dataset but not schedule['metric'].
        # In other words, ASR or BA does not influence the computation of the metric.
        # For ASR_NoTarget, the code will delete all the samples whose ground-truth labels are the target label and then compute the metric.
        'metric': 'BA',
        'y_target': 2,

        'save_dir': 'experiments',
        'experiment_name': f'{model_name}_{dataset_name}_{attack_name}_{defense_name}_BA'
    }

    poisoned_test_schedule = {
        'device': 'GPU',
        'CUDA_VISIBLE_DEVICES': CUDA_VISIBLE_DEVICES,
        'GPU_num': 1,

        # 'test_model': model_path,
        'batch_size': 128,
        'num_workers': 4,

        # 1. ASR: the attack success rate calculated on all poisoned samples
        # 2. ASR_NoTarget: the attack success rate calculated on all poisoned samples whose ground-truth labels are not the target label
        # 3. BA: the accuracy on all benign samples
        # Hint: For ASR and BA, the computation of the metric is decided by the dataset but not schedule['metric'].
        # In other words, ASR or BA does not influence the computation of the metric.
        # For ASR_NoTarget, the code will delete all the samples whose ground-truth labels are the target label and then compute the metric.
        'metric': 'ASR_NoTarget',
        'y_target': 2,

        'save_dir': 'experiments',
        'experiment_name': f'{model_name}_{dataset_name}_{attack_name}_{defense_name}_ASR_NoTarget'
    }


    print("=> DBD: Self-supervised (simclr) with unlabeled data...")
    schedule = {
        'device': 'GPU',
        'CUDA_VISIBLE_DEVICES': CUDA_VISIBLE_DEVICES,
        'GPU_num': 1,

        'batch_size': 512,
        'num_workers': 2,
        'pin_memory': True,

        # transform:
        'transform_aug': aug_transform,

        'sync_bn': True,  # synchronized batch normalization

        # criterion:
        'temperature': 0.5,

        # optimizer:
        'weight_decay': 1.e-4,
        'momentum': 0.9,
        'lr': 0.4,

        # lr_scheduler
        'scheduler': 'cosine_annealing',
        'T_max': 1000,  # same as epochs 1000

        'epochs': 1000,  # 1000
        'early_stop_epoch': 1000,

        # resume
        # 'resume': '',

        'num_stage_epochs': 100,
        'min_interval': 20,
        'max_interval': 100,

        'experiment_name': f'{model_name}_{dataset_name}_{attack_name}_{defense_name}_simclr',

        'log_iteration_interval': 30,
        'test_epoch_interval': 10,
        'save_epoch_interval': 100,

        'save_dir': 'experiments',

        # test schedule:
        'benign_test_schedule': benign_test_schedule,
        'poisoned_test_schedule': poisoned_test_schedule,

    }

    defense.simclr(schedule)

    print("=> DBD: Finetune and Mixmatch...")
    schedule = {
        'device': 'GPU',
        'CUDA_VISIBLE_DEVICES': CUDA_VISIBLE_DEVICES,
        'GPU_num': 1,

        # pretrain simclr model
        # 'pretrain_simclr_backbone_checkpoint': './experiments/ResNet-18_CIFAR-10_DBD_against_BadNet_simclr_2022-05-31_21:13:36/ckpt_epoch_100.pth',
        # 'pretrain_simclr_backbone_checkpoint': './experiments/epoch100.pt_backbone_state_dict.pt',

        # 'resume':'./experiments/latest_model.pt_model_state_dict.pt',

        # train transform
        'train_transform': transform_train,

        # warm loader
        'warmup_batch_size': 128,
        'warmup_num_workers': 4,
        'warmup_pin_memory': True,

        # warmup-criterion:
        'warmup_alpha': 0.1,
        'warmup_beta': 1,
        'warmup_epochs': 10,

        'semi_epsilon': 0.5,

        # semi-criterion:
        'semi_lambda_u': 15,
        'semi_rampup_length': 190,

        # semi-loader:
        'semi_batch_size': 64,
        'semi_num_workers': 4,
        'semi_pin_memory': True,

        # optimizer-Adam
        'lr': 0.002,

        # scheduler
        # 'scheduler':'',

        # mixmatch:
        'mixmatch_train_iteration': 1024,
        'mixmatch_temperature': 0.5,
        'mixmatch_alpha': 0.75,
        'semi_epochs': 190,

        'log_iteration_interval': 30,
        'test_epoch_interval': 10,
        'save_epoch_interval': 10,

        'save_dir': 'experiments',
        'experiment_name': f'{model_name}_{dataset_name}_{attack_name}_{defense_name}' + '_mixmatch_finetune',

        # test schedule:
        'benign_test_schedule': benign_test_schedule,
        'poisoned_test_schedule': poisoned_test_schedule,

    }
    defense.mixmatch_finetune(schedule)
    print("=> DBD Done")
    return




# ========== ResNet-18 GTSRB Defense against labelconsistant with DBD ==========
def DBD_against_LabelConsistent_GTSRB():
    print('=> Testing DBD against LabelConsistent on GTSRB')
    model_name = 'ResNet-18'
    dataset_name = 'GTSRB'
    attack_name = 'LabelConsistent'
    defense_name = 'DBD'
    print('=> Loading GTSRB...')

    transform_train_pre = Compose([
        ToPILImage(),
        Resize((32, 32))
    ])

    transform_train_primary = Compose([
        RandomResizedCrop(size=(32, 32), scale=(0.2, 1.0), ratio=(0.75, 1.3333), interpolation=3),
        RandomHorizontalFlip(p=0.5)
    ])

    transform_train_remaining = Compose([
        ToTensor()
    ])
    transform_train = Compose([transform_train_primary, transform_train_remaining])

    clean_train_data = DatasetFolder(
        root=osp.join(datasets_root_dir, 'GTSRB', 'train'),  # please replace this with path to your training set
        loader=cv2.imread,
        extensions=('png',),
        transform=transform_train_pre,
        target_transform=None,
        is_valid_file=None)

    transform_test = Compose([
        ToPILImage(),
        Resize((32, 32)),
        ToTensor()])

    clean_test_data = DatasetFolder(
        root=osp.join(datasets_root_dir, 'GTSRB', 'testset'),  # please replace this with path to your test set
        loader=cv2.imread,
        extensions=('png',),
        transform=transform_test,
        target_transform=None,
        is_valid_file=None)

    adv_model = core.models.ResNet(18, 43)
    adv_ckpt = torch.load(
        './experiments/ResNet-18_GTSRB_Benign_2023-01-29_19:17:30//ckpt_epoch_30.pth')
    adv_model.load_state_dict(adv_ckpt)

    pattern = torch.zeros((32, 32), dtype=torch.uint8)
    pattern[-1, -1] = 255
    pattern[-1, -3] = 255
    pattern[-3, -1] = 255
    pattern[-2, -2] = 255

    pattern[0, -1] = 255
    pattern[1, -2] = 255
    pattern[2, -3] = 255
    pattern[2, -1] = 255

    pattern[0, 0] = 255
    pattern[1, 1] = 255
    pattern[2, 2] = 255
    pattern[2, 0] = 255

    pattern[-1, 0] = 255
    pattern[-1, 2] = 255
    pattern[-2, 1] = 255
    pattern[-3, 0] = 255

    weight = torch.zeros((32, 32), dtype=torch.float32)
    weight[:3, :3] = 1.0
    weight[:3, -3:] = 1.0
    weight[-3:, :3] = 1.0
    weight[-3:, -3:] = 1.0

    schedule = {
        'device': 'GPU',
        'CUDA_VISIBLE_DEVICES': CUDA_VISIBLE_DEVICES,
        'GPU_num': 1,

        'benign_training': False,  # Train Attacked Model
        'batch_size': 256,
        'num_workers': 8,

        'lr': 0.01,
        'momentum': 0.9,
        'weight_decay': 5e-4,
        'gamma': 0.1,
        'schedule': [20],

        'epochs': 50,

        'log_iteration_interval': 100,
        'test_epoch_interval': 10,
        'save_epoch_interval': 10,

        'save_dir': 'experiments',
        'experiment_name': 'ResNet-18_GTSRB_LabelConsistent'
    }

    eps = 16
    alpha = 1.5
    steps = 100
    max_pixel = 255
    poisoned_rate = 0.5

    label_consistent = core.LabelConsistent(
        train_dataset=clean_train_data,
        test_dataset=clean_test_data,
        model=core.models.ResNet(18, 43),
        adv_model=adv_model,
        adv_dataset_dir=f'./adv_dataset/GTSRB_eps{eps}_alpha{alpha}_steps{steps}_poisoned_rate{poisoned_rate}_seed{global_seed}',
        loss=nn.CrossEntropyLoss(),
        y_target=2,
        poisoned_rate=poisoned_rate,
        adv_transform=Compose([ToPILImage(), Resize((32, 32)), ToTensor()]),
        pattern=pattern,
        weight=weight,
        eps=eps,
        alpha=alpha,
        steps=steps,
        max_pixel=max_pixel,
        poisoned_transform_train_index=2,
        poisoned_transform_test_index=2,
        poisoned_target_transform_index=0,
        schedule=schedule,
        seed=global_seed,
        deterministic=True
    )
    print('=> Creating poisoned dataset...')
    poisoned_train_data, poisoned_test_data = label_consistent.get_poisoned_dataset()


    print("=> Training ResNet18 with DBD...")
    transform_aug_pre = Compose([])

    transform_aug_primary = Compose([
        RandomResizedCrop(size=(32, 32), scale=(0.2, 1.0), ratio=(0.75, 1.3333), interpolation=3),
        RandomHorizontalFlip(p=0.5),
        RandomApply(
            [ColorJitter(brightness=[0.6, 1.4], contrast=[0.6, 1.4], saturation=[0.6, 1.4], hue=[-0.1, 0.1])],
            p=0.8),
        RandomGrayscale(p=0.2),
        RandomApply([GaussianBlur(sigma=[0.1, 2.0])], p=0.5)
    ])

    transform_aug_remaining = Compose([
        ToTensor()
    ])

    aug_transform = {
        "pre": transform_aug_pre,
        "primary": transform_aug_primary,
        "remaining": transform_aug_remaining,
    }

    defense = core.DBD(model_name='resnet18',
                       num_classes=len(clean_train_data.classes),
                       head='mlp',
                       poisoned_train_dataset=poisoned_train_data,
                       poisoned_test_dataset=poisoned_test_data,
                       benign_test_dataset=clean_test_data,
                       seed=global_seed,
                       deterministic=deterministic)

    benign_test_schedule = {
        'device': 'GPU',
        'CUDA_VISIBLE_DEVICES': CUDA_VISIBLE_DEVICES,
        'GPU_num': 1,

        # 'test_model': model_path,
        'batch_size': 128,
        'num_workers': 4,

        # 1. ASR: the attack success rate calculated on all poisoned samples
        # 2. ASR_NoTarget: the attack success rate calculated on all poisoned samples whose ground-truth labels are not the target label
        # 3. BA: the accuracy on all benign samples
        # Hint: For ASR and BA, the computation of the metric is decided by the dataset but not schedule['metric'].
        # In other words, ASR or BA does not influence the computation of the metric.
        # For ASR_NoTarget, the code will delete all the samples whose ground-truth labels are the target label and then compute the metric.
        'metric': 'BA',
        'y_target': 2,

        'save_dir': 'experiments',
        'experiment_name': f'{model_name}_{dataset_name}_{attack_name}_{defense_name}_BA'
    }

    poisoned_test_schedule = {
        'device': 'GPU',
        'CUDA_VISIBLE_DEVICES': CUDA_VISIBLE_DEVICES,
        'GPU_num': 1,

        # 'test_model': model_path,
        'batch_size': 128,
        'num_workers': 4,

        # 1. ASR: the attack success rate calculated on all poisoned samples
        # 2. ASR_NoTarget: the attack success rate calculated on all poisoned samples whose ground-truth labels are not the target label
        # 3. BA: the accuracy on all benign samples
        # Hint: For ASR and BA, the computation of the metric is decided by the dataset but not schedule['metric'].
        # In other words, ASR or BA does not influence the computation of the metric.
        # For ASR_NoTarget, the code will delete all the samples whose ground-truth labels are the target label and then compute the metric.
        'metric': 'ASR_NoTarget',
        'y_target': 2,

        'save_dir': 'experiments',
        'experiment_name': f'{model_name}_{dataset_name}_{attack_name}_{defense_name}_ASR_NoTarget'
    }

    print("=> DBD: Self-supervised (simclr) with unlabeled data...")
    schedule = {
        'device': 'GPU',
        'CUDA_VISIBLE_DEVICES': CUDA_VISIBLE_DEVICES,
        'GPU_num': 1,

        'batch_size': 512,
        'num_workers': 4,
        'pin_memory': True,

        # transform:
        'transform_aug': aug_transform,

        'sync_bn': True,  # synchronized batch normalization

        # criterion:
        'temperature': 0.5,

        # optimizer:
        'weight_decay': 1.e-4,
        'momentum': 0.9,
        'lr': 0.4,

        # lr_scheduler
        'scheduler': 'cosine_annealing',
        'T_max': 1000,  # same as epochs 1000

        'epochs': 1000,  # 1000
        'early_stop_epoch': 1000,

        # resume
        # 'resume': '',

        'num_stage_epochs': 100,
        'min_interval': 20,
        'max_interval': 100,

        'experiment_name': f'{model_name}_{dataset_name}_{attack_name}_{defense_name}_simclr',

        'log_iteration_interval': 30,
        'test_epoch_interval': 10,
        'save_epoch_interval': 100,

        'save_dir': 'experiments',

        # test schedule:
        'benign_test_schedule': benign_test_schedule,
        'poisoned_test_schedule': poisoned_test_schedule,

    }

    defense.simclr(schedule)

    print("=> DBD: Finetune and Mixmatch...")
    schedule = {
        'device': 'GPU',
        'CUDA_VISIBLE_DEVICES': CUDA_VISIBLE_DEVICES,
        'GPU_num': 1,

        # pretrain simclr model
        # 'pretrain_simclr_backbone_checkpoint': './'

        # 'resume':'./',

        # train transform
        'train_transform': transform_train,

        # warm loader
        'warmup_batch_size': 128,
        'warmup_num_workers': 4,
        'warmup_pin_memory': True,

        # warmup-criterion:
        'warmup_alpha': 0.1,
        'warmup_beta': 1,
        'warmup_epochs': 10,

        'semi_epsilon': 0.5,

        # semi-criterion:
        'semi_lambda_u': 15,
        'semi_rampup_length': 190,

        # semi-loader:
        'semi_batch_size': 64,
        'semi_num_workers': 4,
        'semi_pin_memory': True,

        # optimizer-Adam
        'lr': 0.002,

        # scheduler
        # 'scheduler':'',

        # mixmatch:
        'mixmatch_train_iteration': 1024,
        'mixmatch_temperature': 0.5,
        'mixmatch_alpha': 0.75,
        'semi_epochs': 190,

        'log_iteration_interval': 30,
        'test_epoch_interval': 10,
        'save_epoch_interval': 10,

        'save_dir': 'experiments',
        'experiment_name': f'{model_name}_{dataset_name}_{attack_name}_{defense_name}' + '_mixmatch_finetune',

        # test schedule:
        'benign_test_schedule': benign_test_schedule,
        'poisoned_test_schedule': poisoned_test_schedule,
    }
    defense.mixmatch_finetune(schedule)
    print("=> DBD Done")
    return



# ========== ResNet-18 CIFAR-10 Defense against Wanet with DBD ==========
def DBD_against_WaNet_CIFAR10():
    print('=> Testing DBD against WaNet on CIFAR-10')
    model_name = 'ResNet-18'
    dataset_name = 'CIFAR10'
    attack_name = 'WaNet'
    defense_name = 'DBD'
    print('=> Loading CIFAR10...')
    dataset = torchvision.datasets.CIFAR10

    transform_train_pre = Compose([])
    transform_train_primary = Compose(
        [RandomResizedCrop(size=(32, 32), scale=(0.2, 1.0), ratio=(0.75, 1.3333), interpolation=3),
         RandomHorizontalFlip(p=0.5)])
    transform_train_ramaining = Compose([
        ToTensor(),
        ])

    transform_train = Compose([transform_train_primary, transform_train_ramaining])

    clean_train_data = dataset(datasets_root_dir, train=True, transform=transform_train_pre, download=False)

    transform_test = Compose([
        ToTensor()
    ])
    clean_test_data = dataset(datasets_root_dir, train=False, transform=transform_test, download=False)

    identity_grid, noise_grid = gen_grid(32, 4)
    torch.save(identity_grid, 'ResNet-18_CIFAR-10_WaNet_identity_grid.pth')
    torch.save(noise_grid, 'ResNet-18_CIFAR-10_WaNet_noise_grid.pth')
    wanet = core.WaNet(
        train_dataset=clean_train_data,
        test_dataset=clean_test_data,
        model=core.models.ResNet(18),
        loss=nn.CrossEntropyLoss(),
        y_target=0,
        poisoned_rate=0.1,
        identity_grid=identity_grid,
        noise_grid=noise_grid,
        noise=False,
        seed=global_seed,
        deterministic=deterministic
    )
    print('=> Creating poisoned dataset...')
    poisoned_train_data, poisoned_test_data = wanet.get_poisoned_dataset()

    print("=> Training ResNet18 with DBD...")
    transform_aug_pre = Compose([])
    transform_aug_primary = Compose([
        RandomResizedCrop(size=(32, 32), scale=(0.2, 1.0), ratio=(0.75, 1.3333), interpolation=3),
        RandomHorizontalFlip(p=0.5),
        RandomApply(
            [ColorJitter(brightness=[0.6, 1.4], contrast=[0.6, 1.4], saturation=[0.6, 1.4], hue=[-0.1, 0.1])],
            p=0.8),
        RandomGrayscale(p=0.2),
        RandomApply([GaussianBlur(sigma=[0.1, 2.0])], p=0.5)
    ])

    transform_aug_remaining = Compose([
        ToTensor(),
    ])

    aug_transform = {
        "pre": transform_aug_pre,
        "primary": transform_aug_primary,
        "remaining": transform_aug_remaining,
    }

    defense = core.DBD(model_name='resnet18',
                       num_classes=10,
                       head='mlp',
                       poisoned_train_dataset=poisoned_train_data,
                       poisoned_test_dataset=poisoned_test_data,
                       benign_test_dataset=clean_test_data,
                       seed=global_seed,
                       deterministic=deterministic)

    benign_test_schedule = {
        'device': 'GPU',
        'CUDA_VISIBLE_DEVICES': CUDA_VISIBLE_DEVICES,
        'GPU_num': 1,

        # 'test_model': model_path,
        'batch_size': 128,
        'num_workers': 4,

        # 1. ASR: the attack success rate calculated on all poisoned samples
        # 2. ASR_NoTarget: the attack success rate calculated on all poisoned samples whose ground-truth labels are not the target label
        # 3. BA: the accuracy on all benign samples
        # Hint: For ASR and BA, the computation of the metric is decided by the dataset but not schedule['metric'].
        # In other words, ASR or BA does not influence the computation of the metric.
        # For ASR_NoTarget, the code will delete all the samples whose ground-truth labels are the target label and then compute the metric.
        'metric': 'BA',
        'y_target': 0,

        'save_dir': 'experiments',
        'experiment_name': f'{model_name}_{dataset_name}_{attack_name}_{defense_name}_BA'
    }

    poisoned_test_schedule = {
        'device': 'GPU',
        'CUDA_VISIBLE_DEVICES': CUDA_VISIBLE_DEVICES,
        'GPU_num': 1,

        # 'test_model': model_path,
        'batch_size': 128,
        'num_workers': 4,

        # 1. ASR: the attack success rate calculated on all poisoned samples
        # 2. ASR_NoTarget: the attack success rate calculated on all poisoned samples whose ground-truth labels are not the target label
        # 3. BA: the accuracy on all benign samples
        # Hint: For ASR and BA, the computation of the metric is decided by the dataset but not schedule['metric'].
        # In other words, ASR or BA does not influence the computation of the metric.
        # For ASR_NoTarget, the code will delete all the samples whose ground-truth labels are the target label and then compute the metric.
        'metric': 'ASR_NoTarget',
        'y_target': 0,

        'save_dir': 'experiments',
        'experiment_name': f'{model_name}_{dataset_name}_{attack_name}_{defense_name}_ASR_NoTarget'
    }


    print("=> DBD: Self-supervised (simclr) with unlabeled data...")
    schedule = {
        'device': 'GPU',
        'CUDA_VISIBLE_DEVICES': CUDA_VISIBLE_DEVICES,
        'GPU_num': 1,

        'batch_size': 512,
        'num_workers': 4,
        'pin_memory': True,

        # transform:
        'transform_aug': aug_transform,

        'sync_bn': True,  # synchronized batch normalization

        # criterion:
        'temperature': 0.5,

        # optimizer:
        'weight_decay': 1.e-4,
        'momentum': 0.9,
        'lr': 0.4,

        # lr_scheduler
        'scheduler': 'cosine_annealing',
        'T_max': 1000,  # same as epochs 1000

        'epochs': 1000,  # 1000
        'early_stop_epoch': 1000,

        # resume
        # 'resume': '',

        'num_stage_epochs': 100,
        'min_interval': 20,
        'max_interval': 100,

        'experiment_name': f'{model_name}_{dataset_name}_{attack_name}_{defense_name}_simclr',

        'log_iteration_interval': 30,
        'test_epoch_interval': 10,
        'save_epoch_interval': 100,

        'save_dir': 'experiments',

        # test schedule:
        'benign_test_schedule': benign_test_schedule,
        'poisoned_test_schedule': poisoned_test_schedule,
    }

    defense.simclr(schedule)

    print("=> DBD: Finetune and Mixmatch...")
    schedule = {
        'device': 'GPU',
        'CUDA_VISIBLE_DEVICES': CUDA_VISIBLE_DEVICES,
        'GPU_num': 1,

        # pretrain simclr model
        # 'pretrain_simclr_backbone_checkpoint': './',

        # 'resume':'./',

        # train transform
        'train_transform': transform_train,

        # warm loader
        'warmup_batch_size': 128,
        'warmup_num_workers': 4,
        'warmup_pin_memory': True,

        # warmup-criterion:
        'warmup_alpha': 0.1,
        'warmup_beta': 1,
        'warmup_epochs': 10,

        'semi_epsilon': 0.5,

        # semi-criterion:
        'semi_lambda_u': 15,
        'semi_rampup_length': 190,

        # semi-loader:
        'semi_batch_size': 64,
        'semi_num_workers': 4,
        'semi_pin_memory': True,

        # optimizer-Adam
        'lr': 0.002,

        # scheduler
        # 'scheduler':'',

        # mixmatch:
        'mixmatch_train_iteration': 1024,
        'mixmatch_temperature': 0.5,
        'mixmatch_alpha': 0.75,
        'semi_epochs': 190,

        'log_iteration_interval': 100,
        'test_epoch_interval': 10,
        'save_epoch_interval': 10,

        'save_dir': 'experiments',
        'experiment_name': f'{model_name}_{dataset_name}_{attack_name}_{defense_name}'+'_mixmatch_fine',

        # test schedule:
        'benign_test_schedule': benign_test_schedule,
        'poisoned_test_schedule': poisoned_test_schedule,
    }
    defense.mixmatch_finetune(schedule)
    print("=> DBD Done")
    return



# ========== ResNet-18 GTSRB Defense against Wanet with DBD ==========
def DBD_against_WatNet_GTSRB():
    print('=> Testing DBD against WaNet on GTSRB')
    model_name = 'ResNet-18'
    dataset_name = 'GTSRB'
    attack_name = 'WaNet'
    defense_name = 'DBD'
    print('=> Loading GTSRB...')

    transform_train_pre = Compose([
        ToTensor(),
        RandomHorizontalFlip(),
        ToPILImage(),
        Resize((32, 32))
    ])

    transform_train_primary = Compose([
        RandomResizedCrop(size=(32, 32), scale=(0.2, 1.0), ratio=(0.75, 1.3333), interpolation=3),
        RandomHorizontalFlip(p=0.5)
    ])

    transform_train_remaining = Compose([
        ToTensor()
    ])

    transform_train = Compose([transform_train_primary, transform_train_remaining])

    transform_test = Compose([
        ToTensor(),
        ToPILImage(),
        Resize((32, 32)),
        ToTensor()
    ])

    clean_train_data = DatasetFolder(
        root=osp.join(datasets_root_dir, 'GTSRB', 'train'),  # please replace this with path to your training set
        loader=cv2.imread,
        extensions=('png',),
        transform=transform_train_pre,
        target_transform=None,
        is_valid_file=None)

    clean_test_data = DatasetFolder(
        root=osp.join(datasets_root_dir, 'GTSRB', 'testset'),  # please replace this with path to your test set
        loader=cv2.imread,
        extensions=('png',),
        transform=transform_test,
        target_transform=None,
        is_valid_file=None)

    identity_grid, noise_grid = gen_grid(32, 4)
    torch.save(identity_grid, 'ResNet-18_GTSRB_WaNet_identity_grid.pth')
    torch.save(noise_grid, 'ResNet-18_GTSRB_WaNet_noise_grid.pth')
    wanet = core.WaNet(
        train_dataset=clean_train_data,
        test_dataset=clean_test_data,
        model=core.models.ResNet(18, 43),
        loss=nn.CrossEntropyLoss(),
        y_target=0,
        poisoned_rate=0.1,
        identity_grid=identity_grid,
        noise_grid=noise_grid,
        noise=True,
        seed=global_seed,
        deterministic=deterministic
    )
    print('=> Creating poisoned dataset...')
    poisoned_train_data, poisoned_test_data = wanet.get_poisoned_dataset()

    print("=> Training ResNet18 with DBD...")
    transform_aug_pre = Compose([])

    transform_aug_primary = Compose([
        RandomResizedCrop(size=(32, 32), scale=(0.2, 1.0), ratio=(0.75, 1.3333), interpolation=3),
        RandomHorizontalFlip(p=0.5),
        RandomApply(
            [ColorJitter(brightness=[0.6, 1.4], contrast=[0.6, 1.4], saturation=[0.6, 1.4], hue=[-0.1, 0.1])],
            p=0.8),
        RandomGrayscale(p=0.2),
        RandomApply([GaussianBlur(sigma=[0.1, 2.0])], p=0.5)
    ])

    transform_aug_remaining = Compose([
        ToTensor()
    ])

    aug_transform = {
        "pre": transform_aug_pre,
        "primary": transform_aug_primary,
        "remaining": transform_aug_remaining,
    }

    defense = core.DBD(model_name='resnet18',
                       num_classes=len(clean_train_data.classes),
                       head='mlp',
                       poisoned_train_dataset=poisoned_train_data,
                       poisoned_test_dataset=poisoned_test_data,
                       benign_test_dataset=clean_test_data,
                       seed=global_seed,
                       deterministic=deterministic)
    benign_test_schedule = {
        'device': 'GPU',
        'CUDA_VISIBLE_DEVICES': CUDA_VISIBLE_DEVICES,
        'GPU_num': 1,

        # 'test_model': model_path,
        'batch_size': 128,
        'num_workers': 4,

        # 1. ASR: the attack success rate calculated on all poisoned samples
        # 2. ASR_NoTarget: the attack success rate calculated on all poisoned samples whose ground-truth labels are not the target label
        # 3. BA: the accuracy on all benign samples
        # Hint: For ASR and BA, the computation of the metric is decided by the dataset but not schedule['metric'].
        # In other words, ASR or BA does not influence the computation of the metric.
        # For ASR_NoTarget, the code will delete all the samples whose ground-truth labels are the target label and then compute the metric.
        'metric': 'BA',
        'y_target': 0,

        'save_dir': 'experiments',
        'experiment_name': f'{model_name}_{dataset_name}_{attack_name}_{defense_name}_BA'
    }

    poisoned_test_schedule = {
        'device': 'GPU',
        'CUDA_VISIBLE_DEVICES': CUDA_VISIBLE_DEVICES,
        'GPU_num': 1,

        # 'test_model': model_path,
        'batch_size': 128,
        'num_workers': 4,

        # 1. ASR: the attack success rate calculated on all poisoned samples
        # 2. ASR_NoTarget: the attack success rate calculated on all poisoned samples whose ground-truth labels are not the target label
        # 3. BA: the accuracy on all benign samples
        # Hint: For ASR and BA, the computation of the metric is decided by the dataset but not schedule['metric'].
        # In other words, ASR or BA does not influence the computation of the metric.
        # For ASR_NoTarget, the code will delete all the samples whose ground-truth labels are the target label and then compute the metric.
        'metric': 'ASR_NoTarget',
        'y_target': 0,

        'save_dir': 'experiments',
        'experiment_name': f'{model_name}_{dataset_name}_{attack_name}_{defense_name}_ASR_NoTarget'
    }

    print("=> DBD: Self-supervised (simclr) with unlabeled data...")
    schedule = {
        'device': 'GPU',
        'CUDA_VISIBLE_DEVICES': CUDA_VISIBLE_DEVICES,
        'GPU_num': 1,

        'batch_size': 512,
        'num_workers': 4,
        'pin_memory': True,

        # transform:
        'transform_aug': aug_transform,

        'sync_bn': True,  # synchronized batch normalization

        # criterion:
        'temperature': 0.5,

        # optimizer:
        'weight_decay': 1.e-4,
        'momentum': 0.9,
        'lr': 0.4,

        # lr_scheduler
        'scheduler': 'cosine_annealing',
        'T_max': 1000,  # same as epochs 1000

        'epochs': 1000,  # 1000
        'early_stop_epoch': 1000,

        # resume
        # 'resume': '',

        'num_stage_epochs': 100,
        'min_interval': 20,
        'max_interval': 100,

        'experiment_name': f'{model_name}_{dataset_name}_{attack_name}_{defense_name}_simclr',

        'log_iteration_interval': 30,
        'test_epoch_interval': 10,
        'save_epoch_interval': 100,

        'save_dir': 'experiments',

        # test schedule:
        'benign_test_schedule': benign_test_schedule,
        'poisoned_test_schedule': poisoned_test_schedule,

    }

    defense.simclr(schedule)

    print("=> DBD: Finetune and Mixmatch...")
    schedule = {
        'device': 'GPU',
        'CUDA_VISIBLE_DEVICES': CUDA_VISIBLE_DEVICES,
        'GPU_num': 1,

        # pretrain simclr model
        # 'pretrain_simclr_backbone_checkpoint': './'

        # 'resume':'',

        # train transform
        'train_transform': transform_train,

        # warm loader
        'warmup_batch_size': 128,
        'warmup_num_workers': 4,
        'warmup_pin_memory': True,

        # warmup-criterion:
        'warmup_alpha': 0.1,
        'warmup_beta': 1,
        'warmup_epochs': 10,

        'semi_epsilon': 0.5,

        # semi-criterion:
        'semi_lambda_u': 15,
        'semi_rampup_length': 190,

        # semi-loader:
        'semi_batch_size': 64,
        'semi_num_workers': 4,
        'semi_pin_memory': True,

        # optimizer-Adam
        'lr': 0.002,

        # scheduler
        # 'scheduler':'',

        # mixmatch:
        'mixmatch_train_iteration': 1024,
        'mixmatch_temperature': 0.5,
        'mixmatch_alpha': 0.75,
        'semi_epochs': 190,

        'log_iteration_interval': 30,
        'test_epoch_interval': 10,
        'save_epoch_interval': 10,

        'save_dir': 'experiments',
        'experiment_name': f'{model_name}_{dataset_name}_{attack_name}_{defense_name}' + '_mixmatch_finetune',

        # test schedule:
        'benign_test_schedule': benign_test_schedule,
        'poisoned_test_schedule': poisoned_test_schedule,
    }
    defense.mixmatch_finetune(schedule)
    print("=> DBD Done")
    return


if __name__ == '__main__':
    DBD_against_BadNets_CIFAR10()
    DBD_against_WaNet_CIFAR10()
    DBD_against_LabelConsistent_CIFAR10()

    DBD_against_BadNets_GTSRB()
    DBD_against_WatNet_GTSRB()
    DBD_against_LabelConsistent_GTSRB()

