import os
from copy import deepcopy
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os.path as osp
import random
from PIL import ImageFilter

from .base import Base
from ..utils import Log, test

import copy

import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
import time
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
import PIL
from PIL import Image
from torchvision.datasets import DatasetFolder, CIFAR10, MNIST
from tabulate import tabulate

class Record(object):
    def __init__(self, name, size):
        self.name = name
        self.size = size
        self.reset()

    def reset(self):
        self.ptr = 0
        self.data = torch.zeros(self.size)

    def update(self, batch_data):
        self.data[self.ptr : self.ptr + len(batch_data)] = batch_data
        self.ptr += len(batch_data)

def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch

    return offsets


def interleave(xy, batch):
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p] : offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]

    return [torch.cat(v, dim=0) for v in xy]


class MixMatchDataset(Dataset):
    """Semi-supervised MixMatch dataset.

    Args:
        dataset (Dataset): The dataset to be wrapped.
        semi_idx (np.array): An 0/1 (labeled/unlabeled) array with shape ``(len(dataset), )``.
        labeled (bool): If True, creates dataset from labeled set, otherwise creates from unlabeled
            set (default: True).
    """

    def __init__(self, dataset, semi_idx, labeled=True):
        super(MixMatchDataset, self).__init__()
        self.dataset = copy.deepcopy(dataset)
        if labeled:
            self.semi_indice = np.nonzero(semi_idx == 1)[0]
        else:
            self.semi_indice = np.nonzero(semi_idx == 0)[0]
        self.labeled = labeled


    def __getitem__(self, index):
        if self.labeled:
            img, target = self.dataset[self.semi_indice[index]]
            item = {"img": img, "target": target, "labeled": True}
        else:
            img1, target = self.dataset[self.semi_indice[index]]
            img2, target = self.dataset[self.semi_indice[index]]
            item = {"img1": img1, "img2": img2, "target": target, "labeled": False}
        return item

    def __len__(self):
        return len(self.semi_indice)



class SCELoss(nn.Module):
    """Symmetric Cross Entropy."""

    def __init__(self, alpha=0.1, beta=1, num_classes=10, reduction="mean"):
        super(SCELoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.reduction = reduction

    def forward(self, x, target):
        ce = torch.nn.CrossEntropyLoss(reduction=self.reduction)
        rce = RCELoss(num_classes=self.num_classes, reduction=self.reduction)
        ce_loss = ce(x, target)
        rce_loss = rce(x, target)
        loss = self.alpha * ce_loss + self.beta * rce_loss

        return loss

class RCELoss(nn.Module):
    """Reverse Cross Entropy Loss."""

    def __init__(self, num_classes=10, reduction="mean"):
        super(RCELoss, self).__init__()
        self.num_classes = num_classes
        self.reduction = reduction

    def forward(self, x, target):
        prob = F.softmax(x, dim=-1)
        prob = torch.clamp(prob, min=1e-7, max=1.0)
        one_hot = F.one_hot(target, self.num_classes).float()
        one_hot = torch.clamp(one_hot, min=1e-4, max=1.0)
        loss = -1 * torch.sum(prob * torch.log(one_hot), dim=-1)
        if self.reduction == "mean":
            loss = loss.mean()

        return loss


class AverageMeter(object):
    """Computes and stores the average and current value.

    Modified from https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """

    def __init__(self, name, fmt=None):
        self.name = name
        self.reset()

    def reset(self):
        self.batch_avg = 0
        self.total_avg = 0
        self.sum = 0
        self.count = 0

    def update(self, avg, n=1):
        self.batch_avg = avg
        self.sum += avg * n
        self.count += n
        self.total_avg = self.sum / self.count



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)

        return out


class ResNetBackbone(nn.Module):
    def __init__(
        self, block, num_blocks, num_classes=10, in_channel=3, zero_init_residual=False
    ):
        super(ResNetBackbone, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(
            in_channel, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves
        # like an identity. This improves the model by 0.2~0.3% according to:
        # https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i in range(num_blocks):
            stride = strides[i]
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return out


def get_backbone(model_name, num_classes=10):
    if model_name == 'resnet18':
        backbone = ResNetBackbone(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
        backbone.feature_dim = 512
    return backbone


class SelfModel(nn.Module):
    def __init__(self, backbone, head="mlp", proj_dim=128):
        super(SelfModel, self).__init__()
        self.backbone = backbone
        self.head = head
        if head == "linear":
            self.proj_head = nn.Linear(self.backbone.feature_dim, proj_dim)
        elif head == "mlp":
            self.proj_head = nn.Sequential(
                nn.Linear(self.backbone.feature_dim, self.backbone.feature_dim),
                nn.BatchNorm1d(self.backbone.feature_dim),
                nn.ReLU(),
                nn.Linear(self.backbone.feature_dim, proj_dim),
            )
        else:
            raise ValueError("Invalid head {}".format(head))

    def forward(self, x):
        feature = self.proj_head(self.backbone(x))
        feature = F.normalize(feature, dim=1)

        return feature

class SimclrPoisonDatasetFolder(DatasetFolder):
    """Self-supervised poison-label contrastive dataset.
    """
    def __init__(self,
                 dataset,
                 aug_transform
                 ):
        super(SimclrPoisonDatasetFolder, self).__init__(
            dataset.root,
            dataset.loader,
            dataset.extensions,
            dataset.transform,
            None)

        self.dataset = copy.deepcopy(dataset)
        self.aug_transform = aug_transform
        self.pre_transform = aug_transform["pre"]
        self.primary_transform = aug_transform["primary"]
        self.remaining_transform = aug_transform["remaining"]

    def __getitem__(self, index):
        img, target = self.dataset[index]
        if index in self.dataset.poisoned_set:
            img = self.augment(img)
            img1 = img
            img2 = img
            _, origin = self.dataset.samples[index]
            origin = int(origin)
            target = int(target)
            poison = 1
        else:
            target = int(target)
            origin = target
            img = self.augment(img)
            img1 = img
            img2 = img
            poison = 0
        item = {
            "img1": img1,
            "img2": img2,
            "target": target,
            "poison": poison,
            "origin": origin,
        }
        return item

    def __len__(self):
        return len(self.dataset.samples)

    def augment(self, img):
        '''
        Args:
            img: PIL.Image.Image

        Returns:
            img: Tensor
        '''
        img = self.pre_transform(img)
        img = np.array(img)

        img = Image.fromarray(img)
        img = self.primary_transform(img)
        img = self.remaining_transform(img)
        return img



class SimclrPoisonCifar10(Dataset):
    """Self-supervised poison-label contrastive dataset.
    """
    def __init__(self,
                 dataset,
                 aug_transform
                 ):
        super(SimclrPoisonCifar10, self).__init__()

        self.dataset = copy.deepcopy(dataset)

        self.aug_transform = aug_transform
        self.pre_transform = aug_transform["pre"]
        self.primary_transform = aug_transform["primary"]
        self.remaining_transform = aug_transform["remaining"]


    def __getitem__(self, index):
        img, target = self.dataset[index]

        if index in self.dataset.poisoned_set:
            img = self.augment(img)
            img1 = img
            img2 = img
            origin = int(self.dataset.targets[index])
            target = int(target)
            poison = 1

        else:
            poison = 0
            target = int(target)
            origin = target
            img1 = self.augment(img)
            img2 = self.augment(img)

        item = {
            "img1": img1,
            "img2": img2,
            "target": target,
            "poison": poison,
            "origin": origin,
        }
        return item

    def __len__(self):
        return len(self.dataset.data)

    def augment(self, img):
        img = self.pre_transform(img)
        img = np.array(img)

        img = Image.fromarray(img)
        img = self.primary_transform(img)
        img = self.remaining_transform(img)
        return img



class MixMatchPoisonDataset(Dataset):
    """Self-supervised poison-label contrastive dataset.
    """
    def __init__(self,
                 dataset,
                 train_transform
                 ):
        super(MixMatchPoisonDataset, self).__init__()

        self.dataset = copy.deepcopy(dataset)
        self.train_transform = train_transform


    def __getitem__(self, index):
        img, target = self.dataset[index]
        if isinstance(img, PIL.Image.Image):
            img = self.train_transform(img)
        elif isinstance(img, np.ndarray):
            img = img.transpose(2, 0, 1)
            img = torch.from_numpy(img).float()
        return img, target

    def __len__(self):
        if hasattr(self.dataset, 'data'):
            return len(self.dataset.data)
        elif hasattr(self.dataset, 'samples'):
            return len(self.dataset.samples)


class MixMatchLoss(nn.Module):
    """SemiLoss in MixMatch.

    Modified from https://github.com/YU1ut/MixMatch-pytorch/blob/master/train.py.
    """

    def __init__(self, rampup_length, lambda_u=75):
        super(MixMatchLoss, self).__init__()
        self.rampup_length = rampup_length
        self.lambda_u = lambda_u
        self.current_lambda_u = lambda_u

    def linear_rampup(self, epoch):
        if self.rampup_length == 0:
            return 1.0
        else:
            current = np.clip(epoch / self.rampup_length, 0.0, 1.0)
            self.current_lambda_u = float(current) * self.lambda_u

    def forward(self, xoutput, xtarget, uoutput, utarget, epoch):
        self.linear_rampup(epoch)
        uprob = torch.softmax(uoutput, dim=1)
        Lx = -torch.mean(torch.sum(F.log_softmax(xoutput, dim=1) * xtarget, dim=1))
        Lu = torch.mean((uprob - utarget) ** 2)

        return Lx, Lu, self.current_lambda_u

class SimCLRLoss(nn.Module):
    """Modified from https://github.com/wvangansbeke/Unsupervised-Classification."""

    def __init__(self, temperature, reduction="mean"):
        super(SimCLRLoss, self).__init__()
        self.temperature = temperature
        self.reduction = reduction

    def forward(self, features):
        """
        input:
            - features: hidden feature representation of shape [b, 2, dim]
        output:
            - loss: loss computed according to SimCLR
        """

        b, n, dim = features.size()
        assert n == 2
        mask = torch.eye(b, dtype=torch.float32).cuda()

        contrast_features = torch.cat(torch.unbind(features, dim=1), dim=0)
        anchor = features[:, 0]

        # Dot product
        dot_product = torch.matmul(anchor, contrast_features.T) / self.temperature

        # Log-sum trick for numerical stability
        logits_max, _ = torch.max(dot_product, dim=1, keepdim=True)
        logits = dot_product - logits_max.detach()

        mask = mask.repeat(1, 2)
        logits_mask = torch.scatter(
            torch.ones_like(mask), 1, torch.arange(b).view(-1, 1).cuda(), 0
        )
        mask = mask * logits_mask

        # Log-softmax
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # Mean log-likelihood for positive
        if self.reduction == "mean":
            loss = -((mask * log_prob).sum(1) / mask.sum(1)).mean()
        elif self.reduction == "none":
            loss = -((mask * log_prob).sum(1) / mask.sum(1))
        else:
            raise ValueError("The reduction must be mean or none!")

        return loss


class GatherLayer(torch.autograd.Function):
    """Gather tensors from all process, supporting backward propagation.

    Modified from https://github.com/open-mmlab/OpenSelfSup.
    """

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        (input,) = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR.

    Borrowed from https://github.com/facebookresearch/moco/blob/master/moco/loader.py.
    """

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))

        return x


class LinearModel(nn.Module):
    def __init__(self, backbone, feature_dim, num_classes):
        super(LinearModel, self).__init__()
        self.backbone = backbone
        self.linear = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        feature = self.backbone(x)
        out = self.linear(feature)
        return out

    def update_encoder(self, backbone):
        self.backbone = backbone


class DBD(Base):
    """Construct defense datasets with ShrinkPad method.

    Args:
        model_name='resnet18',
       head='mlp',
       poisoned_train_data=clean_train_data,
       poisoned_test_dataset=poisoned_test_data,
       benign_test_dataset=clean_test_data,
       y_target=1,
       poison_train_idx=poison_train_idx,
       seed=global_seed,
       deterministic=deterministic

        seed (int): Global seed for random numbers. Default: 0.
        deterministic (bool): Sets whether PyTorch operations must use "deterministic" algorithms.
            That is, algorithms which, given the same input, and when run on the same software and hardware,
            always produce the same output. When enabled, operations will use deterministic algorithms when available,
            and if only nondeterministic algorithms are available they will throw a RuntimeError when called. Default: False.
    """

    def __init__(self,
                 model_name,
                 num_classes,
                 head='mlp',
                 poisoned_train_dataset=None,
                 poisoned_test_dataset=None,
                 benign_test_dataset=None,
                 seed=0,
                 deterministic=False):
        super(DBD, self).__init__(seed=seed, deterministic=deterministic)

        # prepare backbone and head
        self.num_classes = num_classes
        backbone = get_backbone(model_name=model_name, num_classes=self.num_classes)
        self.backbone_model = SelfModel(backbone=backbone, head=head)

        # prepare the linear model
        self.linear_model = LinearModel(self.backbone_model.backbone,
                                        self.backbone_model.backbone.feature_dim,
                                        self.num_classes)

        # prepare unlabeled dataset for simclr
        self.poisoned_train_dataset = poisoned_train_dataset
        self.poisoned_test_dataset = poisoned_test_dataset
        self.benign_test_dataset = benign_test_dataset

        # prepare poisoned dataset
        self.poison_set = self.poisoned_train_dataset.poisoned_set
        self.global_schedule = None

    def simclr(self, schedule):
        if schedule is None and self.global_schedule is None:
            raise AttributeError("Training schedule is None, please check your schedule setting.")
        elif schedule is not None and self.global_schedule is None:
            self.current_schedule = deepcopy(schedule)
        elif schedule is None and self.global_schedule is not None:
            self.current_schedule = deepcopy(self.global_schedule)
        elif schedule is not None and self.global_schedule is not None:
            self.current_schedule = deepcopy(schedule)

        # prepare log
        work_dir = osp.join(self.current_schedule['save_dir'],
                            self.current_schedule['experiment_name'] + '_' + time.strftime("%Y-%m-%d_%H:%M:%S",
                                                                                           time.localtime()))
        os.makedirs(work_dir, exist_ok=True)
        self.log = Log(osp.join(work_dir, 'log.txt'))

        if 'resume' in self.current_schedule:
            self.backbone_model.load_state_dict(torch.load(self.current_schedule['resume']), strict=True)

        # Use GPU
        if 'device' in self.current_schedule and self.current_schedule['device'] == 'GPU':
            if 'CUDA_VISIBLE_DEVICES' in schedule:
                os.environ['CUDA_VISIBLE_DEVICES'] = self.current_schedule['CUDA_VISIBLE_DEVICES']

            assert torch.cuda.device_count() > 0, 'This machine has no cuda devices!'
            assert self.current_schedule['GPU_num'] > 0, 'GPU_num should be a positive integer'
            print(f"This machine has {torch.cuda.device_count()} cuda devices, and use {self.current_schedule['GPU_num']} of them to train.")

            if self.current_schedule['GPU_num'] == 1:
                device = torch.device("cuda:0")
            else:
                gpus = list(range(self.current_schedule['GPU_num']))
                self.backbone_model= nn.DataParallel(self.backbone_model.cuda(), device_ids=gpus, output_device=gpus[0])
                # TODO: DDP training
                pass
        # Use CPU
        else:
            device = torch.device("cpu")

        self.log('Create dataloader for simclr')
        if isinstance(self.poisoned_train_dataset, DatasetFolder):
            self.simclr_poisoned_dataset = SimclrPoisonDatasetFolder(
                dataset=self.poisoned_train_dataset,
                aug_transform=self.current_schedule['transform_aug']
                )
        elif isinstance(self.poisoned_train_dataset, CIFAR10):
            self.simclr_poisoned_dataset = SimclrPoisonCifar10(dataset=self.poisoned_train_dataset,
                                                               aug_transform=self.current_schedule['transform_aug']
                                                               )
        self.simclr_poisoned_train_loader = DataLoader(self.simclr_poisoned_dataset,
                                                     batch_size=self.current_schedule['batch_size'],
                                                     shuffle=True,
                                                     num_workers=self.current_schedule['num_workers'],
                                                     drop_last=False,
                                                     pin_memory=True
                                                     )

        self.backbone_model = self.backbone_model.to(device)

        criterion = SimCLRLoss(temperature=self.current_schedule['temperature'])
        criterion = criterion.to(device)

        optimizer = torch.optim.SGD(self.backbone_model.parameters(),
                                    weight_decay=self.current_schedule['weight_decay'],
                                    momentum=self.current_schedule['momentum'],
                                    lr=self.current_schedule['lr'])
        self.log("Create optimizer: {}".format(optimizer))

        if 'scheduler' in self.current_schedule:
            if self.current_schedule['scheduler'] == 'cosine_annealing':
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=self.current_schedule['T_max']
                )
            else:
                scheduler = None
                self.log("Only support cosine_annealing")

            self.log("Create scheduler: {}".format(self.current_schedule['scheduler']))
        else:
            scheduler = None

        # log and output:
        # 1. ouput loss and time
        # 2. test and output statistics
        # 3. save checkpoint

        msg = f"Total train samples: {len(self.poisoned_train_dataset)}\nTotal test samples: {len(self.poisoned_test_dataset)}\nBatch size: {self.current_schedule['batch_size']}\niteration every epoch: {len(self.poisoned_train_dataset) // self.current_schedule['batch_size']}\nInitial learning rate: {self.current_schedule['lr']}\n"
        self.log(msg)

        self.log("SimCLR training...")
        for i in range(self.current_schedule["epochs"]):
            if i >= self.current_schedule['early_stop_epoch']:
                break
            self.log("===Epoch: {}/{}===".format(i + 1, self.current_schedule["epochs"]))

            self_train_result = self._simclr_train(criterion, optimizer, epoch=i)

            if (i + 1) % self.current_schedule['test_epoch_interval'] == 0:
                self.backbone_model.eval()
                test(self.backbone_model, self.benign_test_dataset, schedule=self.current_schedule['benign_test_schedule'])
                test(self.backbone_model, self.poisoned_test_dataset,
                     schedule=self.current_schedule['poisoned_test_schedule'])

                self.backbone_model.train()

            if (i + 1) % self.current_schedule['save_epoch_interval'] == 0:
                self.backbone_model.eval()
                self.backbone_model.backbone= self.backbone_model.backbone.cpu()
                ckpt_model_filename = "ckpt_epoch_" + str(i+1) + ".pth"
                ckpt_model_path = os.path.join(work_dir, ckpt_model_filename)
                torch.save(self.backbone_model.backbone.state_dict(), ckpt_model_path)
                self.backbone_model = self.backbone_model.to(device)
                self.backbone_model.train()

            if scheduler is not None:
                scheduler.step()
                self.log("Adjust learning rate to {}".format(optimizer.param_groups[0]["lr"]))
        return

    def _simclr_train(self, criterion, optimizer, epoch):
        loss_meter = AverageMeter("loss")
        meter_list = [loss_meter]

        self.backbone_model.train()
        iteration = 0
        last_time = time.time()
        gpu = next(self.backbone_model.parameters()).device
        ddp = isinstance(self.backbone_model, DistributedDataParallel)
        for batch_idx, batch in enumerate(self.simclr_poisoned_train_loader):
            img1, img2 = batch["img1"], batch["img2"]
            data = torch.cat([img1.unsqueeze(1), img2.unsqueeze(1)], dim=1)
            b, c, h, w = img1.size()
            data = data.view(-1, c, h, w)
            data = data.cuda(gpu, non_blocking=True)

            optimizer.zero_grad()

            output = self.backbone_model(data).view(b, 2, -1)
            if ddp:
                output = torch.cat(GatherLayer.apply(output), dim=0)
            loss = criterion(output)
            loss.backward()
            optimizer.step()
            loss_meter.update(loss.item())
            iteration += 1
            if iteration % self.current_schedule['log_iteration_interval'] == 0:
                msg = time.strftime("[%Y-%m-%d_%H:%M:%S] ",
                                    time.localtime()) + f"Epoch:{epoch + 1}/{self.current_schedule['epochs']}, iteration:{batch_idx + 1}/{len(self.poisoned_train_dataset) // self.current_schedule['batch_size']}, lr: {optimizer.param_groups[0]['lr']}, loss: {float(loss)}, time: {time.time() - last_time}\n"
                last_time = time.time()
                self.log(msg)

        result = {m.name: m.total_avg for m in meter_list}
        return result

    def get_semi_idx(self, record_list, ratio):
        """Get labeled and unlabeled index.
        """
        keys = [r.name for r in record_list]
        loss = record_list[keys.index("loss")].data.numpy()
        semi_idx = np.zeros(len(loss))
        # Sort loss and fetch `ratio` of the smallest indices.
        indice = loss.argsort()[: int(len(loss) * ratio)]
        self.log(
             "{}/{} poisoned samples in semi_idx".format(len(set(indice) & self.poison_set), len(indice))
        )
        semi_idx[indice] = 1
        return semi_idx

    def _accuracy(self, output, target, topk=(1,)):
        """Computes the precision@k for the specified values of k"""
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

    def mixmatch_finetune(self, schedule):
        if schedule is None and self.global_schedule is None:
            raise AttributeError("Training schedule is None, please check your schedule setting.")
        elif schedule is not None and self.global_schedule is None:
            self.current_schedule = deepcopy(schedule)
        elif schedule is None and self.global_schedule is not None:
            self.current_schedule = deepcopy(self.global_schedule)
        elif schedule is not None and self.global_schedule is not None:
            self.current_schedule = deepcopy(schedule)

        # prepare log
        work_dir = osp.join(self.current_schedule['save_dir'],
                            self.current_schedule['experiment_name'] + '_' + time.strftime("%Y-%m-%d_%H:%M:%S",
                                                                                           time.localtime()))
        os.makedirs(work_dir, exist_ok=True)
        self.log = Log(osp.join(work_dir, 'log.txt'))

        # Use GPU
        if 'device' in self.current_schedule and self.current_schedule['device'] == 'GPU':
            if 'CUDA_VISIBLE_DEVICES' in schedule:
                os.environ['CUDA_VISIBLE_DEVICES'] = self.current_schedule['CUDA_VISIBLE_DEVICES']

            assert torch.cuda.device_count() > 0, 'This machine has no cuda devices!'
            assert self.current_schedule['GPU_num'] > 0, 'GPU_num should be a positive integer'
            print(
                f"This machine has {torch.cuda.device_count()} cuda devices, and use {self.current_schedule['GPU_num']} of them to train.")

            if self.current_schedule['GPU_num'] == 1:
                device = torch.device("cuda:0")
            else:
                gpus = list(range(self.current_schedule['GPU_num']))
                self.backbone_model = nn.DataParallel(self.backbone_model.cuda(), device_ids=gpus, output_device=gpus[0])
                # TODO: DDP training
                pass
        # Use CPU
        else:
            device = torch.device("cpu")

        self.log('Create dataloader for mixmatch_finetune')
        self.mixmatch_poisoned_train_dataset = MixMatchPoisonDataset(self.poisoned_train_dataset, self.current_schedule['train_transform'])

        self.poisoned_train_dataloader = DataLoader(self.mixmatch_poisoned_train_dataset,
                                                     batch_size=self.current_schedule['warmup_batch_size'],
                                                     num_workers=self.current_schedule['warmup_num_workers'],
                                                     drop_last=False,
                                                     pin_memory=self.current_schedule['warmup_pin_memory'],
                                                     shuffle=True,
                                                     )
        self.poisoned_eval_loader = DataLoader(self.mixmatch_poisoned_train_dataset,
                                      batch_size=self.current_schedule['warmup_batch_size'],
                                      num_workers=self.current_schedule['warmup_num_workers'],
                                      drop_last=False,
                                      pin_memory=True
                                      )
        self.benign_test_loader = DataLoader(self.benign_test_dataset,
                                             batch_size=self.current_schedule['warmup_batch_size'],
                                             num_workers=self.current_schedule['warmup_num_workers'],
                                             drop_last=False,
                                             pin_memory=True)
        self.poisoned_test_loader = DataLoader(self.poisoned_test_dataset,
                                             batch_size=self.current_schedule['warmup_batch_size'],
                                             num_workers=self.current_schedule['warmup_num_workers'],
                                             drop_last=False,
                                             pin_memory=True)


        self.log('Setup backbone and linear model')
        if 'pretrain_simclr_backbone_checkpoint' in self.current_schedule:
            self.linear_model.backbone.load_state_dict(torch.load(self.current_schedule['pretrain_simclr_backbone_checkpoint']), strict=False)
            self.log('Load backbone from {}'.format(self.current_schedule['pretrain_simclr_backbone_checkpoint']))
        else:
            self.log('Load the lastest backbone.')
            self.linear_model.update_encoder(self.backbone_model.backbone)


        if 'resume' in self.current_schedule:
            self.linear_model.load_state_dict(torch.load(self.current_schedule['resume']), strict=True)
            self.log('Load model from {}'.format(self.current_schedule['resume']))

        self.linear_model.backbone.to(device)
        self.linear_model.linear.to(device)

        self.log('Setup training')
        self.log("Create criterion for warmup")
        warmup_criterion = SCELoss(alpha=self.current_schedule['warmup_alpha'],
                                   beta=self.current_schedule['warmup_beta'],
                                   num_classes=self.num_classes)

        warmup_criterion = warmup_criterion.to(device)
        self.log("Create criterion: {} for warmup".format(warmup_criterion))

        semi_criterion = MixMatchLoss(rampup_length=self.current_schedule['semi_rampup_length'],
                                      lambda_u=self.current_schedule['semi_lambda_u'])
        semi_criterion = semi_criterion.to(device)
        self.log("Create criterion: {} for semi-training".format(semi_criterion))

        optimizer = torch.optim.Adam(self.linear_model.parameters(),
                                     lr=self.current_schedule['lr'])
        self.log("Create optimizer: {}".format(optimizer))

        if 'scheduler' in self.current_schedule:
            if self.current_schedule['scheduler'] == 'cosine_annealing':
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=self.current_schedule['T_max']
                )
            else:
                scheduler = None
                self.log("Only support cosine_annealing")

            self.log("Create scheduler: {}".format(self.current_schedule['scheduler']))
        else:
            scheduler = None

        # log and output:
        # 1. ouput loss and time
        # 2. test and output statistics
        # 3. save checkpoint

        self.log("Warmup and Mixmatch training...")
        num_epochs = self.current_schedule["warmup_epochs"] + self.current_schedule["semi_epochs"]

        for i in range(num_epochs):
            self.log("===Epoch: {}/{}===".format(i + 1, num_epochs))
            if (i + 1) <= self.current_schedule["warmup_epochs"]:
                if i == 0:
                    self.log("Poisoned linear warmup...")
                    msg = f"Total train samples: {len(self.mixmatch_poisoned_train_dataset)}\nTotal test samples: {len(self.poisoned_test_dataset)}\nBatch size: {self.current_schedule['warmup_batch_size']}\niteration every epoch: {len(self.mixmatch_poisoned_train_dataset) // self.current_schedule['warmup_batch_size']}\nInitial learning rate: {self.current_schedule['lr']}\n"
                    self.log(msg)
                self.poison_linear_train(
                    i,
                    warmup_criterion,
                    optimizer,
                    device
                )
            else:
                if i == self.current_schedule["warmup_epochs"]:
                    self.log("Mixmatch training...")
                    msg = f"Total train samples: {len(self.mixmatch_poisoned_train_dataset)}\nTotal test samples: {len(self.poisoned_test_dataset)}\nBatch size: {self.current_schedule['semi_batch_size']}\niteration every epoch: {len(self.mixmatch_poisoned_train_dataset) // self.current_schedule['semi_batch_size']}\nInitial learning rate: {self.current_schedule['lr']}\n"
                    self.log(msg)
                record_list = self.poison_linear_record(
                    self.poisoned_eval_loader, warmup_criterion, device
                )
                self.log("Mining clean data from poisoned dataset...")
                semi_idx = self.get_semi_idx(record_list, self.current_schedule['semi_epsilon'])
                xdata = MixMatchDataset(self.mixmatch_poisoned_train_dataset, semi_idx, labeled=True)
                udata = MixMatchDataset(self.mixmatch_poisoned_train_dataset, semi_idx, labeled=False)
                xloader = DataLoader(xdata,
                                     batch_size=self.current_schedule['semi_batch_size'],
                                     num_workers=self.current_schedule['semi_num_workers'],
                                     pin_memory=self.current_schedule['semi_pin_memory'],
                                     shuffle=True,
                                     drop_last=True
                                     )
                uloader = DataLoader(udata,
                                     batch_size=self.current_schedule['semi_batch_size'],
                                     num_workers=self.current_schedule['semi_num_workers'],
                                     pin_memory=self.current_schedule['semi_pin_memory'],
                                     shuffle=True,
                                     drop_last=True
                                     )
                self.log("MixMatch training...")
                poison_train_result = self.mixmatch_train(
                    xloader,
                    uloader,
                    semi_criterion,
                    optimizer,
                    i
                )

            if (i + 1) % self.current_schedule['test_epoch_interval'] == 0:
                # test result on benign test dataset
                test(self.linear_model, self.benign_test_dataset,
                     schedule=self.current_schedule['benign_test_schedule'])

                test(self.linear_model, self.poisoned_test_dataset,
                     schedule=self.current_schedule['poisoned_test_schedule'])


                self.backbone_model.train()

            if (i + 1) % self.current_schedule['save_epoch_interval'] == 0:
                self.linear_model.eval()
                self.linear_model = self.linear_model.cpu()
                ckpt_model_filename = "ckpt_epoch_" + str(i + 1) + ".pth"
                ckpt_model_path = os.path.join(work_dir, ckpt_model_filename)
                torch.save(self.linear_model.state_dict(), ckpt_model_path)
                self.linear_model = self.linear_model.to(device)
                self.linear_model.train()

            if scheduler is not None:
                scheduler.step()
                self.log("Adjust learning rate to {}".format(optimizer.param_groups[0]["lr"]))
        return

    def poison_linear_train(self, epoch, criterion, optimizer, device, frozen=True):
        loss_meter = AverageMeter("loss")
        poison_loss_meter = AverageMeter("poison loss")
        clean_loss_meter = AverageMeter("clean loss")
        acc_meter = AverageMeter("acc")
        poison_acc_meter = AverageMeter("poison acc")
        clean_acc_meter = AverageMeter("clean acc")
        meter_list = [
            loss_meter,
            poison_loss_meter,
            clean_loss_meter,
            acc_meter,
            poison_acc_meter,
            clean_acc_meter,
        ]

        if frozen:
            # Freeze the backbone.
            for param in self.linear_model.backbone.parameters():
                param.require_grad = False
        self.linear_model.train()

        iteration = 0
        last_time = time.time()
        for batch_idx, batch in enumerate(self.poisoned_train_dataloader):
            data = batch[0].to(device)
            target = batch[1].to(device)
            if frozen:
                with torch.no_grad():
                    feature = self.linear_model.backbone(data)
            else:
                feature = self.linear_model.backbone(data)
            output = self.linear_model.linear(feature)
            criterion.reduction = "none"
            raw_loss = criterion(output, target)
            criterion.reduction = "mean"
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_meter.update(loss.item())
            pred = output.argmax(dim=1, keepdim=True)
            truth = pred.view_as(target).eq(target)
            acc_meter.update((torch.sum(truth).float() / len(truth)).item())
            iteration += 1

            if iteration % self.current_schedule['log_iteration_interval'] == 0:
                msg = time.strftime("[%Y-%m-%d_%H:%M:%S] ",
                                    time.localtime()) + \
                                    f"Epoch:{epoch + 1}/{self.current_schedule['warmup_epochs']}, iteration:{batch_idx + 1}/{len(self.mixmatch_poisoned_train_dataset) // self.current_schedule['warmup_batch_size']}, lr: {self.current_schedule['lr']}, loss: {float(loss)}, time: {time.time() - last_time}\n"
                last_time = time.time()
                self.log(msg)

        if frozen:
            # Unfreeze the backbone.
            for param in self.linear_model.backbone.parameters():
                param.require_grad = True
        return

    def poison_linear_record(self, loader, criterion, device):
        num_data = len(loader.dataset)
        target_record = Record("target", num_data)
        loss_record = Record("loss", num_data)
        poison_record = Record("poison", num_data)
        feature_record = Record("feature", (num_data, self.linear_model.backbone.feature_dim))
        record_list = [
            target_record,
            poison_record,
            loss_record,
            feature_record,
        ]

        self.linear_model.eval()
        for _, batch in enumerate(loader):
            data = batch[0].to(device)
            target = batch[1].to(device)
            with torch.no_grad():
                feature = self.linear_model.backbone(data)
                output = self.linear_model.linear(feature)
            criterion.reduction = "none"
            raw_loss = criterion(output, target)

            target_record.update(batch[1])
            loss_record.update(raw_loss.cpu())
            feature_record.update(feature.cpu())

        return record_list

    def tabulate_step_meter(self, batch_idx, num_batches, num_intervals, meter_list):
        """Tabulate current average value of meters every ``step_interval``.

            Args:
                batch_idx (int): The batch index in an epoch.
                num_batches (int): The number of batch in an epoch.
                num_intervals (int): The number of interval to tabulate.
                meter_list (list or tuple of AverageMeter): A list of meters.
                logger (logging.logger): The logger.
            """
        step_interval = int(num_batches / num_intervals)
        if batch_idx % step_interval == 0:
            step_meter = {"Iteration": ["{}/{}".format(batch_idx, num_batches)]}
            for m in meter_list:
                step_meter[m.name] = [m.batch_avg]
            table = tabulate(step_meter, headers="keys", tablefmt="github", floatfmt=".5f")
            if batch_idx == 0:
                table = table.split("\n")
                table = "\n".join([table[1]] + table)
            else:
                table = table.split("\n")[2]
            self.log(table)


    def mixmatch_train(self, xloader, uloader, criterion, optimizer, epoch):
        loss_meter = AverageMeter("loss")
        xloss_meter = AverageMeter("xloss")
        uloss_meter = AverageMeter("uloss")
        lambda_u_meter = AverageMeter("lambda_u")
        meter_list = [loss_meter, xloss_meter, uloss_meter, lambda_u_meter]

        xiter = iter(xloader)
        uiter = iter(uloader)

        self.linear_model.train()
        gpu = next(self.linear_model.parameters()).device
        last_time = time.time()
        iteration = 0
        for batch_idx in range(self.current_schedule["mixmatch_train_iteration"]):
            try:
                xbatch = next(xiter)
                xinput, xtarget = xbatch["img"], xbatch["target"]
            except:
                xiter = iter(xloader)
                xbatch = next(xiter)
                xinput, xtarget = xbatch["img"], xbatch["target"]

            try:
                ubatch = next(uiter)
                uinput1, uinput2 = ubatch["img1"], ubatch["img2"]
            except:
                uiter = iter(uloader)
                ubatch = next(uiter)
                uinput1, uinput2 = ubatch["img1"], ubatch["img2"]

            batch_size = xinput.size(0)
            xtarget = torch.zeros(batch_size, self.num_classes).scatter_(
                1, xtarget.view(-1, 1).long(), 1
            )
            xinput = xinput.cuda(gpu, non_blocking=True)
            xtarget = xtarget.cuda(gpu, non_blocking=True)
            uinput1 = uinput1.cuda(gpu, non_blocking=True)
            uinput2 = uinput2.cuda(gpu, non_blocking=True)

            with torch.no_grad():
                # compute guessed labels of unlabel samples
                uoutput1 = self.linear_model(uinput1)
                uoutput2 = self.linear_model(uinput2)
                p = (torch.softmax(uoutput1, dim=1) + torch.softmax(uoutput2, dim=1)) / 2
                pt = p ** (1 / self.current_schedule["mixmatch_temperature"])
                utarget = pt / pt.sum(dim=1, keepdim=True)
                utarget = utarget.detach()

            # mixup
            all_input = torch.cat([xinput, uinput1, uinput2], dim=0)
            all_target = torch.cat([xtarget, utarget, utarget], dim=0)
            l = np.random.beta(self.current_schedule["mixmatch_alpha"], self.current_schedule["mixmatch_alpha"])
            l = max(l, 1 - l)
            idx = torch.randperm(all_input.size(0))
            input_a, input_b = all_input, all_input[idx]
            target_a, target_b = all_target, all_target[idx]
            mixed_input = l * input_a + (1 - l) * input_b
            mixed_target = l * target_a + (1 - l) * target_b

            # interleave labeled and unlabeled samples between batches to get correct batchnorm calculation
            mixed_input = list(torch.split(mixed_input, batch_size))
            mixed_input = interleave(mixed_input, batch_size)

            logit = [self.linear_model(mixed_input[0])]
            for input in mixed_input[1:]:
                logit.append(self.linear_model(input))

            # put interleaved samples back
            logit = interleave(logit, batch_size)
            xlogit = logit[0]
            ulogit = torch.cat(logit[1:], dim=0)

            Lx, Lu, lambda_u = criterion(
                xlogit,
                mixed_target[:batch_size],
                ulogit,
                mixed_target[batch_size:],
                epoch + batch_idx / self.current_schedule["mixmatch_train_iteration"],
            )
            loss = Lx + lambda_u * Lu
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            loss_meter.update(loss.item())
            xloss_meter.update(Lx.item())
            uloss_meter.update(Lu.item())
            lambda_u_meter.update(lambda_u)

            iteration += 1

            self.tabulate_step_meter(batch_idx, self.current_schedule["mixmatch_train_iteration"], 3, meter_list)


        result = {m.name: m.total_avg for m in meter_list}

        return result

    def _test(self, dataset, device, batch_size=16, num_workers=8, model=None):
        if model is None:
            model = self.backbone_model
        else:
            model = model

        with torch.no_grad():
            test_loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                drop_last=False,
                pin_memory=True
            )

            model = model.to(device)
            model.eval()

            predict_digits = []
            labels = []
            for batch in test_loader:
                batch_img, batch_label = batch
                batch_img = batch_img.to(device)
                batch_img = model(batch_img)
                batch_img = batch_img.cpu()
                predict_digits.append(batch_img)
                labels.append(batch_label)

            predict_digits = torch.cat(predict_digits, dim=0)
            labels = torch.cat(labels, dim=0)
            return predict_digits, labels
