import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms.functional as TF
import numpy as np
import os
import math
import random
import logging
import logging.handlers
from matplotlib import pyplot as plt

from scipy.ndimage import zoom
import SimpleITK as sitk
from medpy import metric

import cv2
from scipy.ndimage import rotate
from scipy.ndimage import affine_transform
from PIL import Image


def set_seed(seed):
    # for hash
    os.environ['PYTHONHASHSEED'] = str(seed)
    # for python and numpy
    random.seed(seed)
    np.random.seed(seed)
    # for cpu gpu
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # for cudnn
    cudnn.benchmark = False
    cudnn.deterministic = True


def get_logger(name, log_dir):
    '''
    Args:
        name(str): name of logger
        log_dir(str): path of log
    '''

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    info_name = os.path.join(log_dir, '{}.info.log'.format(name))
    info_handler = logging.handlers.TimedRotatingFileHandler(info_name,
                                                             when='D',
                                                             encoding='utf-8')
    info_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')

    info_handler.setFormatter(formatter)

    logger.addHandler(info_handler)

    return logger


def log_config_info(config, logger):
    config_dict = config.__dict__
    log_info = f'#----------Config info----------#'
    logger.info(log_info)
    for k, v in config_dict.items():
        if k[0] == '_':
            continue
        else:
            log_info = f'{k}: {v},'
            logger.info(log_info)


def get_optimizer(config, model):
    assert config.opt in ['Adadelta', 'Adagrad', 'Adam', 'AdamW', 'Adamax', 'ASGD', 'RMSprop', 'Rprop',
                          'SGD'], 'Unsupported optimizer!'

    if config.opt == 'Adadelta':
        return torch.optim.Adadelta(
            model.parameters(),
            lr=config.lr,
            rho=config.rho,
            eps=config.eps,
            weight_decay=config.weight_decay
        )
    elif config.opt == 'Adagrad':
        return torch.optim.Adagrad(
            model.parameters(),
            lr=config.lr,
            lr_decay=config.lr_decay,
            eps=config.eps,
            weight_decay=config.weight_decay
        )
    elif config.opt == 'Adam':
        return torch.optim.Adam(
            model.parameters(),
            lr=config.lr,
            betas=config.betas,
            eps=config.eps,
            weight_decay=config.weight_decay,
            amsgrad=config.amsgrad
        )
    elif config.opt == 'AdamW':
        return torch.optim.AdamW(
            model.parameters(),
            lr=config.lr,
            betas=config.betas,
            eps=config.eps,
            weight_decay=config.weight_decay,
            amsgrad=config.amsgrad
        )
    elif config.opt == 'Adamax':
        return torch.optim.Adamax(
            model.parameters(),
            lr=config.lr,
            betas=config.betas,
            eps=config.eps,
            weight_decay=config.weight_decay
        )
    elif config.opt == 'ASGD':
        return torch.optim.ASGD(
            model.parameters(),
            lr=config.lr,
            lambd=config.lambd,
            alpha=config.alpha,
            t0=config.t0,
            weight_decay=config.weight_decay
        )
    elif config.opt == 'RMSprop':
        return torch.optim.RMSprop(
            model.parameters(),
            lr=config.lr,
            momentum=config.momentum,
            alpha=config.alpha,
            eps=config.eps,
            centered=config.centered,
            weight_decay=config.weight_decay
        )
    elif config.opt == 'Rprop':
        return torch.optim.Rprop(
            model.parameters(),
            lr=config.lr,
            etas=config.etas,
            step_sizes=config.step_sizes,
        )
    elif config.opt == 'SGD':
        return torch.optim.SGD(
            model.parameters(),
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
            dampening=config.dampening,
            nesterov=config.nesterov
        )
    else:  # default opt is SGD
        return torch.optim.SGD(
            model.parameters(),
            lr=0.01,
            momentum=0.9,
            weight_decay=0.05,
        )


def get_scheduler(config, optimizer):
    assert config.sch in ['StepLR', 'MultiStepLR', 'ExponentialLR', 'CosineAnnealingLR', 'ReduceLROnPlateau',
                          'CosineAnnealingWarmRestarts', 'WP_MultiStepLR', 'WP_CosineLR'], 'Unsupported scheduler!'
    if config.sch == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.step_size,
            gamma=config.gamma,
            last_epoch=config.last_epoch
        )
    elif config.sch == 'MultiStepLR':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=config.milestones,
            gamma=config.gamma,
            last_epoch=config.last_epoch
        )
    elif config.sch == 'ExponentialLR':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=config.gamma,
            last_epoch=config.last_epoch
        )
    elif config.sch == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.T_max,
            eta_min=config.eta_min,
            last_epoch=config.last_epoch
        )
    elif config.sch == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=config.mode,
            factor=config.factor,
            patience=config.patience,
            threshold=config.threshold,
            threshold_mode=config.threshold_mode,
            cooldown=config.cooldown,
            min_lr=config.min_lr,
            eps=config.eps
        )
    elif config.sch == 'CosineAnnealingWarmRestarts':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config.T_0,
            T_mult=config.T_mult,
            eta_min=config.eta_min,
            last_epoch=config.last_epoch
        )
    elif config.sch == 'WP_MultiStepLR':
        lr_func = lambda \
            epoch: epoch / config.warm_up_epochs if epoch <= config.warm_up_epochs else config.gamma ** len(
            [m for m in config.milestones if m <= epoch])
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)
    elif config.sch == 'WP_CosineLR':
        lr_func = lambda epoch: epoch / config.warm_up_epochs if epoch <= config.warm_up_epochs else 0.5 * (
                math.cos((epoch - config.warm_up_epochs) / (config.epochs - config.warm_up_epochs) * math.pi) + 1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)

    return scheduler


# def save_imgs(img, msk, msk_pred, i, save_path, datasets, threshold=0.5, test_data_name=None):
#     img = img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
#     img = img / 255. if img.max() > 1.1 else img
#     if datasets == 'retinal':
#         msk = np.squeeze(msk, axis=0)
#         msk_pred = np.squeeze(msk_pred, axis=0)
#     else:
#         msk = np.where(np.squeeze(msk, axis=0) > 0.5, 1, 0)
#         msk_pred = np.where(np.squeeze(msk_pred, axis=0) > threshold, 1, 0)
#
#     plt.figure(figsize=(7, 15))
#
#     plt.subplot(3, 1, 1)
#     plt.imshow(img)
#     plt.axis('off')
#
#     plt.subplot(3, 1, 2)
#     plt.imshow(msk, cmap='gray')
#     plt.axis('off')
#
#     plt.subplot(3, 1, 3)
#     plt.imshow(msk_pred, cmap='gray')
#     plt.axis('off')
#
#     if test_data_name is not None:
#         save_path = save_path + test_data_name + '_'
#     plt.savefig(save_path + str(i) + '.png')
#     plt.close()
def save_imgs(img, msk, msk_pred, i, save_path, datasets, threshold=0.5, test_data_name=None):
    os.makedirs(save_path, exist_ok=True)
    print('>>> NEW save_imgs CALLED <<<')
    print(f'[save_imgs] save_path = {save_path}')
    img = img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    img = img / 255.0 if img.max() > 1.1 else img

    if datasets == 'retinal':
        msk = np.squeeze(msk, axis=0)
        msk_pred = np.squeeze(msk_pred, axis=0)
    else:
        msk = np.where(np.squeeze(msk, axis=0) > 0.5, 1, 0)
        msk_pred = np.where(np.squeeze(msk_pred, axis=0) > threshold, 1, 0)

    if img.dtype != np.uint8:
        img_to_save = (img * 255).clip(0, 255).astype(np.uint8)
    else:
        img_to_save = img

    if datasets == 'retinal':
        msk_to_save = np.asarray(msk)
        pred_to_save = np.asarray(msk_pred)
        if msk_to_save.dtype != np.uint8:
            msk_to_save = (msk_to_save * 255).clip(0, 255).astype(np.uint8)
        if pred_to_save.dtype != np.uint8:
            pred_to_save = (pred_to_save * 255).clip(0, 255).astype(np.uint8)
    else:
        msk_to_save = (msk * 255).astype(np.uint8)
        pred_to_save = (msk_pred * 255).astype(np.uint8)

    if test_data_name is not None:
        prefix = f'{test_data_name}_{i}'
    else:
        prefix = str(i)
    img_file = os.path.join(save_path, f'{prefix}_img.png')
    gt_file = os.path.join(save_path, f'{prefix}_gt.png')
    pred_file = os.path.join(save_path, f'{prefix}_pred.png')

    print(f'[save_imgs] img_file = {img_file}')
    print(f'[save_imgs] gt_file = {gt_file}')
    print(f'[save_imgs] pred_file = {pred_file}')

    # Image.fromarray(img_to_save).save(os.path.join(save_path, f'{prefix}_img.png'))
    # Image.fromarray(msk_to_save).save(os.path.join(save_path, f'{prefix}_gt.png'))
    # Image.fromarray(pred_to_save).save(os.path.join(save_path, f'{prefix}_pred.png'))
    Image.fromarray(img_to_save).save(img_file)
    Image.fromarray(msk_to_save).save(gt_file)
    Image.fromarray(pred_to_save).save(pred_file)
    print('>>> SAVE DONE <<<')

class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
        ######################
        #为了适配我原本的visionmamba与transunet双分支的AMP混合精度训练
        #self.bceloss = nn.BCELoss()
        self.bceloss = nn.BCEWithLogitsLoss()
        ####################################
    def forward(self, pred, target):
        size = pred.size(0)
        pred_ = pred.view(size, -1)
        target_ = target.view(size, -1)

        return self.bceloss(pred_, target_)


# Dice 和 bce loss 的选择，根据是什么，如果分割小病灶，是否需要调参？
# https://blog.csdn.net/longshaonihaoa/article/details/111824916
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target):
        ######
        #smooth = 1
        smooth = 1e-5
        ####
        size = pred.size(0)
        ###################################33
        # pred_ = pred.view(size, -1)
        # target_ = target.view(size, -1)
        # intersection = pred_ * target_
        #dice_score = (2 * intersection.sum(1) + smooth) / (pred_.sum(1) + target_.sum(1) + smooth)
        prob = torch.sigmoid(pred)  # ⭐只在 Dice 内
        prob_ = prob.view(size, -1)
        target_ = target.view(size, -1)

        intersection = prob_ * target_
        dice_score = (2 * intersection.sum(1) + smooth) / (prob_.sum(1) + target_.sum(1) + smooth)
        ####################################33

        dice_loss = 1 - dice_score.sum() / size

        return dice_loss


class nDiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(nDiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(),
                                                                                                  target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


class CeDiceLoss(nn.Module):
    def __init__(self, num_classes, loss_weight=[0.4, 0.6]):
        super(CeDiceLoss, self).__init__()
        self.celoss = nn.CrossEntropyLoss()
        self.diceloss = nDiceLoss(num_classes)
        self.loss_weight = loss_weight

    def forward(self, pred, target):
        loss_ce = self.celoss(pred, target[:].long())
        loss_dice = self.diceloss(pred, target, softmax=True)
        loss = self.loss_weight[0] * loss_ce + self.loss_weight[1] * loss_dice
        return loss


class BceDiceLoss(nn.Module):
    ############################
    def __init__(self, wb=1, wd=1):#原本wb=1, wd=1,6号运行时设置wd=0慢慢增加至最大1
    ###############################
        super(BceDiceLoss, self).__init__()
        self.bce = BCELoss()
        self.dice = DiceLoss()
        self.wb = wb
        self.wd = wd

    def forward(self, pred, target):
        bceloss = self.bce(pred, target)
        diceloss = self.dice(pred, target)

        loss = self.wd * diceloss + self.wb * bceloss
        return loss
################33
class IoULoss(nn.Module):
    def __init__(self):
        super(IoULoss, self).__init__()

    def forward(self, pred, target):
        smooth = 1

        pred = torch.sigmoid(pred)   # logits → probability

        pred = pred.view(pred.size(0), -1)
        target = target.view(target.size(0), -1)

        intersection = (pred * target).sum(1)
        union = pred.sum(1) + target.sum(1) - intersection

        iou = (intersection + smooth) / (union + smooth)

        loss = 1 - iou.mean()

        return loss
# class SegLoss(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.bce = BCELoss()
#         self.dice = DiceLoss()
#         self.iou = IoULoss()
#
#     def forward(self, pred, target):
#
#         loss_bce = self.bce(pred, target)
#         loss_dice = self.dice(pred, target)
#         loss_iou = self.iou(pred, target)
#
#         return loss_bce + loss_dice + loss_iou
class SegLoss(nn.Module):
    def __init__(self, lambda_dice=1.0, lambda_iou=1.0):
        super().__init__()
        self.bce = BCELoss()
        self.dice = DiceLoss()
        self.iou = IoULoss()
        self.lambda_dice = lambda_dice
        self.lambda_iou = lambda_iou

    def forward(self, pred, target):
        loss_bce = self.bce(pred, target)
        loss_dice = self.dice(pred, target)
        loss_iou = self.iou(pred, target)

        total_loss = loss_bce + self.lambda_dice * loss_dice + self.lambda_iou * loss_iou

        return total_loss
##################333

class GT_BceDiceLoss(nn.Module):
    def __init__(self, wb=1, wd=1):
        super(GT_BceDiceLoss, self).__init__()
        self.bcedice = BceDiceLoss(wb, wd)

    def forward(self, gt_pre, out, target):
        bcediceloss = self.bcedice(out, target)
        gt_pre5, gt_pre4, gt_pre3, gt_pre2, gt_pre1 = gt_pre
        gt_loss = self.bcedice(gt_pre5, target) * 0.1 + self.bcedice(gt_pre4, target) * 0.2 + self.bcedice(gt_pre3,
                                                                                                           target) * 0.3 + self.bcedice(
            gt_pre2, target) * 0.4 + self.bcedice(gt_pre1, target) * 0.5
        return bcediceloss + gt_loss


class myToTensor:
    def __init__(self):
        pass

    def __call__(self, data):
        image, mask = data
        return torch.tensor(image).permute(2, 0, 1), torch.tensor(mask).permute(2, 0, 1)


class myResize:
    def __init__(self, size_h=256, size_w=256):
        self.size_h = size_h
        self.size_w = size_w

    def __call__(self, data):
        image, mask = data
        #return TF.resize(image, [self.size_h, self.size_w]), TF.resize(mask, [self.size_h, self.size_w])
        img_resized = cv2.resize(image, (self.size_w, self.size_h), interpolation=cv2.INTER_LINEAR)
        mask_resized = cv2.resize(mask, (self.size_w, self.size_h), interpolation=cv2.INTER_NEAREST)
        return img_resized, mask_resized

class myRandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        image, mask = data
        if random.random() < self.p:
            #return TF.hflip(image), TF.hflip(mask)
            return np.fliplr(image).copy(), np.fliplr(mask).copy()
        else:
            return image, mask


class myRandomVerticalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        image, mask = data
        if random.random() < self.p:
            #return TF.vflip(image), TF.vflip(mask)
            return np.flipud(image).copy(), np.flipud(mask).copy()
        else:
            return image, mask

#######################################3
# class myRandomRotation:
#     def __init__(self, p=0.5, degree=[0, 360]):
#         self.angle = random.uniform(degree[0], degree[1])
#         self.p = p
#
#     def __call__(self, data):
#         image, mask = data
#         if random.random() < self.p:
#             return TF.rotate(image, self.angle), TF.rotate(mask, self.angle)
#         else:
#             return image, mask

class myRandomRotation:
    def __init__(self, p=0.5, degree=[0, 360]):
        self.p = p
        self.degree = degree

    def __call__(self, data):
        image, mask = data

        if random.random() < self.p:
            angle = random.uniform(self.degree[0], self.degree[1])
            img_rot = rotate(image, angle, reshape=False, mode='reflect')
            mask_rot = rotate(mask, angle, reshape=False, order=0, mode='reflect')
            return img_rot, mask_rot

        return image, mask


# --------------------------- 高级增强 ---------------------------
class myRandomResizedCrop:
    def __init__(self, p=0.5, size=(224,224), scale=(0.8,1.0), ratio=(0.75,1.33)):
        self.p = p
        self.size = size
        self.scale = scale
        self.ratio = ratio

    def __call__(self, data):
        img, mask = data
        if random.random() < self.p:
            h, w = img.shape[:2]
            for _ in range(10):
                target_area = random.uniform(*self.scale) * h * w
                aspect_ratio = random.uniform(*self.ratio)
                new_w = int(round(np.sqrt(target_area * aspect_ratio)))
                new_h = int(round(np.sqrt(target_area / aspect_ratio)))
                if new_w <= w and new_h <= h:
                    x1 = random.randint(0, w - new_w)
                    y1 = random.randint(0, h - new_h)
                    img_crop = img[y1:y1+new_h, x1:x1+new_w]
                    mask_crop = mask[y1:y1+new_h, x1:x1+new_w]
                    img_resized = cv2.resize(img_crop, (self.size[1], self.size[0]), interpolation=cv2.INTER_LINEAR)
                    mask_resized = cv2.resize(mask_crop, (self.size[1], self.size[0]), interpolation=cv2.INTER_NEAREST)
                    return img_resized, mask_resized
        return cv2.resize(img, (self.size[1], self.size[0]), interpolation=cv2.INTER_LINEAR), \
               cv2.resize(mask, (self.size[1], self.size[0]), interpolation=cv2.INTER_NEAREST)

class myRandomAffine:
    def __init__(self, p=0.5, degrees=15, translate=0.0, scale=0.0, shear=0.0):
        self.p = p
        self.degrees = degrees if isinstance(degrees,(list,tuple)) else (-degrees,degrees)
        self.translate = translate
        self.scale = scale if isinstance(scale,(list,tuple)) else (1.0-scale, 1.0+scale)
        self.shear = shear if isinstance(shear,(list,tuple)) else (-shear,shear)

    def __call__(self, data):
        img, mask = data
        if random.random() < self.p:
            h, w = img.shape[:2]
            angle = random.uniform(*self.degrees)
            scale = random.uniform(*self.scale)
            max_dx = self.translate * w
            max_dy = self.translate * h
            tx = random.uniform(-max_dx, max_dx)
            ty = random.uniform(-max_dy, max_dy)
            shear = random.uniform(*self.shear)

            # Affine matrix
            center = (w/2, h/2)
            M = cv2.getRotationMatrix2D(center, angle, scale)
            M[0,2] += tx
            M[1,2] += ty

            img_aff = cv2.warpAffine(img, M, (w,h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            mask_aff = cv2.warpAffine(mask, M, (w,h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)
            return img_aff, mask_aff
        return img, mask


class myColorJitter:
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, p=0.5):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.p = p

    def __call__(self, data):
        img, mask = data
        if random.random() < self.p:
            img = img.astype(np.float32)
            img = img * (1 + random.uniform(-self.contrast, self.contrast))
            img = img + random.uniform(-self.brightness*255, self.brightness*255)
            img = np.clip(img,0,255)
        return img, mask


class myRandomGamma:
    def __init__(self, gamma_range=(0.8,1.2), p=0.5):
        self.gamma_range = gamma_range
        self.p = p
    def __call__(self, data):
        img, mask = data
        if random.random() < self.p:
            gamma = random.uniform(*self.gamma_range)
            img = ((img/255.)**gamma)*255.
        return img, mask


class myGaussianNoise:
    def __init__(self, mean=0., std=5., p=0.3):
        self.mean = mean
        self.std = std
        self.p = p
    def __call__(self, data):
        img, mask = data
        if random.random() < self.p:
            noise = np.random.normal(self.mean, self.std, img.shape)
            img = img.astype(np.float32) + noise
            img = np.clip(img,0,255)
        return img, mask


class myGaussianBlur:
    def __init__(self, radius=(0.1,2.0), p=0.3):
        self.radius = radius
        self.p = p
    def __call__(self, data):
        img, mask = data
        if random.random() < self.p:
            r = random.uniform(*self.radius)
            ksize = int(max(1, round(r*2)*2+1))
            img = cv2.GaussianBlur(img,(ksize,ksize),0)
        return img, mask

#随机遮挡图像区域（类似 DropBlock），强制模型学习上下文
class myCutout:
    def __init__(self, p=0.5, num_holes=1, max_h_size=32, max_w_size=32):
        self.p = p
        self.num_holes = num_holes
        self.max_h_size = max_h_size
        self.max_w_size = max_w_size
    def __call__(self, data):
        img, mask = data
        if random.random() < self.p:
            h, w = img.shape[:2]
            for _ in range(self.num_holes):
                hole_h = random.randint(1, self.max_h_size)
                hole_w = random.randint(1, self.max_w_size)
                y = random.randint(0, h-1)
                x = random.randint(0, w-1)
                y1 = max(0, y-hole_h//2)
                y2 = min(h, y+hole_h//2)
                x1 = max(0, x-hole_w//2)
                x2 = min(w, x+hole_w//2)
                img[y1:y2, x1:x2, :] = 0
        return img, mask

class myRandomErasing:
    def __init__(self, p=0.3, scale=(0.02,0.2), ratio=(0.3,3.3)):
        self.p = p
        self.scale = scale
        self.ratio = ratio

    def __call__(self, data):

        image, mask = data

        if random.random() < self.p:

            h, w, c = image.shape
            area = h * w

            target_area = random.uniform(*self.scale) * area
            aspect_ratio = random.uniform(*self.ratio)

            h_erase = int(round(np.sqrt(target_area * aspect_ratio)))
            w_erase = int(round(np.sqrt(target_area / aspect_ratio)))

            if h_erase < h and w_erase < w:

                x1 = random.randint(0, h - h_erase)
                y1 = random.randint(0, w - w_erase)

                image[x1:x1+h_erase, y1:y1+w_erase, :] = 0

        return image, mask
#########################################3
class myNormalize:
    def __init__(self, data_name, train=True):
        if data_name == 'isic18':
            if train:
                self.mean = 157.561
                self.std = 26.706
            else:
                self.mean = 149.034
                self.std = 32.022
        elif data_name == 'isic17':
            if train:
                self.mean = 159.922
                self.std = 28.871
            else:
                self.mean = 148.429
                self.std = 25.748
        elif data_name == 'isic18_82':
            if train:
                self.mean = 156.2899
                self.std = 26.5457
            else:
                self.mean = 149.8485
                self.std = 35.3346
        elif data_name == 'polyp':
            if train:
                self.mean = 86.17
                self.std = 69.08
            else:
                self.mean = 86.17
                self.std = 69.08
        elif data_name == 'gim':
            if train:
                self.mean = 87.84
                self.std = 55.37
            else:
                self.mean = 85.27
                self.std = 54.75
        elif data_name == 'isic_all':
            if train:
                self.mean = 158.6
                self.std = 44.92
            else:
                self.mean = 156.2899
                self.std = 26.5457

    # TODO 注意 息肉数据集合肠化数据集的区别
    # 只是对image 做了归一化和标准化，后来还 TM * 255 窒息
    def __call__(self, data):
        img, msk = data
        img_normalized = (img - self.mean) / self.std
        img_normalized = ((img_normalized - np.min(img_normalized))
                          / (np.max(img_normalized) - np.min(img_normalized))) * 255.
        return img_normalized, msk

from thop import profile  ## 导入thop模块


def cal_params_flops(model, size, logger):
    input = torch.randn(1, 3, size, size).cuda()
    flops, params = profile(model, inputs=(input,))
    print('flops', flops / 1e9)  ## 打印计算量
    print('params', params / 1e6)  ## 打印参数量

    total = sum(p.numel() for p in model.parameters())
    print("Total params: %.2fM" % (total / 1e6))
    logger.info(f'flops: {flops / 1e9}, params: {params / 1e6}, Total params: : {total / 1e6:.4f}')


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum() == 0:
        return 1, 0
    else:
        return 0, 0


def test_single_volume(image, label, net, classes, patch_size=[256, 256],
                       test_save_path=None, case=None, z_spacing=1, val_or_test=False):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
            net.eval()
            with torch.no_grad():
                outputs = net(input)
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[ind] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))

    if test_save_path is not None and val_or_test is True:
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        img_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))
        sitk.WriteImage(prd_itk, test_save_path + '/' + case + "_pred.nii.gz")
        sitk.WriteImage(img_itk, test_save_path + '/' + case + "_img.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + '/' + case + "_gt.nii.gz")
        # cv2.imwrite(test_save_path + '/'+case + '.png', prediction*255)
    return metric_list