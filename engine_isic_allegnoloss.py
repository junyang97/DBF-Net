import numpy as np
from tqdm import tqdm
import torch
from sklearn.metrics import confusion_matrix
#from utils_isic_all import save_imgs
from fusion.utils_polyp_loss_5 import save_imgs


def parse_model_output(out):
    """
    统一兼容：
    1) Tensor
    2) tuple(pred, ...)
    3) dict(final_pred, coarse_pred, edge_map)

    恢复到“无 edge loss 监督”后，这里仍保留兼容，
    但训练/验证/测试只使用 pred。
    """
    pred = None

    if isinstance(out, dict):
        pred = out["final_pred"]
    elif isinstance(out, tuple):
        pred = out[0]
    else:
        pred = out

    return pred


def train_one_epoch(train_loader,
                    model,
                    criterion,
                    optimizer,
                    scheduler,
                    epoch,
                    step,
                    logger,
                    config,
                    writer,
                    device):
    """
    train model for one epoch
    """
    model.train()
    loss_list = []

    # 保留你原来的 Dice 权重接口
    wd = 1.0
    if hasattr(criterion, 'wd'):
        criterion.wd = wd
    if hasattr(criterion, 'wb'):
        criterion.wb = 1.0

    for iter, data in enumerate(train_loader):
        step += 1
        optimizer.zero_grad()

        images, targets = data
        images = images.to(device, non_blocking=True).float()
        targets = targets.to(device, non_blocking=True).float()

        out = model(images)
        pred = parse_model_output(out)

        loss = criterion(pred, targets)

        loss.backward()
        optimizer.step()

        loss_list.append(loss.item())
        now_lr = optimizer.state_dict()['param_groups'][0]['lr']

        if writer is not None:
            writer.add_scalar('train/loss', loss.item(), global_step=step)

        if iter % config.print_interval == 0:
            print(
                f"batch {iter} pred min: {pred.min().item():.4f}, "
                f"max: {pred.max().item():.4f}, mean: {pred.mean().item():.4f}"
            )
            log_info = (
                f"[Train] epoch {epoch} iter {iter} | "
                f"loss: {np.mean(loss_list):.4f} | "
                f"lr: {now_lr:.6f}"
            )
            print(log_info)
            logger.info(log_info)

    scheduler.step()

    epoch_loss = np.mean(loss_list)
    if writer is not None:
        writer.add_scalar('train/epoch_loss', epoch_loss, global_step=epoch)

    return step


def val_one_epoch(test_loader,
                  model,
                  criterion,
                  epoch,
                  logger,
                  config,
                  writer=None,
                  val_data_name=None):
    """
    validation for one epoch
    """
    model.eval()
    preds = []
    gts = []
    loss_list = []

    f1_or_dsc = 0
    miou = 0

    with torch.no_grad():
        for iter, data in enumerate(tqdm(test_loader)):
            img, msk = data
            img = img.cuda(non_blocking=True).float()
            msk = msk.cuda(non_blocking=True).float()

            out = model(img)
            pred = parse_model_output(out)

            loss = criterion(pred, msk)
            loss_list.append(loss.item())

            gts.append(msk.squeeze(1).cpu().detach().numpy())

            if pred.shape[1] > 1:  # 多类别
                pred_class = torch.argmax(pred, dim=1)
            else:  # 二分类
                pred_class = (torch.sigmoid(pred) > config.threshold).float().squeeze(1)

            preds.append(pred_class.cpu().numpy())

    if epoch % config.val_interval == 0:
        gts = np.concatenate([g.reshape(-1) for g in gts])
        preds = np.concatenate([p.reshape(-1) for p in preds])

        y_pre = np.where(preds >= config.threshold, 1, 0)
        y_true = np.where(gts >= 0.5, 1, 0)

        confusion = confusion_matrix(y_true, y_pre)
        TN, FP, FN, TP = confusion[0, 0], confusion[0, 1], confusion[1, 0], confusion[1, 1]

        accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
        sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
        specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
        f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
        miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0

        if val_data_name is not None:
            log_info = f'val_datasets_name: {val_data_name}'
            print(log_info)
            logger.info(log_info)

        log_info = (
            f' val epoch: {epoch}, loss: {np.mean(loss_list):.4f}, '
            f'miou: {miou}, f1_or_dsc: {f1_or_dsc}, accuracy: {accuracy}, '
            f'specificity: {specificity}, sensitivity: {sensitivity}, confusion_matrix: {confusion}'
        )
        print(log_info)
        logger.info(log_info)

    else:
        log_info = f' val epoch: {epoch}, loss: {np.mean(loss_list):.4f}'
        print(log_info)
        logger.info(log_info)

    if writer is not None:
        writer.add_scalar('val/loss', np.mean(loss_list), global_step=epoch)
        writer.add_scalar('val/dice', f1_or_dsc, global_step=epoch)
        writer.add_scalar('val/miou', miou, global_step=epoch)

    return {
        'loss': np.mean(loss_list),
        'f1_or_dsc': f1_or_dsc,
        'miou': miou
    }


def test_one_epoch(test_loader,
                   model,
                   criterion,
                   logger,
                   config,
                   test_data_name=None):
    """
    test best model
    """
    model.eval()
    preds = []
    gts = []
    loss_list = []

    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            img, msk = data
            img = img.cuda(non_blocking=True).float()
            msk = msk.cuda(non_blocking=True).float()

            out = model(img)
            pred = parse_model_output(out)

            loss = criterion(pred, msk)
            loss_list.append(loss.item())

            msk_np = msk.squeeze(1).cpu().detach().numpy()
            gts.append(msk_np)
            ##########genval——one_epochduojiayigeerfenleiduofenleidequfen
            if pred.shape[1] > 1:  # 多类别
                pred=pred
            else:  # 二分类
                pred = torch.sigmoid(pred)
            pred_np = pred.squeeze(1).cpu().detach().numpy()
            preds.append(pred_np)

            if i % config.save_interval == 0:
                save_imgs(
                    img,
                    msk_np,
                    pred_np,
                    i,
                    config.work_dir + 'outputs/',
                    config.datasets,
                    config.threshold,
                    test_data_name=test_data_name
                )

        preds = np.array(preds).reshape(-1)
        gts = np.array(gts).reshape(-1)

        y_pre = np.where(preds >= config.threshold, 1, 0)
        y_true = np.where(gts >= 0.5, 1, 0)

        confusion = confusion_matrix(y_true, y_pre)
        TN, FP, FN, TP = confusion[0, 0], confusion[0, 1], confusion[1, 0], confusion[1, 1]

        accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
        sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
        specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
        f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
        miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0

        if test_data_name is not None:
            log_info = f'test_datasets_name: {test_data_name}'
            print(log_info)
            logger.info(log_info)

        log_info = (
            f'test of best model, loss: {np.mean(loss_list):.4f}, '
            f'miou: {miou}, f1_or_dsc: {f1_or_dsc}, accuracy: {accuracy}, '
            f'specificity: {specificity}, sensitivity: {sensitivity}, confusion_matrix: {confusion}'
        )
        print(log_info)
        logger.info(log_info)

    return np.mean(loss_list)