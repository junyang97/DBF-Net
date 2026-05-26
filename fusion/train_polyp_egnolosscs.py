import os
import sys
import warnings
import numpy as np
import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from datasets.dataset_isic_all_loss import NPY_datasets, Polyp_datasets
from fusion.models.fusion_model_3drop import build_fusion_model
from engine_isic_allegnoloss import *
#from fusion.utils_polyp_loss import *
from fusion.utils_polyp_loss_5 import *
from fusion.utils_polyp_losscompare import cal_params,cal_flops,benchmark_inference
from fusion.configs.config_setting_polyp import setting_config

warnings.filterwarnings("ignore")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(config):
    print('#----------Creating logger----------#')
    sys.path.append(config.work_dir + '/')
    log_dir = os.path.join(config.work_dir, 'log')
    checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')
    resume_model = os.path.join(checkpoint_dir, 'latest.pth')
    outputs = os.path.join(config.work_dir, 'outputs')

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(outputs, exist_ok=True)

    global logger
    logger = get_logger('train', log_dir)
    global writer
    writer = SummaryWriter(config.work_dir + 'summary')

    log_config_info(config, logger)

    print('#----------GPU init----------#')
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
    set_seed(config.seed)
    torch.cuda.empty_cache()

    print('#----------Preparing dataset----------#')
    train_dataset = Polyp_datasets(config.data_path, config, train=True)
    #train_dataset = NPY_datasets(config.data_path, config, train=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=config.num_workers
    )

    val_loader_dict = {}
    eval_datasets = ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']
    for dataset in eval_datasets:
        val_dataset = Polyp_datasets(config.data_path, config, train=False, test_dataset=dataset)
        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            pin_memory=True,
            num_workers=config.num_workers,
            drop_last=True
        )
        val_loader_dict[dataset] = val_loader
    # val_dataset = NPY_datasets(config.data_path, config, train=False)
    # val_loader = DataLoader(
    #     val_dataset,
    #     batch_size=1,
    #     shuffle=False,
    #     pin_memory=True,
    #     num_workers=config.num_workers,
    #     drop_last=True
    # )

    print('#----------Prepareing Model----------#')
    model_cfg = config.model_config
    if config.network == 'fusion_egalayerall':#dropega_adaptive
        # model = build_fusion_model(
        #     load_pretrained=True,
        #     npz_path=config.transunet_pretrained_path,
        #     ega_input_mode="adaptive"
        # )
        model = build_fusion_model(
                load_pretrained=True,
                npz_path=config.transunet_pretrained_path,
                ega_input_mode="adaptive",
                # branch_mode="mamba"
                use_cross_guided_fusion=True,
                use_two_stage_decoder=True,
                use_ega_refine=True

        )
    else:
        raise Exception('network in not right!')

    model = model.to(device)
    ###############################
    param_info = cal_params(model, logger)
    # flops_info = cal_flops(model, input_size=(1, 3, 256, 256), logger=logger)
    from calflops import calculate_flops
    input_shape = (1, 3, 256, 256)  # 请根据你的实际图像分辨率修改，比如 (1,3,256,256)

    flops, macs, params = calculate_flops(
        model=model,
        input_shape=input_shape,
        print_results=True,  # 是否打印详细的层级表格
        print_detailed=False  # 设置为 False 防止打印内容过长
    )

    print(f"Total FLOPs: {flops}")
    print(f"Total MACs: {macs}")
    print(f"Total Params: {params}")

    bench_info = benchmark_inference(
        model,
        input_size=(1, 3, 256, 256),
        warmup=50,
        runs=200,
        logger=logger
    )

    print("\n#----------Model Complexity Report----------#")
    print(f"Total Params (M): {param_info['total_params_m']:.4f}")
    print(f"Trainable Params (M): {param_info['trainable_params_m']:.4f}")
    # print(f"MACs (G): {flops_info['macs_g']:.4f}")
    # print(f"FLOPs (G): {flops_info['flops_est_g']:.4f}")
    print(f"GPU Memory Usage (MB): {bench_info['peak_memory_mb']:.2f}")
    print(f"Latency (ms/img): {bench_info['latency_ms']:.4f}")
    print(f"FPS: {bench_info['fps']:.2f}")

    logger.info("#----------Model Complexity Report----------#")
    logger.info(f"Total Params (M): {param_info['total_params_m']:.4f}")
    logger.info(f"Trainable Params (M): {param_info['trainable_params_m']:.4f}")
    # logger.info(f"MACs (G): {flops_info['macs_g']:.4f}")
    # logger.info(f"FLOPs (G): {flops_info['flops_est_g']:.4f}")
    logger.info(f"GPU Memory Usage (MB): {bench_info['peak_memory_mb']:.2f}")
    logger.info(f"Latency (ms/img): {bench_info['latency_ms']:.4f}")
    logger.info(f"FPS: {bench_info['fps']:.2f}")

    ###############################
    print('#----------Prepareing loss, opt, sch and amp----------#')
    criterion = config.criterion
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)

    print('#----------Set other params----------#')
    start_epoch = 1

    best_loss = float('inf')
    best_loss_epoch = 1

    best_dice = -float('inf')
    best_dice_epoch = 1

    best_miou = -float('inf')
    best_miou_epoch = 1

    last_loss = None
    last_dice = None
    last_miou = None

    if os.path.exists(resume_model):
        print('#----------Resume Model and Other params----------#')
        checkpoint = torch.load(resume_model, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        saved_epoch = checkpoint['epoch']
        start_epoch = saved_epoch + 1

        best_loss = checkpoint.get('best_loss', float('inf'))
        best_loss_epoch = checkpoint.get('best_loss_epoch', 1)

        best_dice = checkpoint.get('best_dice', -float('inf'))
        best_dice_epoch = checkpoint.get('best_dice_epoch', 1)

        best_miou = checkpoint.get('best_miou', -float('inf'))
        best_miou_epoch = checkpoint.get('best_miou_epoch', 1)

        last_loss = checkpoint.get('loss', None)
        last_dice = checkpoint.get('dice', None)
        last_miou = checkpoint.get('miou', None)

        log_info = (
            f"resuming model from {resume_model}. "
            f"resume_epoch: {saved_epoch}, "
            f"best_loss: {best_loss:.4f} (epoch {best_loss_epoch}), "
            f"best_dice: {best_dice:.4f} (epoch {best_dice_epoch}), "
            f"best_miou: {best_miou:.4f} (epoch {best_miou_epoch}), "
            f"last_loss: {last_loss}, "
            f"last_dice: {last_dice}, last_miou: {last_miou}"
        )
        logger.info(log_info)

    step = 0
    print('#----------Training----------#')
    for epoch in range(start_epoch, config.epochs + 1):
        torch.cuda.empty_cache()

        step = train_one_epoch(
            train_loader,
            model,
            criterion,
            optimizer,
            scheduler,
            epoch,
            step,
            logger,
            config,
            writer,
            device=device
        )

        loss_all = []
        miou_all = []
        dice_all = []

        for name in eval_datasets:
            val_loader_t = val_loader_dict[name]
            val_metrics = val_one_epoch(
                val_loader_t,
                model,
                criterion,
                epoch,
                logger,
                config,
                val_data_name=name
            )
        # val_metrics = val_one_epoch(
        #     val_loader,
        #     model,
        #     criterion,
        #     epoch,
        #     logger,
        #     config,
        # )

            loss_all.append(val_metrics['loss'])
            miou_all.append(val_metrics['miou'])
            dice_all.append(val_metrics['f1_or_dsc'])
        # loss_all.append(val_metrics['loss'])
        # miou_all.append(val_metrics['miou'])
        # dice_all.append(val_metrics['f1_or_dsc'])
        loss = float(np.mean(loss_all))
        miou = float(np.mean(miou_all))
        dice = float(np.mean(dice_all))

        if loss < best_loss:
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best_loss.pth'))
            best_loss = loss
            best_loss_epoch = epoch
            logger.info(f'[BEST-LOSS] Update at epoch {epoch}: best_loss={best_loss:.4f}')

        if dice > best_dice:
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best_dice.pth'))
            best_dice = dice
            best_dice_epoch = epoch
            logger.info(f'[BEST-DICE] Update at epoch {epoch}: best_dice={best_dice:.4f}')

        if miou > best_miou:
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best_miou.pth'))
            best_miou = miou
            best_miou_epoch = epoch
            logger.info(f'[BEST-MIOU] Update at epoch {epoch}: best_miou={best_miou:.4f}')

        torch.save(
            {
                'epoch': epoch,

                'best_loss': best_loss,
                'best_loss_epoch': best_loss_epoch,

                'best_dice': best_dice,
                'best_dice_epoch': best_dice_epoch,

                'best_miou': best_miou,
                'best_miou_epoch': best_miou_epoch,

                'loss': loss,
                'miou': miou,
                'dice': dice,

                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            },
            os.path.join(checkpoint_dir, 'latest.pth')
        )

        print(
            f"Epoch {epoch} summary: "
            f"loss={loss:.4f}, miou={miou:.4f}, dice={dice:.4f} | "
            f"best_loss={best_loss:.4f}(ep{best_loss_epoch}), "
            f"best_dice={best_dice:.4f}(ep{best_dice_epoch}), "
            f"best_miou={best_miou:.4f}(ep{best_miou_epoch})"
        )

    print('#----------Testing----------#')

    best_model_infos = [
        ('best_loss.pth', 'best_loss', best_loss_epoch, best_loss),
        ('best_dice.pth', 'best_dice', best_dice_epoch, best_dice),
        ('best_miou.pth', 'best_miou', best_miou_epoch, best_miou),
    ]

    for ckpt_name, metric_name, metric_epoch, metric_value in best_model_infos:
        ckpt_path = os.path.join(checkpoint_dir, ckpt_name)
        if not os.path.exists(ckpt_path):
            logger.info(f'{ckpt_name} not found, skip testing.')
            continue

        print(f'#----------Testing {ckpt_name} ----------#')
        logger.info(f'Loading {ckpt_name} for testing...')

        best_weight = torch.load(ckpt_path, map_location=torch.device('cpu'))
        model.load_state_dict(best_weight)

        for name in eval_datasets:
            val_loader_t = val_loader_dict[name]
            _ = test_one_epoch(
                val_loader_t,
                model,
                criterion,
                logger,
                config,
                test_data_name=f'{name} [{metric_name}]'
            )
        # _ = test_one_epoch(
        #     val_loader,
        #     model,
        #     criterion,
        #     logger,
        #     config,
        # )

        new_name = f'{metric_name}-epoch{metric_epoch}-{metric_name}{metric_value:.4f}.pth'
        new_path = os.path.join(checkpoint_dir, new_name)

        if not os.path.exists(new_path):
            os.rename(ckpt_path, new_path)
        else:
            logger.info(f'{new_path} already exists, keep original {ckpt_name}.')

    logger.info('Training and testing finished.')


if __name__ == '__main__':
    config = setting_config
    main(config)