import argparse
import time
import os
import importlib
import datetime
from tqdm import tqdm

import numpy as np
import torch
import transforms as T
from my_dataset import VOCSegmentation
from train_utils import train_one_epoch, evaluate, create_lr_scheduler


from model.swin_deeplab import SwinDeepLab


def parse_arge():
    parser = argparse.ArgumentParser(description="pytorch transdeeplab training")
    parser.add_argument('--data_path', type=str,
                        default=r"E:\IRSA\front_detection\dataset\data_raw",
                        help='root dir for data')
    # parser.add_argument('--data_path', type=str,
    #                     default=r"E:\IRSA\Ice_Shelf\DLCode\deep-learning-for-image-processing-master\pytorch_segmentation\unet\snowmelt_dataset_unnorm",
    #                     help='root dir for data')
    parser.add_argument('--config_file', type=str,
                        default='swin_224_7_4level', help='config file name w/o suffix')
    # background not included
    parser.add_argument('--num_classes', type=int,
                        default=2, help='output channel of network')
    parser.add_argument('--output_dir', type=str, default='.', help='output dir')
    parser.add_argument('--epochs', type=int,
                        default=150, help='maximum epoch number to train')
    parser.add_argument('--batch_size', type=int,
                        default=64, help='batch_size per gpu')
    parser.add_argument('--img_size', type=int,
                        default=224, help='input patch size of network input')
    parser.add_argument("--device", default="cuda", help="training device")
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--print-freq', default=30, type=int, help='print frequency')
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument("--amp", default=False, type=bool,
                        help="Use torch.cuda.amp for mixed precision training")
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='start epoch')

    args = parser.parse_args()
    return args
# if args.dataset == "Synapse":
#     args.root_path = os.path.join(args.root_path, "train_npz")



class SegmentationPresetTrain:
    def __init__(self, base_size, crop_size, hflip_prob=0.5, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        min_size = int(0.5 * base_size)
        max_size = int(2.0 * base_size)

        # trans = [T.RandomResize(min_size, max_size)]
        trans = []
        if hflip_prob > 0:
            trans.append(T.RandomHorizontalFlip(hflip_prob))
        trans.extend([
            T.RandomCrop(crop_size),
            T.ToTensor(),
            # T.Normalize(mean=mean, std=std),
        ])
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        return self.transforms(img, target)


class SegmentationPresetEval:
    def __init__(self, base_size, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = T.Compose([
            T.RandomResize(base_size, base_size),
            T.ToTensor(),
            # T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)


def get_transform(train, size):
    base_size = size
    crop_size = size

    return SegmentationPresetTrain(base_size, crop_size) if train else SegmentationPresetEval(base_size)

def main(args):


    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    # num_classes = categories + background
    num_classes = args.num_classes + 1
    batch_size = args.batch_size
    base_lr = args.lr
    if batch_size != 24 and batch_size % 6 == 0:
        base_lr *= batch_size / 24

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    print(f"The model is traing on {device}")

    model_config = importlib.import_module(f'model.configs.{args.config_file}')
    model = SwinDeepLab(
        model_config.EncoderConfig,
        model_config.ASPPConfig,
        model_config.DecoderConfig
    )
    model.to(device)

    if model_config.EncoderConfig.encoder_name == 'swin' and model_config.EncoderConfig.load_pretrained:
        model.encoder.load_from('./pretrained_ckpt/swin_tiny_patch4_window7_224.pth')
    if model_config.ASPPConfig.aspp_name == 'swin' and model_config.ASPPConfig.load_pretrained:
        model.aspp.load_from('./pretrained_ckpt/swin_tiny_patch4_window7_224.pth')
    if model_config.DecoderConfig.decoder_name == 'swin' and model_config.DecoderConfig.load_pretrained and not model_config.DecoderConfig.extended_load:
        model.decoder.load_from('./pretrained_ckpt/swin_tiny_patch4_window7_224.pth')
    if model_config.DecoderConfig.decoder_name == 'swin' and model_config.DecoderConfig.load_pretrained and model_config.DecoderConfig.extended_load:
        model.decoder.load_from_extended('./pretrained_ckpt/swin_tiny_patch4_window7_224.pth')



    # 用来保存训练以及验证过程中信息
    results_file = "./results_record/results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    # VOCdevkit -> VOC2012 -> ImageSets -> Segmentation -> train.txt
    train_dataset = VOCSegmentation(args.data_path,
                                    year="2012",
                                    transforms=get_transform(train=True, size=224),
                                    txt_name="train.txt")

    # VOCdevkit -> VOC2012 -> ImageSets -> Segmentation -> val.txt
    val_dataset = VOCSegmentation(args.data_path,
                                  year="2012",
                                  transforms=get_transform(train=False, size=224),
                                  txt_name="val.txt")

    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               num_workers=num_workers,
                                               shuffle=True,
                                               pin_memory=True,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             num_workers=num_workers,
                                             pin_memory=True,
                                             collate_fn=val_dataset.collate_fn)



    optimizer = torch.optim.SGD(model.parameters(), lr=base_lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler() if args.amp else None


    # 创建学习率更新策略，这里是每个step更新一次(不是每个epoch)
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=True)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1

    model_info = f"Change num_classes by Modifying the Config file, args.num_classes: {num_classes}\n" \
               f"batch_size: {batch_size}\n" \
               f"base lr: {base_lr}\n" \
               f"dataset: {args.data_path}\n\n"
    print(model_info)

    with open(results_file, "a") as f:
        # record model's num_classes, batch_size, lr and dataset
        f.write(model_info)

    start_time = time.time()
    for epoch in tqdm(range(args.start_epoch, args.epochs)):
        mean_loss, lr = train_one_epoch(model, optimizer, train_loader, device, epoch,
                                        lr_scheduler=lr_scheduler, num_classes=num_classes, print_freq=args.print_freq, scaler=scaler)

        confmat = evaluate(model, val_loader, device=device, num_classes=num_classes)
        val_info = str(confmat)
        print(val_info)
        # write into txt
        with open(results_file, "a") as f:
            # 记录每个epoch对应的train_loss、lr以及验证集各指标
            train_info = f"[epoch: {epoch}]\n" \
                         f"train_loss: {mean_loss:.4f}\n" \
                         f"lr: {lr:.6f}\n"
            f.write(train_info + val_info + "\n\n")

        save_file = {"model": model.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "lr_scheduler": lr_scheduler.state_dict(),
                     "epoch": epoch,
                     "args": args}
        if args.amp:
            save_file["scaler"] = scaler.state_dict()
        torch.save(save_file, os.path.join(weights_path, f"Swin_{epoch}.pth"))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("training time {}".format(total_time_str))

if __name__ == "__main__":
    args = parse_arge()
    weights_path = "./save_weights/weights_{}".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    if not os.path.exists(weights_path):
        os.mkdir(weights_path)
    main(args)

