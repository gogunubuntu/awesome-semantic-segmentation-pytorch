from __future__ import print_function

import os
import sys
from time import sleep, time

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from core.data.dataloader import get_segmentation_dataset
from core.models.model_zoo import get_segmentation_model
from core.utils.score import SegmentationMetric
from core.utils.visualize import get_color_pallete
from core.utils.logger import setup_logger
from core.utils.distributed import (
    synchronize,
    get_rank,
    make_data_sampler,
    make_batch_data_sampler,
)
from core.utils.parser import parse_args


class Evaluator(object):
    def __init__(self, args):
        self.args = args
        self.img_dir = args.eval_img_dir
        self.device = torch.device(args.device)

        # image transform
        input_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        # dataset and dataloader
        val_dataset = get_segmentation_dataset(
            args.dataset, split="val", mode="testval", transform=input_transform
        )
        val_sampler = make_data_sampler(val_dataset, False, args.distributed)
        val_batch_sampler = make_batch_data_sampler(val_sampler, images_per_batch=1)
        self.val_loader = data.DataLoader(
            dataset=val_dataset,
            batch_sampler=val_batch_sampler,
            num_workers=args.workers,
            pin_memory=True,
        )

        # create network
        BatchNorm2d = nn.SyncBatchNorm if args.distributed else nn.BatchNorm2d
        self.model = get_segmentation_model(
            model=args.model,
            dataset=args.dataset,
            backbone=args.backbone,
            aux=args.aux,
            pretrained=True,
            pretrained_base=False,
            norm_layer=BatchNorm2d,
        ).to(self.device)
        if args.distributed:
            self.model = nn.parallel.DistributedDataParallel(
                self.model, device_ids=[args.local_rank], output_device=args.local_rank
            )
        self.model.to(self.device)

        self.metric = SegmentationMetric(val_dataset.num_class)

    def eval(self):
        self.metric.reset()
        self.model.eval()
        if self.args.distributed:
            model = self.model.module
        else:
            model = self.model
        file_names = os.listdir(self.img_dir)

        for file_name in file_names:
            img_full_path = os.path.join(self.img_dir, file_name)
            np_image = cv.imread(img_full_path)
            np_image = cv.resize(np_image, dsize=(640, 360))
            np_image_tp = np_image.transpose([2, 0, 1])
            np_image_tp = np.array([np_image_tp])

            image = torch.from_numpy(np_image_tp).to(self.device).to(torch.float)
            image = image / 255
            with torch.no_grad():
                tic = time()
                outputs = model(image)
            if self.args.save_pred:
                pred = torch.argmax(outputs[0], 1)
                toc = time()
                print(f"{1 / (toc - tic)}hz")
                pred = pred.cpu().data.numpy()
                predict = pred.squeeze(0)
                mask = get_color_pallete(predict, self.args.dataset)
                mask.save(os.path.join(outdir, os.path.splitext(file_name)[0] + ".png"))
                saveimg = cv.imread(
                    os.path.join(outdir, os.path.splitext(file_name)[0] + ".png")
                )
                alpha = 0.7
                saveimg = cv.addWeighted(saveimg, alpha, np_image, (1 - alpha), 0)
                cv.imwrite(
                    os.path.join(
                        outdir, "norm01" + os.path.splitext(file_name)[0] + ".png"
                    ),
                    np.uint8(saveimg),
                )


if __name__ == "__main__":
    args = parse_args()
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1
    if not args.no_cuda and torch.cuda.is_available():
        cudnn.benchmark = True
        args.device = "cuda"
    else:
        args.distributed = False
        args.device = "cpu"
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    # TODO: optim code
    args.save_pred = True
    if args.save_pred:
        outdir = "../runs/pred_pic/{}_{}_{}".format(
            args.model, args.backbone, args.dataset
        )
        if not os.path.exists(outdir):
            os.makedirs(outdir)

    logger = setup_logger(
        "semantic_segmentation",
        args.log_dir,
        get_rank(),
        filename="{}_{}_{}_log.txt".format(args.model, args.backbone, args.dataset),
        mode="a+",
    )

    evaluator = Evaluator(args)
    evaluator.eval()

    torch.cuda.empty_cache()
