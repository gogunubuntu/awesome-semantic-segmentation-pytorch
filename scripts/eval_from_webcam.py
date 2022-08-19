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

from skimage import io
from skimage import color
from skimage import segmentation

from torchvision import transforms
from core.data.dataloader import get_segmentation_dataset
from core.models.model_zoo import get_segmentation_model
from core.utils.score import SegmentationMetric
from core.utils.visualize import get_color_pallete, show_colorful_images
from core.utils.logger import setup_logger
from core.utils.distributed import (
    synchronize,
    get_rank,
    make_data_sampler,
    make_batch_data_sampler,
)
from train import parse_args

ret = 1.0635049780625927
mtx = np.array(
    [
        [454.14992519, 0.0, 330.13040917],
        [0.0, 453.86658378, 273.88395007],
        [0.0, 0.0, 1.0],
    ]
)
dist = np.array([[-0.46692931, 0.29388721, -0.01096731, 0.00328268, -0.10274585]])
newcameramtx = np.array(
    [
        [453.44033813, 0.0, 329.61458425],
        [0.0, 452.92102051, 273.31335421],
        [0.0, 0.0, 1.0],
    ]
)


class Evaluator(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)

        # image transform
        input_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        # dataset and dataloader

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

    def eval(self):
        self.model.eval()
        if self.args.distributed:
            model = self.model.module
        else:
            model = self.model

        # cam = cv.VideoCapture(0)
        sleep(1)
        sample_full_paths = [
            os.path.join(self.args.video_file, fname)
            for fname in os.listdir(self.args.video_file)
        ]
        print(sample_full_paths)

        for sample_full_path in sample_full_paths:
            cam = cv.VideoCapture(sample_full_path)
            if self.args.save_pred:
                out = cv.VideoWriter(
                    sample_full_path.replace(".mp4", "sunrgbd.mp4").replace(
                        "samples", "results"
                    ),
                    cv.VideoWriter_fourcc("M", "J", "P", "G"),
                    20,
                    (320 * 2, 240),
                )
            while True:
                tic = time()
                ret, np_image = cam.read()
                # print(ret)
                if not ret:
                    break
                np_image = cv.undistort(np_image, mtx, dist, None, newcameramtx)
                np_image = cv.resize(np_image, dsize=(320, 240))
                np_image_tp = np_image.transpose([2, 0, 1])
                np_image_tp = np.array([np_image_tp])

                image = torch.from_numpy(np_image_tp).to(self.device).to(torch.float)
                image = image / 255
                with torch.no_grad():

                    outputs = model(image)
                    toc = time()
                    print(f"{1 / (toc - tic):.2f}hz")
                    pred = torch.argmax(outputs[0], 1)
                    pred = pred.cpu().data.numpy()
                    mask = pred.squeeze(0)
                    # predict = pred.squeeze(0)
                    # mask = get_color_pallete(predict, self.args.dataset)
                    # mask = np.array(mask)
                    overlap = (np.array(color.label2rgb(mask, np_image)) * 255).astype(
                        np.uint8
                    )
                    save_frame = cv.resize(
                        np.hstack([overlap, np_image]), (320 * 2, 240)
                    )

                    cv.imshow(f"overlap", overlap)
                    # cv.imshow("mask", mask)
                    cv.imshow("ori", np_image)
                    cv.imshow("ddd", save_frame)
                    cv.waitKey(1)
                if self.args.save_pred:
                    out.write(save_frame)
            cam.release()
            if self.args.save_pred:
                out.release()


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
    # args.save_pred = True
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
