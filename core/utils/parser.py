import argparse


def str2bool(boostr: str):
    boostr = boostr.lower()
    if boostr == "true":
        return True
    elif boostr == "false":
        return False


def parse_args():
    parser = argparse.ArgumentParser(
        description="Semantic Segmentation Training With Pytorch"
    )
    # model and dataset

    parser.add_argument("--eval_img_dir", type=str, required=False)
    parser.add_argument("--tag", type=str, required=False)

    parser.add_argument(
        "--model",
        type=str,
        default="fcn",
        choices=[
            "fcn32s",
            "fcn16s",
            "fcn8s",
            "fcn",
            "psp",
            "deeplabv3",
            "deeplabv3_plus",
            "danet",
            "denseaspp",
            "bisenet",
            "encnet",
            "dunet",
            "icnet",
            "enet",
            "ocnet",
            "psanet",
            "cgnet",
            "espnet",
            "lednet",
            "dfanet",
            "swnet",
        ],
        help="model name (default: fcn32s)",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="resnet50",
        choices=[
            "vgg16",
            "resnet18",
            "resnet34",
            "resnet50",
            "resnet101",
            "resnet152",
            "densenet121",
            "densenet161",
            "densenet169",
            "densenet201",
        ],
        help="backbone name (default: vgg16)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="pascal_voc",
        choices=[
            "pascal_voc",
            "pascal_aug",
            "ade20k",
            "ade20k_gnd",
            "ade20k_lres",
            "ade20k_gho",
            "sunrgbd",
            "sunrgbd_resized",
            "citys",
            "sbu",
        ],
        help="dataset name (default: pascal_voc)",
    )
    parser.add_argument("--base-size", type=int, default=520, help="base image size")
    parser.add_argument("--crop-size", type=int, default=480, help="crop image size")
    parser.add_argument(
        "--workers", "-j", type=int, default=4, metavar="N", help="dataloader threads"
    )
    parser.add_argument(
        "--video-file",
        type=str,
        default="",
        required=False,
    )
    # training hyper params
    parser.add_argument("--jpu", action="store_true", default=False, help="JPU")
    parser.add_argument(
        "--use-ohem", type=bool, default=False, help="OHEM Loss for cityscapes dataset"
    )
    parser.add_argument(
        "--aux", action="store_true", default=False, help="Auxiliary loss"
    )
    parser.add_argument(
        "--aux-weight", type=float, default=0.4, help="auxiliary loss weight"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        metavar="N",
        help="input batch size for training (default: 8)",
    )
    parser.add_argument(
        "--start_epoch",
        type=int,
        default=0,
        metavar="N",
        help="start epochs (default:0)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        metavar="N",
        help="number of epochs to train (default: 50)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        metavar="LR",
        help="learning rate (default: 1e-4)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        metavar="M",
        help="momentum (default: 0.9)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        metavar="M",
        help="w-decay (default: 5e-4)",
    )
    parser.add_argument("--warmup-iters", type=int, default=0, help="warmup iters")
    parser.add_argument(
        "--warmup-factor", type=float, default=1.0 / 3, help="lr = warmup_factor * lr"
    )
    parser.add_argument(
        "--warmup-method", type=str, default="linear", help="method of warmup"
    )
    # cuda setting
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument("--local_rank", type=int, default=0)
    # checkpoint and log
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="put the path to resuming file if needed",
    )
    parser.add_argument(
        "--save-pred",
        type=str2bool,
        default=False,
        required=False,
    )
    parser.add_argument(
        "--save-dir",
        default="~/.torch/models",
        help="Directory for saving checkpoint models",
    )
    parser.add_argument(
        "--save-epoch", type=int, default=10, help="save model every checkpoint-epoch"
    )
    parser.add_argument(
        "--log-dir",
        default="../runs/logs/",
        help="Directory for saving checkpoint models",
    )
    parser.add_argument(
        "--log-iter", type=int, default=10, help="print log every log-iter"
    )
    # evaluation only
    parser.add_argument(
        "--val-epoch", type=int, default=1, help="run validation every val-epoch"
    )
    parser.add_argument(
        "--skip-val",
        action="store_true",
        default=False,
        help="skip validation during training",
    )
    parser.add_argument(
        "--imshow",
        type = str2bool,
        default=True,
        required=False
    )
    args = parser.parse_args()

    # default settings for epochs, batch_size and lr
    if args.epochs is None:
        epoches = {
            "coco": 30,
            "pascal_aug": 80,
            "pascal_voc": 50,
            "pcontext": 80,
            "ade20k": 160,
            "ade20k_gnd": 160,
            "ade20k_gho": 160,
            "citys": 120,
            "sbu": 160,
        }
        args.epochs = epoches[args.dataset.lower()]
    if args.lr is None:
        lrs = {
            "coco": 0.004,
            "pascal_aug": 0.001,
            "pascal_voc": 0.0001,
            "pcontext": 0.001,
            "ade20k": 0.01,
            "ade20k_gnd": 0.01,
            "ade20k_gho": 0.01,
            "citys": 0.01,
            "sbu": 0.001,
        }
        args.lr = lrs[args.dataset.lower()] / 8 * args.batch_size
    return args
