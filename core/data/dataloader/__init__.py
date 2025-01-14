"""
This module provides data loaders and transformers for popular vision datasets.
"""
from .mscoco import COCOSegmentation
from .cityscapes import CitySegmentation
from .ade import ADE20KSegmentation
from .ade_gnd import ADE20KSegmentationGround
from .ade_gho import ADE20KSegmentationGho
from .sunrgbd import SunrgbdSegmentation
from .sunrgbd_resized import SunrgbdResizedSegmentation
from .pascal_voc import VOCSegmentation
from .pascal_aug import VOCAugSegmentation
from .sbu_shadow import SBUSegmentation

datasets = {
    "ade20k": ADE20KSegmentation,
    "ade20k_gnd": ADE20KSegmentationGround,
    "ade20k_gho": ADE20KSegmentationGho,
    "sunrgbd": SunrgbdSegmentation,
    "sunrgbd_resized": SunrgbdResizedSegmentation,
    "pascal_voc": VOCSegmentation,
    "pascal_aug": VOCAugSegmentation,
    "coco": COCOSegmentation,
    "citys": CitySegmentation,
    "sbu": SBUSegmentation,
}


def get_segmentation_dataset(name, **kwargs):
    """Segmentation Datasets"""
    return datasets[name.lower()](**kwargs)
