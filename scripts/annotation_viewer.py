from re import I
import cv2 as cv
import numpy as np
import sys, os


def masking_ground(img, class_id):
    img[img == 4] = 255
    img[img == 7] = 255
    img[img == 14] = 255
    img[img == 30] = 255
    img[img == 53] = 255
    img[img == 55] = 255
    img[img == 12] = 255  # pavement
    img[img == 10] = 255
    img[img != 255] = 0
    img = cv.dilate(img, kernel=np.ones((3, 3)))
    img = cv.erode(img, kernel=np.ones((3, 3)))
    img[img == 255] = class_id
    return img


def masking_human(img, class_id):
    img[img == 13] = 255
    img[img != 255] = 0
    img = cv.dilate(img, kernel=np.ones((3, 3)))
    img = cv.erode(img, kernel=np.ones((3, 3)))
    img[img == 255] = class_id
    return img


train_annotation_dir = "/home/nscl2004/segmentation/datasets/ade20k_gnd/ADEChallengeData2016/annotations/training"
valid_annotation_dir = "/home/nscl2004/segmentation/datasets/ade20k_gnd/ADEChallengeData2016/annotations/validation"

train_image_dir = "/home/nscl2004/segmentation/datasets/ade20k_gnd/ADEChallengeData2016/images/training"
valid_image_dir = "/home/nscl2004/segmentation/datasets/ade20k_gnd/ADEChallengeData2016/images/validation"

result_train_dir = "/home/nscl2004/segmentation/datasets/ade20k_gnd/ADEChallengeData2016/annotations/gh_train"
result_train_dir = "/home/nscl2004/segmentation/datasets/ade20k_gnd/ADEChallengeData2016/annotations/gh_validation"

train_annotation_files = os.listdir(train_annotation_dir)
valid_annotation_files = os.listdir(valid_annotation_dir)
train_image_files = os.listdir(train_image_dir)
valid_image_files = os.listdir(valid_image_dir)

train_annotation_files.sort()
valid_annotation_files.sort()
train_image_files.sort()
valid_image_files.sort()


floor_labels = (9, 15, 33, 43, 44, 145)

for i, fname in enumerate(train_annotation_files):
    train_fname = os.path.join(train_annotation_dir, fname)
    train_annotation_files[i] = train_fname

for i, fname in enumerate(valid_annotation_files):
    valid_fname = os.path.join(valid_annotation_dir, fname)
    valid_annotation_files[i] = valid_fname

for i, fname in enumerate(train_image_files):
    train_fname = os.path.join(train_image_dir, fname)
    train_image_files[i] = train_fname

for i, fname in enumerate(valid_image_files):
    valid_fname = os.path.join(valid_image_dir, fname)
    valid_image_files[i] = valid_fname

for i, fname in enumerate(train_annotation_files):
    img_anno = cv.imread(fname)

    mask_gnd = img_anno.copy()
    mask_human = img_anno.copy()

    mask_human = masking_human(mask_human, 100)
    mask_gnd = masking_ground(mask_gnd, 200)
    cv.imshow("train", mask_human + mask_gnd)
    cv.waitKey(1)
    if i % 100 == 0:
        print(f"{i}'th train img converted")

for i, fname in enumerate(valid_annotation_files):
    img_anno = cv.imread(fname)

    mask_gnd = img_anno.copy()
    mask_human = img_anno.copy()

    mask_human = masking_human(mask_human, 100)
    mask_gnd = masking_ground(mask_gnd, 200)
    cv.imshow("valid", mask_human + mask_gnd)
    cv.waitKey(1)
    if i % 100 == 0:
        print(f"{i}'th annotation img converted")
print(id(img_anno))
print(id(mask_gnd))

cv.waitKey(0)
