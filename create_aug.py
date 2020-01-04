from data_aug.data_aug import *
from data_aug.bbox_util import *
import ast
import cv2
import pickle as pkl
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt


def create_tag_dicts(annotations_file_gt):
    tags_dict = {}
    with open(annotations_file_gt, "r") as ann_f:
        lines = ann_f.readlines()

    for line_ in lines:
        line = line_.replace(' ', '')
        imName = line.split(':')[0]
        anns_ = line[line.index(':') + 1:].replace('\n', '')
        anns = ast.literal_eval(anns_)
        if (not isinstance(anns, tuple)):
            anns = [anns]
        tags_dict[imName] = anns
    return tags_dict

# def get_boxes_and_labels(tags_dict, bus_dir, filename):
#     # load images ad masks
#     img_path = os.path.join("busesTrain", self.imgs[idx])
#     img = Image.open(img_path).convert("RGB")
#     anns = self.tags_dict[self.imgs[idx]]
#     # "([xmin1, ymin1, width1, height1,color1], [xmin1, ymin1, width1, height1,color1])"
#     # get bounding box coordinates for each mask
#     num_anns = len(anns)  # num of boxes
#     boxes = []
#     labels = []
#     for ann in anns:
#         xmin = ann[0]
#         xmax = xmin + ann[2]
#         ymin = ann[1]
#         ymax = ymin + ann[3]
#         boxes.append([xmin, ymin, xmax, ymax])
#         labels.append(ann[4])
#
#     boxes = torch.as_tensor(boxes, dtype=torch.float32)
#
#     # there is only one class
#     labels = torch.as_tensor(labels, dtype=torch.int64)
#
#     image_id = torch.tensor([idx])
#     area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
#     # suppose all instances are not crowd
#     iscrowd = torch.zeros((num_anns,), dtype=torch.int64)
#
#         target = {}
#         target["boxes"] = boxes
#         target["labels"] = labels
#         target["image_id"] = image_id
#         target["area"] = area
#         target["iscrowd"] = iscrowd
#
#         if self.transforms is not None:
#             img, target = self.transforms(img, target)
#
#         return img, target


def create_bboxes(anns):
    bboxes = []
    for ann in anns:
        xmin = ann[0]
        xmax = xmin + ann[2]
        ymin = ann[1]
        ymax = ymin + ann[3]
        label = ann[4]
        bboxes.append([xmin, ymin, xmax, ymax, label])
    bboxes = np.array(bboxes, dtype=np.float64)
    return bboxes


def create_line(pic_name, bboxes):
    line_str = f"{pic_name}:"  # DSCF1013.JPG:[1217,1690,489,201,1],[1774,1619,475,224,2]
    first = True
    for bbox in bboxes:
        x0, y0, x1, y1, label = bbox[0], bbox[1], bbox[2], bbox[3], bbox[4]
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
        if first:
            box_str = f"[{x0},{y0},{x1 - x0},{y1 - y0},{label}]"
            first = False
        else:
            box_str = f",[{x0},{y0},{x1 - x0},{y1 - y0},{label}]"

        line_str += box_str
    line_str += '\n'
    return line_str


def do_augmentation(img, bboxes):
    """
    flip_p - do horizontal flip
    scale_arg - scaling for both x and y directions are randomly sampled from (-scale_arg, scale_arg)
    trans_arg - translating factors x,y are randomly sampled from (-trans_arg, trans_arg)
    rotation_arg - rotating angle, in degrees, is sampled from (- rotation_arg, rotation_arg)
    shear_arg - shearing factor is sampled from (- shear_arg, shear_arg)
    """
    flip_p = 0.1
    scale_arg = 0.3
    trans_arg = 0.1
    rotation_arg = 10
    shear_arg = 0.1

    ### NOITCE: we can resize easily with Resize(square_size)
    # The square_size to this augmentation is the side of the square.
    # Maybe it will be usefull in order to train smaller pics..
    transforms = Sequence([RandomHorizontalFlip(flip_p),
                           RandomScale(scale_arg, diff=True),
                           RandomTranslate(trans_arg, diff=True),
                           RandomShear(shear_arg),
                           RandomRotate(rotation_arg)])
    img, bboxes = transforms(img, bboxes)
    return img, bboxes


if __name__ == "__main__":

    buses_dir = "busesTrain"
    aug_buses_dir = "augmentationTrain"
    aug_box_buses_dir = "with_boxes"
    if not os.path.exists(aug_buses_dir):
        os.makedirs(aug_buses_dir)
    if not os.path.exists(os.path.join(aug_buses_dir, aug_box_buses_dir)):
        os.makedirs(os.path.join(aug_buses_dir, aug_box_buses_dir))
    if not os.path.exists(os.path.join(aug_buses_dir, buses_dir)):
        os.makedirs(os.path.join(aug_buses_dir, buses_dir))

    annotations_file_gt ="annotationsTrain.txt"
    annotations_file_aug = os.path.join(aug_buses_dir, annotations_file_gt)

    tags_dict = create_tag_dicts(annotations_file_gt)

    with open(annotations_file_aug, "w") as augAnnFile:
        for index in range(50):
            for img_name in tags_dict:
                anns = tags_dict[img_name]
                img_path = os.path.join(buses_dir, img_name)
                img = cv2.imread(img_path)[:, :, ::-1]  # OpenCV uses BGR channels
                bboxes = create_bboxes(anns)

                # on first loop save images as is
                if index != 0:
                    img, bboxes = do_augmentation(img, bboxes)

                # save image to aug dir
                aug_name = "aug" + str(index) + img_name
                aug_img_path = os.path.join(aug_buses_dir, buses_dir, aug_name)
                cv2.imwrite(aug_img_path, img)
                print(aug_img_path)

                # save image with boxes to augRec dir (for validating they are on the right place by looking..)
                aug_rec_img_path = os.path.join(aug_buses_dir, aug_box_buses_dir, aug_name)
                cv2.imwrite(aug_rec_img_path, draw_rect(img, bboxes))
                print(aug_rec_img_path)

                # write boxes to aug_annotation file
                line = create_line(aug_name, bboxes)
                augAnnFile.write(line)
                print(line)
