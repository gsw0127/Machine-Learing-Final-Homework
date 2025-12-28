# -*- coding: UTF-8 -*-
from pycocotools.coco import COCO
import os

# 数据集路径
coco_root = "/root/autodl-tmp/COCO2017"
images_root = os.path.join(coco_root, "images")
annotations_root = os.path.join(coco_root, "annotations")

# 标签保存路径
labels_root = os.path.join(coco_root, "labels")
os.makedirs(labels_root, exist_ok=True)

# 遍历 train2017 和 val2017
for dataType in ["train2017", "val2017"]:
    print(f"Processing {dataType}...")
    annFile = os.path.join(annotations_root, f"instances_{dataType}.json")
    coco = COCO(annFile)

    # 创建标签目录
    label_dir = os.path.join(labels_root, dataType)
    os.makedirs(label_dir, exist_ok=True)

    # COCO 原始类别 ID 映射到 0~79
    cats = coco.loadCats(coco.getCatIds())
    cls_dict = {cat["id"]: idx for idx, cat in enumerate(cats)}

    # 获取所有图像 ID
    imgIds = coco.getImgIds()
    for img_id in imgIds:
        img_info = coco.loadImgs(img_id)[0]
        file_name = img_info["file_name"]
        width = img_info["width"]
        height = img_info["height"]

        dw = 1.0 / width
        dh = 1.0 / height

        # 获取该图像的所有标注
        ann_ids = coco.getAnnIds(imgIds=img_id, iscrowd=None)
        anns = coco.loadAnns(ann_ids)

        # 写入 YOLO txt
        txt_path = os.path.join(label_dir, file_name[:-4] + ".txt")
        with open(txt_path, "w") as f:
            for ann in anns:
                bbox = ann["bbox"]  # [x, y, w, h]
                cls_id = cls_dict[ann["category_id"]]

                x_c = (bbox[0] + bbox[2]/2.0) * dw
                y_c = (bbox[1] + bbox[3]/2.0) * dh
                w = bbox[2] * dw
                h = bbox[3] * dh

                f.write(f"{cls_id} {x_c} {y_c} {w} {h}\n")

    print(f"{dataType} done! Labels saved to {label_dir}")
print("All done!")