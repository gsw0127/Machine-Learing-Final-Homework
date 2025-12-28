import os
import json
from pycocotools.coco import COCO

def convert_coco_to_yolo(coco_json_path, output_dir, img_dir):
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 加载 COCO 数据集
    coco = COCO(coco_json_path)
    
    # 获取类别信息
    categories = coco.loadCats(coco.getCatIds())
    category_map = {cat['id']: idx for idx, cat in enumerate(categories)}

    # 遍历图片并生成 YOLO 格式标签
    img_ids = coco.getImgIds()
    for img_id in img_ids:
        img_info = coco.loadImgs(img_id)[0]
        img_file_name = img_info['file_name']
        img_width = img_info['width']
        img_height = img_info['height']
        
        # 获取对应图片的标注信息
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        
        # 打开输出的 YOLO 格式文件
        yolo_txt_file = os.path.join(output_dir, os.path.splitext(img_file_name)[0] + '.txt')
        with open(yolo_txt_file, 'w') as f:
            for ann in anns:
                if ann['iscrowd'] == 1:
                    continue
                
                # 获取类别 ID 和边界框
                category_id = ann['category_id']
                bbox = ann['bbox']
                x, y, w, h = bbox
                
                # 归一化坐标
                center_x = (x + w / 2) / img_width
                center_y = (y + h / 2) / img_height
                w /= img_width
                h /= img_height
                
                # 写入 YOLO 格式的标签文件：<class_id> <center_x> <center_y> <width> <height>
                f.write(f"{category_map[category_id]} {center_x} {center_y} {w} {h}\n")

    print("转换完成！")

# 调用转换函数
coco_json_path = '/root/autodl-tmp/COCO2017/annotations/instances_train2017.json'  # COCO的JSON文件路径
output_dir = '/root/autodl-tmp/COCO2017/labels/train2017'  # 保存YOLO格式标签的目录
img_dir = '/root//autodl-tmp/COCO2017/images/train2017'  # 图片目录

convert_coco_to_yolo(coco_json_path, output_dir, img_dir)
