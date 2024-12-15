from pycocotools.coco import COCO
import os
import json
import torch
from PIL import Image
from tqdm import tqdm

from util.visual import *

def categorize_area(area):
    """
    根据面积大小将目标分为小、中、大三类。

    Args:
        area (float): 标注中目标的面积。

    Returns:
        str: 分类结果，小目标/中目标/大目标。
    """
    if area < 32**2:
        return 'small'
    elif 32**2 <= area < 96**2:
        return 'medium'
    else:
        return 'large'

def compute_iou(box1, box2):
    """
    计算两个边界框的 IoU。

    Args:
        box1, box2 (list): 边界框 [xmin, ymin, xmax, ymax]。

    Returns:
        float: IoU 值。
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    iou = inter_area / (box1_area + box2_area - inter_area)
    return iou


if __name__ == '__main__':
    # 定义 COCO 数据路径
    coco_path = "Datasets/Coco_2017"
    val2017_path = os.path.join(coco_path, "val2017")
    annotations_path = os.path.join(coco_path, "annotations/instances_val2017.json")

    # 加载 COCO 数据集标注
    coco = COCO(annotations_path)

    # 遍历所有 val2017 图像 ID
    image_ids = coco.getImgIds()
    results = {'small': [], 'medium': [], 'large': []}

    # 加载 DETR 模型
    model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True)
    model.eval()

    # 遍历所有图像
    for img_id in tqdm(image_ids):
        # 获取图像信息
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(val2017_path, img_info['file_name'])

        # 加载图像
        im = Image.open(img_path).convert("RGB")
        img_tensor = transform(im).unsqueeze(0)

        # 获取图像的标注
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        # 按面积划分目标
        grouped_annotations = {'small': [], 'medium': [], 'large': []}
        for ann in anns:
            area_category = categorize_area(ann['area'])
            grouped_annotations[area_category].append(ann)

        # 推理模型
        with torch.no_grad():
            outputs = model(img_tensor)

        # 提取预测结果
        probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
        keep = probas.max(-1).values > 0.7  # 仅保留置信度大于 0.7 的预测
        pred_bboxes = outputs['pred_boxes'][0, keep].cpu().numpy()

        # 记录每个分组的预测结果
        for category, anns in grouped_annotations.items():
            for ann in anns:
                # COCO GT 的 bbox 格式为 [x_min, y_min, width, height]
                gt_bbox = ann['bbox']
                gt_area = ann['area']
                results[category].append({
                    'image_id': img_id,
                    'gt_bbox': gt_bbox,
                    'gt_area': gt_area,
                    'pred_bboxes': pred_bboxes.tolist()
                })
