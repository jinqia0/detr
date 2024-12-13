import matplotlib.pyplot as plt
from pycocotools.coco import COCO
import matplotlib.image as mpimg
import random

# 设置路径到 COCO 标注 JSON 文件
annotation_file = 'Datasets/Coco_2017/annotations/instances_val2017.json'

# 加载 COCO 数据集
coco = COCO(annotation_file)

# 获取所有类别信息
categories = coco.loadCats(coco.getCatIds())
category_id = random.choice(coco.getCatIds())  # 随机选择一个类别的 ID
category_name = coco.loadCats(category_id)[0]['name']
print(f"随机选择的类别: {category_name} (ID: {category_id})")

# 获取符合该类别的图片 ID
image_ids = coco.getImgIds(catIds=[category_id])
random_image_id = random.choice(image_ids)

# 加载图片信息
image_info = coco.loadImgs(random_image_id)[0]
image_path = f'Datasets/Coco_2017/val2017/{image_info["file_name"]}'
image = mpimg.imread(image_path)

# 加载标注框
annotations = coco.loadAnns(coco.getAnnIds(imgIds=[random_image_id], catIds=[category_id]))
fig, ax = plt.subplots(1, figsize=(12, 8))
ax.imshow(image)

# 绘制边界框
for annotation in annotations:
    bbox = annotation['bbox']
    rect = plt.Rectangle(
        (bbox[0], bbox[1]), bbox[2], bbox[3],
        linewidth=2, edgecolor='r', facecolor='none'
    )
    ax.add_patch(rect)

plt.axis('off')
plt.show()