import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sam import build_sam, SamAutomaticMaskGenerator

# 加载图片
image_path = 'path_to_your_image.jpg'
image = Image.open(image_path)
image_np = np.array(image)

# 加载SAM模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sam = build_sam('sam_vit_h_4b8939.pth')  # 加载预训练权重
sam.to(device)

# 生成分割掩码
mask_generator = SamAutomaticMaskGenerator(sam)
masks = mask_generator.generate(image_np)

# 选择一个掩码（假设你知道目标物体的掩码）
# 这里假设选择第一个掩码
mask = masks[0]['segmentation']

# 应用掩码到原始图片
segmented_image = image_np * mask[..., None]

# 显示分割结果
plt.figure(figsize=(10, 10))
plt.imshow(segmented_image)
plt.axis('off')
plt.show()
