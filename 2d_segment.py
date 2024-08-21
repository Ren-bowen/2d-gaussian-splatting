import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import imageio.v3 as iio
from segment_anything import sam_model_registry, SamPredictor
import os
from os import makedirs

# 加载SAM模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sam = sam_model_registry["vit_h"](checkpoint="C:\\Users\\bowen ren\\Downloads\\sam_vit_h_4b8939 .pth")  # 替换为你的模型权重路径
sam.to(device)

# 创建SAM预测器
predictor = SamPredictor(sam)

# 加载图片
for i in range(1400, 1600):
    # image_path = r'C:\ren\code\2d-gaussian-splatting\data\cloth_new\input_origin' + '\{}.jpg'.format(i)
    image_path = r'C:\Users\bowen ren\Downloads\paper\paper\IMG_{}.HEIC'.format(i)
    # image = Image.open(image_path)
    # 读取 HEIC 文件
    # if exist image_path:
    if os.path.exists(image_path):
        image = iio.imread(image_path)
        image_np = np.array(image)

        # 转换图像为张量并移动到设备
        predictor.set_image(image_np)

        # 保存用户点击的点
        input_points = []
        input_labels = []

        def on_click(event):
            if event.inaxes is not None:
                x, y = int(event.xdata), int(event.ydata)
                print(f"Clicked at ({x}, {y})")

                # 记录前景点（左键）或背景点（右键）
                if event.button == 1:  # 左键
                    input_points.append([x, y])
                    input_labels.append(1)
                    plt.plot(x, y, 'go')  # 绿色点表示前景
                elif event.button == 3:  # 右键
                    input_points.append([x, y])
                    input_labels.append(0)
                    plt.plot(x, y, 'ro')  # 红色点表示背景

                plt.draw()

        def on_key(event):
            if event.key == 'enter':
                plt.close()  # 关闭图像窗口

        # 显示图像并设置鼠标点击事件
        fig, ax = plt.subplots()
        ax.imshow(image_np)
        cid = fig.canvas.mpl_connect('button_press_event', on_click)
        kid = fig.canvas.mpl_connect('key_press_event', on_key)
        plt.show()

        # 转换输入点和标签为numpy数组
        input_points = np.array(input_points)
        input_labels = np.array(input_labels)

        # 预测分割掩码
        masks, scores, logits = predictor.predict(point_coords=input_points, point_labels=input_labels, multimask_output=True)

        # 选择得分最高的掩码
        best_mask_idx = np.argmax(scores)
        best_mask = masks[best_mask_idx]

        # 创建一个白底背景的图像
        # white_background = np.ones_like(image_np) * 255
        black_background = np.zeros_like(image_np)

        # 应用掩码到原始图片并在白底背景上显示
        segmented_image = np.where(best_mask[..., None], image_np, black_background)

        # 显示分割结果
        plt.figure(figsize=(10, 10))
        plt.imshow(segmented_image)
        plt.axis('off')
        plt.show()

        # 保存分割结果为JPEG文件
        output_folder = r'C:\ren\code\2d-gaussian-splatting\data\paper\input'
        origin_folder = r'C:\ren\code\2d-gaussian-splatting\data\paper\input_origin'
        makedirs(output_folder, exist_ok=True)
        makedirs(origin_folder, exist_ok=True)
        output_path = output_folder + r"\image" + str(i) + ".png"
        origin_path = origin_folder + r"\image" + str(i) + ".png"
        segmented_image_pil = Image.fromarray(segmented_image.astype(np.uint8))
        segmented_image_pil.save(output_path, format='png')
        image.save(origin_path, format='png')
