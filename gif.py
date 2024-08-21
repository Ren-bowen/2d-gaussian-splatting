from PIL import Image
import os

# 定义图片路径和GIF输出路径
image_folder = r"C:\ren\code\2d-gaussian-splatting\output\regular\train"  # PNG图片所在的文件夹路径

def animate(i):
    gif_name = "/gif/output{}_rot.gif".format(i)  # 输出GIF文件的名称

    # 获取图片文件名列表并排序
    images = []
    for j in range(0, 36, 2):
        image_path = os.path.join(image_folder, "ours_30000_{}_rot_".format(j), "renders", "{:05}.png".format(i))
        images.append(image_path)
    # images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    # images.sort()  # 确保图片按顺序排列

    # 打开所有图片并将它们转换为Pillow的Image对象
    frames = [Image.open(image) for image in images]

    # 将图片合成GIF
    os.makedirs(os.path.join(image_folder, "gif"), exist_ok=True)
    frames[0].save(image_folder + gif_name, format='GIF',
                append_images=frames[1:], 
                save_all=True, 
                duration=160, loop=0)

    print(f"GIF saved as {gif_name}")

for i in range(52):
    print("i: ", i)
    animate(i)