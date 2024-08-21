from PIL import Image

# 定义图像大小
image = Image.open("cloth_texture.jpg")
width, height = image.size

# 创建一个全白色的图像
white_image = Image.new("RGB", (width, height), "white")

# 保存为 JPG 格式
white_image.save("white_image.jpg")