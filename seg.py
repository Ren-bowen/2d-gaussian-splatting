import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
# from simple_lama_inpainting import SimpleLama
from segment_anything import sam_model_registry, SamPredictor
import os
from os import makedirs

# Load SAM model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sam = sam_model_registry["vit_h"](checkpoint="C:\\Users\\bowen ren\\Downloads\\sam_vit_h_4b8939 .pth")
sam.to(device)

# Create SAM predictor
predictor = SamPredictor(sam)

# Load images and perform segmentation
num = 0
# register_heif_opener()
name = "new_paper"
for i in range(0, 1700):
    # image_path = r'C:\ren\code\2d-gaussian-splatting\data\cloth_new\input_origin' + '\{}.jpg'.format(i)
    # image_path = r'C:\ren\code\2d-gaussian-splatting\data\new_paper\new-paper\IMG_{}.HEIC'.format(i)
    image_path = "../2d-gaussian-splatting/data/" + name + "/images1/image{}.png".format(i)
    # image = Image.open(image_path)
    print("image_path: ", image_path)
    print("os.path.exists(image_path): ", os.path.exists(image_path))
    if os.path.exists(image_path):
        print("i: ", i)
        print("num: ", num)
        image = Image.open(image_path)
        image_np = np.array(image)
        print("image_np.shape", image_np.shape)
        
        predictor.set_image(image_np)

        input_points = []
        input_labels = []

        def on_click(event):
            if event.inaxes is not None:
                x, y = int(event.xdata), int(event.ydata)
                print(f"Clicked at ({x}, {y})")
                if event.button == 1:
                    input_points.append([x, y])
                    input_labels.append(1)
                    plt.plot(x, y, 'go')
                elif event.button == 3:
                    input_points.append([x, y])
                    input_labels.append(0)
                    plt.plot(x, y, 'ro')
                plt.draw()

        def on_key(event):
            if event.key == 'enter':
                plt.close()

        fig, ax = plt.subplots()
        ax.imshow(image_np)
        cid = fig.canvas.mpl_connect('button_press_event', on_click)
        kid = fig.canvas.mpl_connect('key_press_event', on_key)
        plt.show()

        input_points = np.array(input_points)
        input_labels = np.array(input_labels)

        masks, scores, logits = predictor.predict(point_coords=input_points, point_labels=input_labels, multimask_output=True)
        # simple_lama = SimpleLama()
        best_mask_idx = np.argmax(scores)
        best_mask = masks[best_mask_idx]
        '''
        # extend best_mask for inpainting
        inpaint_mask = np.zeros((best_mask.shape), dtype=np.uint8)
        for i in range(best_mask.shape[0]):
            for j in range(best_mask.shape[1]):
                if best_mask[i, j] == 1:
                    for k in range(-8, 9):
                        for l in range(-8, 9):
                            if 0 <= i + k < best_mask.shape[0] and 0 <= j + l < best_mask.shape[1]:
                                inpaint_mask[i + k, j + l] = 1
        result = simple_lama(image_np, inpaint_mask)
        '''
        # Create an alpha channel where the mask is fully opaque (255) and others are fully transparent (0)
        alpha_channel = np.where(best_mask, 255, 0).astype(np.uint8)

        # Combine the RGB image with the alpha channel to create an RGBA image
        rgba_image_np = np.dstack((image_np, alpha_channel))
        plt.figure(figsize=(10, 10))
        plt.imshow(rgba_image_np)
        plt.axis('off')
        plt.show()
        
        # Save the RGBA image
        inpaint_folder = '../2d-gaussian-splatting/data/' + name + '/inpaint'
        output_folder = '../2d-gaussian-splatting/data/' + name + '/images'
        origin_folder = '../2d-gaussian-splatting/data/' + name + '/input_origin'
        # makedirs(inpaint_folder, exist_ok=True)
        makedirs(output_folder, exist_ok=True)
        # makedirs(origin_folder, exist_ok=True)
        # inpaint_path = inpaint_folder + "/image" + str(num) + ".png"
        output_path = output_folder + "/image" + str(num) + ".png"
        # origin_path = origin_folder + "/render_" + str(num) + ".png"
        rgba_image_pil = Image.fromarray(rgba_image_np)
        # print("rgba_image_pil.split(): ", rgba_image_pil.split())
        # result.save(inpaint_path, format='png')
        rgba_image_pil.save(output_path, format='png')
        # image.save(origin_path, format='png')
        
        num += 1
