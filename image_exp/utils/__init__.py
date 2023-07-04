# Adapted from https://github.com/openai/guided-diffusion, which is under the MIT license

import yaml
import os
from PIL import Image
import matplotlib.pyplot as plt 


def txtread(path):
    path = os.path.expanduser(path)
    with open(path, 'r') as f:
        return f.read()


def yamlread(path):
    return yaml.safe_load(txtread(path=path))

def imwrite(path=None, img=None):
    if img.shape[-1] == 1:
        # squeeze the channel dimension if it is 1, i.e. greyscale image 
        img = img.squeeze(-1)
    Image.fromarray(img).save(path) 


def multi_imgwrite(save_path=None, img_list=[], img_names=[], title=None, vmin=None, vmax=None):
    num_imgs = len(img_list)
    # image is (H, W)

    fig, axes = plt.subplots(1, num_imgs, figsize=(4*num_imgs, 4))
    if num_imgs == 1:
        img = img_list[0]
        if img is not None:  
            img_name = img_names[0]
            ax = axes

            im = ax.imshow(img, cmap='gray', vmin=vmin, vmax=vmax)
            ax.set_title(img_name)
            fig.colorbar(im, ax=ax)

    else:
        for ax, img, img_name in zip(axes, img_list, img_names):
            if img is None:
                continue 

            im = ax.imshow(img, cmap='gray', vmin=vmin, vmax=vmax)
            ax.set_title(img_name)
            fig.colorbar(im, ax=ax)
    if title is not None:
        fig.suptitle(title)

    plt.tight_layout() 
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight') 
        plt.close()
    else:
        plt.show()
         