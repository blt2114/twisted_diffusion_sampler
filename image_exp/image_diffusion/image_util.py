# Adpated from https://github.com/andreas128/RePaint,
# which was forked from https://github.com/openai/guided-diffusion, which is under the MIT license


import random
import os

from PIL import Image
import blobfile as bf
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch 
import matplotlib.pyplot as plt 


def get_dataset(dataset:str):
    if dataset == 'mnist':
        dataset_path = "./data/mnist32" # TODO: check this 
    return dataset_path 


def load_data_yield(loader):
    while True:
        yield from loader

def get_dataloader(
    dataset_path, 
    batch_size,
    image_size,
    class_cond=False,
    deterministic=False,
    random_crop=False,
    random_flip=False,
    return_dataloader=False,
    return_dict=False,
    max_len=None,
    drop_last=True,
    rgb=False,
    dataset_label=None, # if not None, get data with a speicifed label
    img_name=None, 
    offset=0,
    dataset_size=None, 
    **kwargs
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """

    dataset_dir = os.path.expanduser(dataset_path)
    dataset_paths = _list_image_files_recursively(dataset_dir, dataset_label=dataset_label, img_name=img_name)

    classes = None
    if class_cond:
        raise NotImplementedError()

    assert image_size is not None 
    dataset = ImageDatasetInpa(
        image_size,
        dataset_paths=dataset_paths,
        classes=classes,
        shard=0,
        num_shards=1,
        random_crop=random_crop,
        random_flip=random_flip,
        rgb=rgb, 
        return_dict=return_dict,
        max_len=max_len,
        offset=offset, 
        dataset_size=dataset_size, 
    )

    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=drop_last
        )

    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=drop_last
        )

    if return_dataloader:
        return loader
    else:
        return load_data_yield(loader)


def _list_image_files_recursively(data_dir, dataset_label=None, img_name=None):
    results = []
    if dataset_label is not None:
        dataset_label = [d for d in str(dataset_label)]
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            if dataset_label is not None:
                actual_label = entry.split("_")[0]
                if actual_label not in dataset_label:
                    continue 
            if img_name is not None:
                if img_name not in entry:
                    continue 
                else:
                    print(f"Found {img_name}!")
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


class ImageDatasetInpa(Dataset):
    def __init__(
        self,
        resolution,
        dataset_paths, 
        classes=None,
        shard=0,
        num_shards=1,
        random_crop=False,
        random_flip=True,
        rgb=True, 
        return_dict=False,
        max_len=None,
        offset=0, 
        dataset_size=None, 
    ):
        super().__init__()
        self.resolution = resolution # image_size

        gt_paths = sorted(dataset_paths)[offset:] # ground truth images 
        total_size = len(gt_paths)
        if dataset_size is not None:
            assert dataset_size > 0, dataset_size 
            dataset_size = min(dataset_size, total_size)
            gt_paths = gt_paths[:dataset_size]

        self.local_gts = gt_paths[shard:][::num_shards]
   
        self.local_classes = None if classes is None else classes[shard:][::num_shards]

        self.random_crop = random_crop
        self.random_flip = random_flip
        self.rgb = rgb 
        self.return_dict = return_dict
        self.max_len = max_len

    def __len__(self):
        if self.max_len is not None:
            return self.max_len

        return len(self.local_gts)

    def __getitem__(self, idx):
        gt_path = self.local_gts[idx]
        pil_gt = self.imread(gt_path)

        if self.random_crop:
            raise NotImplementedError()
        else:
            arr_gt = center_crop_arr(pil_gt, self.resolution)

        if self.random_flip and random.random() < 0.5:
            arr_gt = arr_gt[:, ::-1]

        arr_gt = arr_gt.astype(np.float32) / 127.5 - 1

        if not self.rgb:
            arr_gt = arr_gt[..., 0:1]

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)

        if self.return_dict:
            name = os.path.basename(gt_path)
            return {
                'GT': np.transpose(arr_gt, [2, 0, 1]),# after transpose C, H, W 
                'GT_name': name,
            }
        else:
            raise NotImplementedError()

    def imread(self, path):
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("RGB")
        return pil_image


def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.

    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size]


def gen_mask(mask_type, ref_image, ref_image_name):
    C, H, W = ref_image.shape 
    mask = torch.zeros_like(ref_image, dtype=torch.bool)

    if mask_type == 'load':
        # TODO: load using ref_iamge_name 
        raise NotImplementedError
    
    elif mask_type == 'none':
        return mask 
        
    elif mask_type == 'half':
        # left half of the image is measurement
        mask[:, :, :int(W/2)] = 1 
    
    elif mask_type == 'half-dof':
        mask = torch.zeros(2, C, H, W, dtype=torch.bool, device=ref_image.device)
        mask[0, :, :, :int(W/2)] = 1 # left half is measurement
        mask[1, :, :, int(W/2):] = 1 # right half is measurement 
    
    elif mask_type == 'quarter':
        # top left quater of the image is measurement
        mask[:, :int(H/2), :int(W/2)] = 1 

    elif mask_type == "center":
        mask[:, int(H/4):int(H/4*3), int(W/4):int(W/4*3)] = 1

    elif "random" in mask_type:
        # format random70 --mask out each pixel with prob 0.70 
        prob = float(mask_type[len("random"):]) / 100
        mask = torch.bernoulli(prob*torch.ones_like(mask)) 
        mask = mask.type(torch.bool)
    
    else:
        raise ValueError
    return mask 


def toU8(sample, clamp=True):
    if sample is None:
        return sample

    if clamp:
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    else:
        sample = ((sample + 1) * 127.5).to(torch.uint8)

    if len(sample.shape) == 3: #(C, H, W) -> (H, W, C)
        sample = sample.permute(1,2,0) 
    elif len(sample.shape) == 4:  # (B, C, H, W) -> (B, H, W, C)
       sample = sample.permute(0, 2, 3, 1) 
    elif len(sample.shape) == 5: # (P, B, C, H, W) -> (P, B, H, W, C)
        sample = sample.permute(0,1, 3, 4, 2) 
    else:
        raise NotImplementedError
    
    sample = sample.contiguous()
    sample = sample.detach().cpu().numpy()
    return sample


def visualize_weights(path=None, log_w=None):
    if log_w is not None:
        P = log_w.shape[0]
        w = torch.softmax(log_w-log_w.max(), dim=0).numpy()
        nrow = int(np.ceil(np.sqrt(P)))
        ncol = int(np.ceil(P/nrow))
        w_grid = np.ones(nrow*ncol) * (-1)
        w_grid[:P] = w
        w_grid = w_grid.reshape(nrow,ncol)
        plt.imshow(w_grid,  cmap='viridis')
        plt.colorbar()
        if path is None:
            plt.show()
        else:
            plt.savefig(path)
            plt.close()



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
    
    # use same colorbar for all plots 
    #fig.subplots_adjust(right=0.8)
    #cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    #fig.colorbar(im, cax=cbar_ax)

    plt.tight_layout() 
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight') 
        plt.close()
    else:
        plt.show()
         
