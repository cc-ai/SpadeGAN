"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from __future__ import print_function
from utils import get_config, pytorch03_to_pytorch04, sorted_nicely
from trainer import MUNIT_Trainer
import argparse
from torch.autograd import Variable
import torchvision.utils as vutils
import sys
import torch
import os
from torchvision import transforms
from PIL import Image
import tqdm as tq
import glob
import cv2 as cv
import numpy as np
import math

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, help="network configuration file")
parser.add_argument("--input", type=str, help="directory of input images")
parser.add_argument("--mask_dir", type=str, help="directory of masks corresponding to input images")
parser.add_argument("--output_folder", type=str, help="output image directory")
parser.add_argument("--checkpoint", type=str, help="checkpoint of generator")

parser.add_argument("--seed", type=int, default=10, help="random seed")

parser.add_argument(
    "--synchronized", action="store_true", help="whether use synchronized style code or not",
)
parser.add_argument(
    "--save_input", action="store_true", help="whether use synchronized style code or not",
)
parser.add_argument(
    "--output_path", type=str, default=".", help="path for logs, checkpoints, and VGG model weight",
)
parser.add_argument(
    "--save_mask", action="store_true", help="whether to save mask or not",
)
opts = parser.parse_args()

# Set the seed value
torch.manual_seed(opts.seed)
torch.cuda.manual_seed(opts.seed)

# Create output folder if it does not exist
if not os.path.exists(opts.output_folder):
    os.makedirs(opts.output_folder)

# Load experiment setting
config = get_config(opts.config)

# Setup model and data loader
config["vgg_model_path"] = opts.output_path

# Set Style dimension

trainer = MUNIT_Trainer(config)

# Load the model (here we currently only load the latest model architecture: one single style)
# try:
state_dict = torch.load(opts.checkpoint)
trainer.gen.load_state_dict(state_dict["2"])
# except:
#    sys.exit("Cannot load the checkpoints")

# Send the trainer to cuda
trainer.cuda()
trainer.eval()

# Set param new_size
new_size = config["new_size"]

# Define the list of non-flooded images
list_non_flooded = glob.glob(opts.input + "*")

list_non_flooded = sorted_nicely(list_non_flooded)
# Define list of masks:

list_masks = glob.glob(opts.mask_dir + "*")

list_masks = sorted_nicely(list_masks)

if len(list_non_flooded) != len(list_masks):
    sys.exit("Image list and mask list differ in length")


# Assert there are some elements inside
if len(list_non_flooded) == 0:
    sys.exit("Image list is empty. Please ensure opts.input ends with a /")



# Inference
with torch.no_grad():
    # Define the transform to infer with the generator
    transform = transforms.Compose(
        [
            #transforms.Resize((new_size, new_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    mask_transform = transforms.Compose(
        [transforms.ToTensor(),]
    )

    for i in tq.tqdm(range(len(list_non_flooded))):

        # Define image path
        path_xa = list_non_flooded[i]

        # Mask stuff

        mask = Image.open(list_masks[i])
        # process
        mask = Variable(mask_transform(mask).cuda())

        # Make mask binary
        mask_thresh = (torch.max(mask) - torch.min(mask)) / 2.0
        mask = (mask > mask_thresh).float()
        mask = mask[0].unsqueeze(0).unsqueeze(0)


    
        # Load and transform the non_flooded image
        x_a = Variable(transform(Image.open(path_xa).convert("RGB")).unsqueeze(0).cuda())
        if opts.save_input:
            inputs = (x_a + 1) / 2.0
            path = os.path.join(opts.output_folder, "{:03d}input.jpg".format(i))
            vutils.save_image(inputs.data, path, padding=0, normalize=True)

        if opts.save_mask:
            path = os.path.join(opts.output_folder, "{:03d}mask.jpg".format(i))
            # overlay mask onto image
            save_m_a = x_a - (x_a * mask.repeat(1, 3, 1, 1)) + mask.repeat(1, 3, 1, 1)
            vutils.save_image(save_m_a, path, padding=0, normalize=True)

        # Extract content and style
        # x_a_augment = torch.cat([x_a, mask], dim=1)

        # ----------------------------------------------------------
        # MASK SMOOTHING
        mask = mask.squeeze().detach().cpu().numpy()
        mask = cv.normalize(
            mask, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F
        )
        mask = mask.astype(dtype=np.uint8)
        ret, thresh = cv.threshold(mask, 127, 255, cv.THRESH_BINARY)
        contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        # Find largest contour
        max_area = 0
        max_idx = -1
        for j, cnt in enumerate(contours):
            if cv.contourArea(contours[j]) > max_area:  # just a condition
                max_idx = j
                max_area = cv.contourArea(contours[j])
        hyp_length = math.sqrt(mask.shape[-2] ** 2 + mask.shape[-1] ** 2)
        cnt = contours[max_idx]
        smooth_mask = np.zeros(mask.shape)
        for x in range(mask.shape[0]):
            for y in range(mask.shape[1]):
                dist = cv.pointPolygonTest(cnt, (y, x), True)
                norm_dist = dist / hyp_length
                if norm_dist < 0:
                    norm_dist = -norm_dist
                    mask_value = int(255 * math.exp(-400 * norm_dist))
                    smooth_mask[x, y] = mask_value
        smooth_mask = smooth_mask + mask
        smooth_mask = torch.tensor(smooth_mask, device="cuda").float()
        smooth_mask = smooth_mask.unsqueeze(0) / 255.0
        mask = smooth_mask
        # ----------------------------------------------------------


        latent_size1 = config["new_size"] // (2 ** config["gen"]["n_downsample"])
         
        latent_size2 = mask.shape[-1] // (2 ** config["gen"]["n_downsample"])

        z = (
            torch.empty(1, config["gen"]["dim"], latent_size1, latent_size2,)
            .normal_(mean=0, std=1.0)
            .cuda()
        )
        x_a_masked = x_a * (1.0 - mask)

        x_ab = trainer.gen(z, x_a_masked)

        # Denormalize .Normalize(0.5,0.5,0.5)...
        outputs = (x_ab + 1) / 2.0

        # Define output path
        path = os.path.join(opts.output_folder, "{:03d}output.jpg".format(i))

        # Save image
        vutils.save_image(outputs.data, path, padding=0, normalize=True)




        """
        # ----------------------------------------------------------
        # MASK SMOOTHING
        mask = mask.squeeze().detach().cpu().numpy()
        mask = cv.normalize(
            mask, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F
        )
        mask = mask.astype(dtype=np.uint8)
        ret, thresh = cv.threshold(mask, 127, 255, cv.THRESH_BINARY)
        contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        # Find largest contour
        max_area = 0
        max_idx = -1
        for j, cnt in enumerate(contours):
            if cv.contourArea(contours[j]) > max_area:  # just a condition
                max_idx = j
                max_area = cv.contourArea(contours[j])
        hyp_length = math.sqrt(mask.shape[-2] ** 2 + mask.shape[-1] ** 2)
        cnt = contours[max_idx]
        smooth_mask = np.zeros(mask.shape)
        for x in range(mask.shape[0]):
            for y in range(mask.shape[1]):
                dist = cv.pointPolygonTest(cnt, (y, x), True)
                norm_dist = dist / hyp_length
                if norm_dist < 0:
                    norm_dist = -norm_dist
                    mask_value = int(255 * math.exp(-400 * norm_dist))
                    smooth_mask[x, y] = mask_value
        smooth_mask = smooth_mask + mask
        smooth_mask = torch.tensor(smooth_mask, device="cuda").float()
        smooth_mask = smooth_mask.unsqueeze(0) / 255.0
        mask = smooth_mask
        # ----------------------------------------------------------
        """

