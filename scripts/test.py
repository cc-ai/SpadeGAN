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
try:
    state_dict = torch.load(opts.checkpoint)
    trainer.gen.load_state_dict(state_dict["2"])
except:
    sys.exit("Cannot load the checkpoints")

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
            transforms.Resize((new_size, new_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    mask_transform = transforms.Compose([transforms.Resize((new_size, new_size)), transforms.ToTensor(),])


    for j in tq.tqdm(range(len(list_non_flooded))):

        # Define image path
        path_xa = list_non_flooded[j]

        # Mask stuff

        mask = Image.open(list_masks[j])
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
            path = os.path.join(opts.output_folder, "{:03d}input.jpg".format(j))
            vutils.save_image(inputs.data, path, padding=0, normalize=True)

        if opts.save_mask:
            path = os.path.join(opts.output_folder, "{:03d}mask.jpg".format(j))
            #overlay mask onto image
            save_m_a = x_a - (x_a * mask.repeat(1, 3, 1, 1)) + mask.repeat(1, 3, 1, 1)
            vutils.save_image(save_m_a, path, padding=0, normalize=True)          

        # Extract content and style
        c_a = trainer.gen.encode(x_a, 1)

        # Perform cross domain translation
        x_ab = trainer.gen.decode(c_a, mask, 2)

        # Denormalize .Normalize(0.5,0.5,0.5)...
        outputs = (x_ab + 1) / 2.0

        # Define output path
        path = os.path.join(opts.output_folder, "{:03d}output.jpg".format(j))

        # Save image
        vutils.save_image(outputs.data, path, padding=0, normalize=True)
