"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from __future__ import print_function
from utils import get_config, sorted_nicely
from trainer import MUNIT_Trainer
import argparse
from torch.autograd import Variable
import torchvision.utils as vutils
import torch
import os
from torchvision import transforms
from PIL import Image
import tqdm as tq
import imageio
import numpy as np
from pathlib import Path
from data import is_image_file
from datetime import datetime
from pygifsicle import optimize
from time import time


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, help="network configuration file")
parser.add_argument("--input", type=str, help="directory of input images")
parser.add_argument(
    "--mask_dir", type=str, help="directory of masks corresponding to input images"
)
parser.add_argument("--output_folder", type=str, help="output image directory")
parser.add_argument("--checkpoint", type=str, help="checkpoint of generator")

parser.add_argument("--seed", type=int, default=10, help="random seed")

parser.add_argument(
    "--synchronized",
    action="store_true",
    help="whether use synchronized style code or not",
)
parser.add_argument(
    "--save_input",
    action="store_true",
    help="whether use synchronized style code or not",
)
parser.add_argument(
    "--output_path",
    type=str,
    default=".",
    help="path for logs, checkpoints, and VGG model weight",
)
parser.add_argument(
    "--save_mask", action="store_true", help="whether to save mask or not"
)
parser.add_argument(
    "--gif_length", type=int, default=5, help="Number of inferences per gif"
)
parser.add_argument("--gain", type=float, default=0.1, help="Scale of the perturbation")
parser.add_argument("--fps", type=int, default=15, help="GIF frames per seconds")
parser.add_argument("--batch_size", type=int, default=4, help="Inference batch_size")
parser.add_argument(
    "--smallest_side",
    type=int,
    default=512,
    help="Keep aspect ratio with smallest side at least smallest_side",
)
opts = parser.parse_args()

# Example:
# python animate.py --config /network/tmp1/ccai/checkpoints/sun/outputs/nopatch_twodiscrims_percept_segmask_condmask_deep_idea2_256/config.yaml --input /network/home/schmidtv/testspade/ --mask_dir /network/home/schmidtv/testspademasks/ --output_folder /network/home/schmidtv/testspaderesults/ --checkpoint /network/tmp1/ccai/checkpoints/sun/outputs/nopatch_twodiscrims_percept_segmask_condmask_deep_idea2_256/checkpoints/gen_00085000.pt --gif_size 20 --batch_size 4 --gain 0.1


if __name__ == "__main__":

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

    # Load the model
    # (here we currently only load the latest model architecture: one single style)
    state_dict = torch.load(opts.checkpoint)
    trainer.gen.load_state_dict(state_dict["2"])

    # Send the trainer to cuda
    trainer.cuda()
    trainer.eval()

    # Set param new_size
    new_size = config["new_size"]

    # Define the list of non-flooded images
    list_non_flooded = [
        str(im) for im in Path(opts.input).resolve().glob("*") if is_image_file(im)
    ]

    list_non_flooded = sorted_nicely(list_non_flooded)
    # Define list of masks:

    list_masks = [
        str(im) for im in Path(opts.mask_dir).resolve().glob("*") if is_image_file(im)
    ]

    list_masks = sorted_nicely(list_masks)

    assert len(list_non_flooded) == len(
        list_masks
    ), "Image list and mask list differ in length"

    # Assert there are some elements inside
    assert list_non_flooded, "Image list is empty"

    output_folder = Path(opts.output_folder).resolve()
    output_folder.mkdir(parents=True, exist_ok=True)

    run_id = str(datetime.now())[:19].replace(" ", "_")

    images = {}

    gain = opts.gain
    gl = opts.gif_length
    ss = opts.smallest_side
    bs = opts.batch_size

    # Inference
    with torch.no_grad():
        for i, im in enumerate(list_non_flooded):
            print("Inferring", str(im))
            iters = gl // bs
            if gl % bs != 0:
                iters += 1

            images[str(im)] = []
            file_id = Path(im).stem
            path_xa = list_non_flooded[i]
            output_path = output_folder / "{}-{}.gif".format(
                file_id, str(gain).replace(".", "")
            )

            original_mask = Image.open(list_masks[i])
            pil_im = Image.open(im).convert("RGB")
            w, h = pil_im.size
            # Stay close to aspect ratio with max(w, h) = new_size (600px for instance)
            # But model downsamples and upsamples * 8 so each need to be a multiple of 8
            m = min((h, w))
            ratio = ss / m
            new_h = int(ratio * h) // 8 * 8
            new_w = int(ratio * w) // 8 * 8

            # Define the transform to infer with the generator
            transform = transforms.Compose(
                [
                    transforms.Resize((new_h, new_w)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )
            mask_transform = transforms.Compose(
                [transforms.Resize((new_h, new_w)), transforms.ToTensor()]
            )

            # Make original_mask binary
            original_mask = Variable(mask_transform(original_mask).cuda())
            mask_thresh = (torch.max(original_mask) - torch.min(original_mask)) / 2.0
            original_mask = (original_mask > mask_thresh).float()
            # Expand channel and batch dims
            original_mask = original_mask[0].unsqueeze(0).unsqueeze(0)
            # Create one mask per frame in the GIF
            full_mask = original_mask.repeat(bs * iters, 1, 1, 1)
            # Create one mask per image in a batch
            batch_mask = original_mask.repeat(bs, 1, 1, 1)
            # Perturn the non-zero values of the masks
            perturbation = torch.randn_like(full_mask) * full_mask
            noisy_mask = full_mask + perturbation * opts.gain

            # Create input image for all frames in the GIF
            x_a_v = Variable(transform(pil_im).unsqueeze(0).cuda())
            full_x_a = x_a_v.repeat(bs * iters, 1, 1, 1)
            # Save the background (= part of the image that is outside the mask)
            background = 255 * (x_a_v.repeat(bs, 1, 1, 1) + 1) / 2 * (1 - batch_mask)
            # Create input to the model
            full_input = torch.cat([full_x_a, full_mask], dim=1)

            # Create GIF
            gif_start = time()
            # Translate all frames
            for j in tq.tqdm(range(iters)):
                batch_input = full_input[j * bs : (j + 1) * bs]
                batch_noisy_mask = noisy_mask[j * bs : (j + 1) * bs]
                c_a = trainer.gen.encode(batch_input, 1)
                # Perform cross domain translation
                x_ab = trainer.gen.decode(c_a, batch_noisy_mask, 2)
                # Scale as an image
                flooded = 255 * (x_ab + 1) / 2.0
                # Isolate flood
                flood = flooded * batch_mask
                # Store as uint8 in the `images` dict
                outputs = list(
                    (background + flood)
                    .detach()
                    .to(torch.uint8)
                    .cpu()
                    .numpy()
                    .transpose(0, 2, 3, 1)
                )
                images[str(im)] += outputs
            # Save list of frames as GIF
            imageio.mimsave(
                output_path, images[str(im)], subrectangles=True, duration=0.15
            )
            print("GIF creation duration: {:.3f}\n".format(time() - gif_start))
