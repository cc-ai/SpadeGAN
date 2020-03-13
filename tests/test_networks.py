import os
from pathlib import Path
import argparse
import torch
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from scripts.networks import MultiscaleDiscriminator
from tests.run import opts

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dis = MultiscaleDiscriminator(input_nc=3, params=opts["dis"])
    dis = dis.to(device)

    batch_size = 2
    test_image = torch.Tensor(batch_size, 3, 128, 128).uniform_(-1, 1).to(device)

    dis_output = dis(test_image)
    print("Testing discriminator...")
    print("Num discriminators: ", len(dis_output))
    for i in dis_output:
        for j in i:
            print(j.shape)
        print("-------------")
    print("Testing dis losses...")
    real_image = torch.Tensor(batch_size, 3, 128, 128).uniform_(-1, 1).to(device)
    fake_image = torch.Tensor(batch_size, 3, 128, 128).uniform_(-1, 1).to(device)
    dis_loss = dis.calc_dis_loss(fake_image, real_image)
    print("dis loss: ", dis_loss)
    gen_loss = dis.calc_gen_loss(fake_image, real_image)
    print("gen loss: ", gen_loss)

