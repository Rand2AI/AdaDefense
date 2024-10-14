import os
from backbone import *
from utils import get_config

config = get_config(os.path.dirname(os.path.realpath(__file__)))


def count_parameters(model):
    model_dict = model.state_dict()
    total = 0
    kl = 0
    for k, v in model_dict.items():
        total += v.nelement()
        if "scale" in k or "shift" in k:
            kl += v.nelement()
    return total, kl, kl / total


def main():
    num_classes = 100
    act = "relu"

    LeNet_total, LeNet_l, LeNet_proprotion = count_parameters(LeNet)
    resnet20_total, resnet20_l, resnet20_proprotion = count_parameters(resnet20)
    resnet32_total, resnet32_l, resnet32_proprotion = count_parameters(resnet32)
    resnet18_total, resnet18_l, resnet18_proprotion = count_parameters(resnet18)
    resnet34_total, resnet34_l, resnet34_proprotion = count_parameters(resnet34)
    vgg16_total, vgg16_l, vgg16_proprotion = count_parameters(vgg16)
    print("Done")


if __name__ == "__main__":
    main()
