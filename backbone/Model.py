# -*-coding:utf-8-*-

import torch
from backbone import *
from torchvision.models.resnet import *
from torchvision.models.vgg import *


def build_model(num_classes, config, act="relu"):
    if config["NETWORK"]["BACKBONE"] == "lenet":
        net = lenet(channel=3, hidden=768, num_classes=num_classes)
    elif config["NETWORK"]["BACKBONE"] == "vgg16":
        net = vgg16(num_classes=num_classes)
    elif config["NETWORK"]["BACKBONE"] == "resnet":
        if config["NETWORK"]["LAYER_NUMBER"] == 18:
            net = resnet18(pretrained=False)
            num_ftrs = net.fc.in_features
            net.fc = torch.nn.Linear(num_ftrs, num_classes)
        elif config["NETWORK"]["LAYER_NUMBER"] == 34:
            net = resnet34(pretrained=False)
            num_ftrs = net.fc.in_features
            net.fc = torch.nn.Linear(num_ftrs, num_classes)
        elif config["NETWORK"]["LAYER_NUMBER"] == 20:
            net = resnet20(num_classes=num_classes)
        elif config["NETWORK"]["LAYER_NUMBER"] == 32:
            net = resnet32(num_classes=num_classes)
        else:
            raise Exception("Wrong ResNet Layer Number.")
    else:
        raise Exception("Wrong Backbone Name.")
    return net


def build_leakage_model(net_name, num_classes, act="sigmoid"):
    if net_name == "res18":
        net = resnet18_leak(num_classes=num_classes)
    elif net_name == "res34":
        net = resnet34_leak(num_classes=num_classes)
    elif net_name == "res20":
        net = resnet20(num_classes=num_classes, act=act)
    elif net_name == "res32":
        net = resnet32(num_classes=num_classes, act=act)
    elif net_name == "vgg16":
        net = vgg16_leak(num_classes=num_classes)
    elif net_name == "lenet":
        net = lenet(channel=3, hidden=768, num_classes=num_classes)
    else:
        net = None
        exit("Wrong network name")
    return net
