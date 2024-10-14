# -*- coding: utf-8 -*-
import json, sys
import torch.nn as nn
from utils.Configer import get_config
from utils.Data import gen_dataset, split_iid_data, split_noniid_data
from utils.Optimizer import set_optimizer
from utils.Evaluator import evaluator, tester
from utils.Trainer import trainer
from utils.FedAvg_Weight import fedavg_weight
from utils.Client import local_update

# from utils.Logger import TFLogger


class Logger(object):
    def __init__(self, filename="default.log", stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def save_args_as_json(FLconfig, path):
    with open(str(path), "w") as f:
        json.dump(FLconfig, f, indent=4)


def weight_zero_init(m):
    if isinstance(m, nn.Linear):
        nn.init.constant_(m.weight, 0)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.constant_(m.weight, 0)
        try:
            nn.init.constant_(m.bias, 0)
        except AttributeError:
            pass
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 0)
        nn.init.constant_(m.bias, 0)
