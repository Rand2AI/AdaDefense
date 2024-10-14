# -*-coding:utf-8-*-
import os, torch, argparse

from utils import get_config
from methods import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        "-m",
        type=str,
        default="AdaDefense",
        help="Normal, FedAvg, FedKL, EWWA, AdaDefense",
    )
    parser.add_argument(
        "--optimizer",
        "-o",
        type=str,
        default="adam",
        help="adam, adagrad, yogi, optimizer for AdaDefense",
    )
    parser.add_argument(
        "--network",
        "-n",
        type=str,
        default="resnet",
        help="lenet, resnet, vgg16",
    )
    parser.add_argument(
        "--layer",
        "-l",
        type=int,
        default=20,
        help="18, 34, 20, 32 for resnet only",
    )
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        default="cifar10",
        help="mnist, cifar10, cifar100",
    )
    parser.add_argument("--gpu", "-g", type=int, default=0, help="gpu id")
    parser.add_argument("--batchsize", "-b", type=int, default=32, help="batchsize")
    parser.add_argument("--rounds", "-r", type=int, default=2000, help="rounds")
    parser.add_argument("--iid", type=int, default=1, help="1 for iid, 0 for non-iid")
    parser.add_argument("--debug", type=int, default=0, help="debug mode")
    args = parser.parse_args()

    config = get_config(args, os.path.dirname(os.path.realpath(__file__)))
    if config["DEVICE"]["DEVICE_TOUSE"] == "GPU":
        seed = 0
        torch.manual_seed(seed)  # sets the seed for generating random numbers.
        torch.cuda.manual_seed(seed)
        # Sets the seed for generating random numbers for the current GPU.
        # It’s safe to call this functionif CUDA is not available;
        # in that case, it is silently ignored.
        torch.cuda.manual_seed_all(seed)
        # Sets the seed for generating random numbers on all GPUs.
        # It’s safe to call this function if CUDA is not available;
        # in that case, it is silently ignored.

        if seed == 0:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        torch.multiprocessing.set_start_method("spawn")
        # torch.set_default_tensor_type('torch.cuda.FloatTensor')
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(x) for x in config["DEVICE"]["DEVICE_GPUID"]])
    else:
        raise Exception("Current version does not support CPU yet.")
    eval(config["METHODS"])(config)
