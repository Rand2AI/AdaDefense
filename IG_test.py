import torch, argparse
import torchvision

import os
from utils.Defence_utils import ada_defense
from backbone.ResNet_cifar import resnet20
from backbone.LeNet import lenet

idx = 6666

parser = argparse.ArgumentParser()
parser.add_argument(
    "--network",
    "-n",
    type=str,
    default="res18",
    help="lenet res20 res18",
)
parser.add_argument(
    "--dataset",
    "-d",
    type=str,
    default="imagenet",
    help="mnist cifar10 cifar100 imagenet",
)
parser.add_argument("--gpu", "-g", type=int, default=1, help="gpu id")
parser.add_argument(
    "--with_ad",
    "-a",
    type=int,
    default=0,
    help="with AdaDefense or not",
)

parser.add_argument(
    "--ad_opt",
    "-o",
    type=str,
    default="adam",
    help="adam adagrad yogi",
)
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

key_length = 1024
dataset = args.dataset  # mnist cifar10 cifar100
net_name = args.network  # lenet res20 res18
with_kl = False
share_key = False
gen_key = False
with_lock_layer = False

if net_name == "lenet" or net_name == "res20":
    shape_img = (32, 32)
elif net_name == "res18":
    shape_img = (256, 256)
else:
    raise ValueError("Invalid network name")

if share_key:
    gen_key = False  # force to False as no need to regress key
root_path = "/home/hans/WorkSpace/Models/FL/AdaDefense/IG/"
# root_path = "/home/hans/WorkSpace/Models/FL/FedKL/IG/"

with_ad = args.with_ad
ad_opt = args.ad_opt  # adam, adagrad, yogi

# img index

from utils import inversefed

setup = inversefed.utils.system_startup()
defs = inversefed.training_strategy("conservative")

if dataset == "imagenet":
    data_path = "/home/hans/WorkSpace/Data/Vision/ILSVRC/2012"
else:
    data_path = f"/home/hans/WorkSpace/Data/Vision/{dataset}"

loss_fn, trainloader, validloader, num_classes = inversefed.construct_dataloaders(
    dataset, defs, shape_img[0], data_path=data_path
)

if net_name == "res18":
    model = torchvision.models.resnet18()
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
elif net_name == "res20":
    model = resnet20(num_classes=num_classes)
elif net_name == "lenet":
    model = lenet(channel=3, hidden=768, num_classes=num_classes)

model.to(**setup)
model.eval()

dm = torch.as_tensor(inversefed.consts.imagenet_mean, **setup)[:, None, None]
ds = torch.as_tensor(inversefed.consts.imagenet_std, **setup)[:, None, None]

img, label = validloader.dataset[idx]
labels = torch.as_tensor((label,), device=setup["device"])
ground_truth = img.to(**setup).unsqueeze(0)
# plot(ground_truth)
print([trainloader.dataset.classes[l] for l in labels])
if not os.path.exists(root_path):
    os.makedirs(root_path)
ture_img_path = f"{root_path}/{idx}_{trainloader.dataset.classes[labels[0]]}_{dataset}_{net_name}_true.png"
fake_img_path = f"{root_path}/{idx}_{trainloader.dataset.classes[labels[0]]}_{dataset}_{net_name}_output.png"
if with_ad:
    ture_img_path = ture_img_path.replace(".png", f"_ad.png")
    fake_img_path = fake_img_path.replace(".png", f"_ad.png")


ground_truth_denormalized = torch.clamp(ground_truth * ds + dm, 0, 1)
torchvision.utils.save_image(ground_truth_denormalized, ture_img_path)

model.zero_grad()
target_loss, _, _ = loss_fn(model(ground_truth), labels)
if with_ad:
    input_gradient = ada_defense(target_loss, model, 0)
else:
    input_gradient = torch.autograd.grad(target_loss, model.parameters())
input_gradient = [grad.detach() for grad in input_gradient]
full_norm = torch.stack([g.norm() for g in input_gradient]).mean()
print(f"Full gradient norm is {full_norm:e}.")

config = dict(
    signed=True,
    boxed=True,
    cost_fn="sim",
    indices="def",
    weights="equal",
    lr=0.1,
    optim="adam",
    restarts=2,
    max_iterations=24_000,
    total_variation=1e-6,
    init="randn",
    filter="median",
    lr_decay=True,
    scoring_choice="loss",
)

rec_machine = inversefed.GradientReconstructor(model, (dm, ds), config, num_images=1)
output, stats = rec_machine.reconstruct(input_gradient, labels, img_shape=(3, shape_img[0], shape_img[0]))

output_denormalized = torch.clamp(output * ds + dm, 0, 1)
torchvision.utils.save_image(output_denormalized, fake_img_path)
