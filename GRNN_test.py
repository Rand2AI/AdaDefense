# -*-coding:utf-8-*-

import time, datetime, random, argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

from backbone.Model import build_leakage_model
from utils.GRNN_Generator import generator
from utils.Defence_utils import *

from utils import *

torch.set_default_tensor_type("torch.cuda.FloatTensor")

np.random.seed(999)
torch.manual_seed(999)
torch.cuda.manual_seed_all(999)


class LeNet2(nn.Module):
    def __init__(self, channel=3, hideen=768, num_classes=10):
        super(LeNet2, self).__init__()
        self.body1 = nn.Sequential(
            nn.Conv2d(channel, 6, kernel_size=5),
            nn.BatchNorm2d(6),
            nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
        )
        self.body2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5),
            nn.BatchNorm2d(16),
            nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
        )
        self.body3 = nn.Sequential(
            nn.Conv2d(16, 120, kernel_size=5),
            nn.Sigmoid(),
        )
        self.fc = nn.Sequential(nn.Linear(120, 84), nn.Sigmoid(), nn.Linear(84, num_classes))

    def forward(self, x):
        out = self.body1(x)
        out = self.body2(out)
        out = self.body3(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


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
parser.add_argument("--gpu", "-g", type=int, default=0, help="gpu id")
parser.add_argument(
    "--with_ad",
    "-a",
    type=int,
    default=1,
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


def run():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device0 = 0
    device1 = 0
    batchsize = 1
    save_img_flag = True
    Iteration = 20000
    num_exp = 20

    g_in = 1024
    plot_num = 30
    loss_mode = ["l2", "wd", "tv"]
    loss_set = ["mse", "l1", "l2", "wd", "tv", "swd", "gswd", "mswd", "dswd", "mgswd", "dgswd", "csd"]
    dataset = args.dataset  # mnist cifar10 cifar100 imagenet
    net_name = args.network  # lenet res20 res18
    if net_name == "lenet" or net_name == "res20":
        shape_img = (32, 32)
    elif net_name == "res18":
        shape_img = (256, 256)
    else:
        raise ValueError("Invalid network name")
    # AdaDefense
    with_ad = args.with_ad
    ad_opt = "adam"  # adam, adagrad, yogi

    save_path = f"/home/hans/WorkSpace/Models/AdaDefense/GRNN/GRNN-{net_name}-{dataset}-{shape_img[0]}-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    if with_ad:
        save_path += f"-with_ad"
    else:
        save_path += f"-no_ad"
    save_img_path = save_path + "/saved_img/"

    log_path = save_path + "/Log/"
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    dst, num_classes, channel, hidden = GRNN_gen_dataset(dataset, shape_img)
    tp = transforms.Compose([transforms.ToPILImage()])
    criterion = nn.CrossEntropyLoss().cuda(device1)
    print(f'{str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))}: {save_path}')
    G_train_loader = iter(torch.utils.data.DataLoader(dst, batch_size=batchsize, shuffle=False))
    for idx_net in range(num_exp):
        # train_tfLogger = TFLogger(f'{save_path}/tfrecoard-exp-{str(idx_net).zfill(4)}')
        print(
            f'{str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))}: Running {idx_net+1}|{num_exp} experiment'
        )
        net = build_leakage_model(net_name, num_classes)
        net = net.cuda(device1)

        Gnet = generator(
            num_classes=num_classes, channel=channel, shape_img=shape_img[0], batchsize=batchsize, g_in=g_in
        ).cuda(device0)

        gt_data, gt_label = next(G_train_loader)
        gt_data, gt_label = gt_data.cuda(device1), gt_label.cuda(device1)

        pred = net(gt_data)
        y = criterion(pred, gt_label)

        if with_ad:
            dy_dx = ada_defense(y, net, 0, create_graph=False)
        else:
            dy_dx = list(torch.autograd.grad(y, net.parameters()))
        new_dy_dx = split_gradient(net, dy_dx)
        flatten_true_g = flatten_gradients(new_dy_dx)

        G_ran_in = torch.randn(batchsize, g_in).cuda(device0)
        iter_bar = tqdm(
            range(Iteration),
            total=Iteration,
            desc=f'{str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))}',
            ncols=150,
        )
        history = []
        history_l = []
        tf_his = []
        tf_his_i = []
        G_key = (
            torch.tensor(np.array([random.random() for _ in range(key_length)]))
            .float()
            .cuda(device1)
            .requires_grad_(gen_key)
        )
        gt_G_key = G_key.clone().detach()
        G_optimizer = torch.optim.RMSprop(Gnet.parameters(), lr=0.0001, momentum=0.99)
        # G_optimizer = torch.optim.Adam(Gnet.parameters(), lr=0.0001)
        for iters in iter_bar:
            Gout, Glabel = Gnet(G_ran_in)
            Gout, Glabel = Gout.cuda(device1), Glabel.cuda(device1)
            G_optimizer.zero_grad()
            Gpred = net(Gout)
            # Gloss = -torch.mean(torch.sum(Glabel * torch.log(torch.softmax(Gpred, 1)), dim=-1))
            Gloss = criterion(Gpred, Glabel)

            G_dy_dx = list(torch.autograd.grad(Gloss, net.parameters(), create_graph=True))
            new_G_dy_dx = split_gradient(net, G_dy_dx)
            flatten_fake_g = flatten_gradients(new_G_dy_dx).cuda(device1)
            loss_list = []
            for loss_name in loss_mode:
                loss_list.append(
                    loss_f(
                        loss_name=loss_name,
                        flatten_fake_g=flatten_fake_g,
                        flatten_true_g=flatten_true_g,
                        device1=device1,
                        Gout=Gout,
                    )
                )
            grad_diff = sum(loss_list)
            grad_diff.backward()
            G_optimizer.step()
            iter_bar.set_postfix(
                total_loss=np.round(grad_diff.item(), 8),
                mses_img=round(torch.mean(abs(Gout - gt_data)).item(), 8),
                wd_img=round(wasserstein_distance(Gout.view(1, -1), gt_data.view(1, -1)).item(), 8),
            )

            if iters % int(Iteration / plot_num) == 0:
                tf_his.append([tp(Gout[imidx].detach().cpu()) for imidx in range(batchsize)])
                tf_his_i.append(iters)

            if iters % int(Iteration / plot_num) == 0:
                history.append([tp(Gout[imidx].detach().cpu()) for imidx in range(batchsize)])
                history_l.append([Glabel.argmax(dim=1)[imidx].item() for imidx in range(batchsize)])
            del Gloss, G_dy_dx, flatten_fake_g, grad_diff
        for imidx in range(batchsize):
            plt.figure(figsize=(12, 8))
            plt.subplot(plot_num // 10, 10, 1)
            plt.imshow(tp(gt_data[imidx].cpu()))
            for i in range(min(len(history), plot_num - 1)):
                plt.subplot(plot_num // 10, 10, i + 2)
                plt.imshow(history[i][imidx])
                plt.title("l=%d" % (history_l[i][imidx]))
                # plt.title('i=%d,l=%d' % (history_iters[i], history_l[i][imidx]))
                plt.axis("off")
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            if save_img_flag:
                true_path = save_img_path + f"true_data/exp{str(idx_net).zfill(4)}/"
                fake_path = save_img_path + f"fake_data/exp{str(idx_net).zfill(4)}/"
                if not os.path.exists(true_path) or not os.path.exists(fake_path):
                    os.makedirs(true_path)
                    os.makedirs(fake_path)
                tp(gt_data[imidx].cpu()).save(true_path + f"/{imidx}_{gt_label[imidx].item()}.png")
                history[i][imidx].save(fake_path + f"/{imidx}_{Glabel.argmax(dim=1)[imidx].item()}.png")
            plt.savefig(
                save_path
                + "/exp:%04d-imidx:%03d-tlabel:%d-Glabel:%d.png"
                % (idx_net, imidx, gt_label[imidx].item(), Glabel.argmax(dim=1)[imidx].item())
            )
            plt.close()

        # train_tfLogger.images_summary([Glabel.argmax(dim=1)[imidx].item() for imidx in range(batchsize)], tf_his, tf_his_i)
        torch.cuda.empty_cache()
        history.clear()
        history_l.clear()
        tf_his.clear()
        tf_his_i.clear()
        iter_bar.close()
        # train_tfLogger.close()
        print("----------------------")


def save_img():
    import pickle

    data_path = "/home/hans/WorkSpace/Data/cifar100/"
    with open(data_path + "/cifar-100-python/train", mode="rb") as file:
        # 数据集在当脚本前文件夹下
        data_dict = pickle.load(file, encoding="bytes")
        data = list(data_dict[b"data"])
        labels = list(data_dict[b"fine_labels"])
    with open(data_path + "/cifar-100-python/meta", mode="rb") as file:
        data_dict = pickle.load(file, encoding="bytes")
        label_name = list(data_dict[b"fine_label_names"])
    img = np.reshape(data, [-1, 3, 32, 32])
    save_path = data_path + "/raw_img/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for i in range(100):
        r = img[i][0]
        g = img[i][1]
        b = img[i][2]
        ir = Image.fromarray(r)
        ig = Image.fromarray(g)
        ib = Image.fromarray(b)
        rgb = Image.merge("RGB", (ir, ig, ib))
        name = str(i) + "-" + label_name[labels[i]].decode() + ".png"
        rgb.save(save_path + name, "PNG")


if __name__ == "__main__":
    run()
