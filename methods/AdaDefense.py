# -*- coding: utf-8 -*-
import numpy as np
import os, datetime, copy, shutil

import torch
from torch.utils.data import DataLoader

from backbone.Model import build_model
from utils import *


def AdaDefense(config):
    rounds = config["TRAIN"]["ROUNDS"]
    batchsize = config["TRAIN"]["BATCH_SIZE"]

    train_dataset, test_dataset, img_size, num_classes = gen_dataset(
        config["DATA"]["TRAIN_DATA"],
        config["DATA"]["IMG_SIZE"],
        config["DATA"]["DATA_ROOT"],
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batchsize,
        shuffle=False,
        num_workers=4 * len(config["DEVICE"]["DEVICE_GPUID"]),
        pin_memory=True,
    )
    # select client
    selected_client_num = max(int(config["FED"]["FRACTION"] * config["FED"]["CLIENTS_NUM"]), 1)
    print(f"{selected_client_num} of {config['FED']['CLIENTS_NUM']} clients are selected.")
    idxs_client = np.random.choice(range(config["FED"]["CLIENTS_NUM"]), selected_client_num, replace=False)

    # IID or Non-IID
    if config["DATA"]["IS_IID"]:
        print("IID data")
        dict_users = split_iid_data(train_dataset, config["FED"]["CLIENTS_NUM"])
    else:
        print("Non-IID data")
        dict_users = split_noniid_data(train_dataset, config["FED"]["CLIENTS_NUM"])

    model_global = build_model(num_classes, config)
    print("model builded.")

    if config["DEVICE"]["DEVICE_TOUSE"] == "GPU":
        model_global.cuda()
        if len(config["DEVICE"]["DEVICE_GPUID"]) > 1:
            model_global = torch.nn.DataParallel(
                model_global,
                device_ids=list(range(len(config["DEVICE"]["DEVICE_GPUID"]))),
            )
    if config["TRAIN"]["FINETUNE"]:
        checkpoint = torch.load(config["TRAIN"]["WEIGHT_TOLOAD"])
        model_global.load_state_dict(checkpoint)

    model_zero_weihgt = copy.deepcopy(model_global)

    fedadp_obj = AdaDefenseObj(
        model_global,
        model_zero_weihgt.apply(weight_zero_init).state_dict(),
        config["FED"]["OPTIMIZER"],
    )
    print("Aggregation method loaded.")
    modelID = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if config["NETWORK"]["BACKBONE"] == "resnet":
        model_name = f"{config['NETWORK']['BACKBONE']}{config['NETWORK']['LAYER_NUMBER']}"
    else:
        model_name = config["NETWORK"]["BACKBONE"]
    if "lenet" in model_name:
        model_name = "lenet"
    if config["DATA"]["IS_IID"]:
        save_path = (
            f"{config['TRAIN']['SAVE_ROOT']}/{config['NAME']}/{config['METHODS']}/"
            f"{config['METHODS']}-{config['FED']['OPTIMIZER']}-{model_name}-{config['DATA']['TRAIN_DATA']}-"
            f"iid-B{str(batchsize).zfill(3)}-{modelID}"
        )
    else:
        save_path = (
            f"{config['TRAIN']['SAVE_ROOT']}/{config['NAME']}/{config['METHODS']}/"
            f"{config['METHODS']}-{config['FED']['OPTIMIZER']}-{model_name}-{config['DATA']['TRAIN_DATA']}-"
            f"noniid-B{str(batchsize).zfill(3)}-{modelID}"
        )
    print(f"\n>>>>>>>>>>>>> {save_path}\n")
    if not config["DEBUG"]:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # save arguments to local
        args_json_path = save_path + "/args.json"
        save_args_as_json(config, args_json_path)

        # client model path
        model_path = {}
        for idx in idxs_client:
            model_path[idx] = save_path + f"/Model/Client_{idx}/"
            if not os.path.exists(model_path[idx]):
                os.makedirs(model_path[idx])

    locals = {
        idx: local_update(
            config=config,
            client_idx=idx,
            dataset=train_dataset,
            data_idxs=dict_users[idx],
            model=copy.deepcopy(model_global),
            test_loader=test_loader,
        )
        for idx in idxs_client
    }
    print("local client set up.")
    test_best = 0
    local_best = [0, 0, 0]
    for rd in range(rounds):
        print("\n")
        print("-" * 100)
        print(f"[Round: {rd}/{rounds}]")
        # train
        loss_locals = []
        acc_locals = []
        w_locals = []
        for idx in idxs_client:
            print("-" * 10)
            weight_local, loss_local, acc_local = locals[idx].train(model_global.state_dict())
            if np.mean(acc_local) > local_best[idx] and not config["DEBUG"]:
                local_best[idx] = np.mean(acc_local)
                torch.save(
                    weight_local,
                    model_path[idx]
                    + f"/client:{idx}-epoch:{str(rd).zfill(3)}-trn_loss:{np.round(loss_local, 4)}-trn_acc:{np.round(acc_local, 4)}-{modelID}.pth",
                )
            # _, _ = evaluator(locals[idx].model, test_loader, nn.CrossEntropyLoss(), batchsize)
            new_local_w = fedadp_obj.ad_weight(model_global.state_dict(), weight_local)
            w_locals.append(new_local_w)
            loss_locals.append(loss_local)
            acc_locals.append(acc_local)

        fedadp_obj.t = rd
        model_global.load_state_dict(fedavg_weight(w_locals))

        # test
        test_loss_avg, test_acc_avg = evaluator(model_global, test_loader, nn.CrossEntropyLoss(), batchsize)
        print(save_path)
        print(
            f"Local train loss: {np.mean(loss_locals)}, acc: {np.mean(acc_locals)}\n"
            f"Global test loss: {test_loss_avg}, acc: {test_acc_avg}"
        )
        if np.mean(test_acc_avg) > test_best and not config["DEBUG"]:
            test_best = np.mean(test_acc_avg)
            torch.save(
                model_global.state_dict(),
                f'{save_path}/{modelID}-{config["METHODS"]}-round:{str(rd).zfill(3)}-tst_loss:{np.round(np.mean(test_loss_avg), 4)}-tst_acc:{np.round(np.mean(test_acc_avg), 4)}-test_best.pth',
            )
    if not config["DEBUG"]:
        log_name = f'{config["METHODS"]}-{config["NETWORK"]["BACKBONE"]}{config["NETWORK"]["LAYER_NUMBER"]}-{config["DATA"]["TRAIN_DATA"]}.txt'
        if config["DATA"]["IS_IID"]:
            log_name = log_name.replace(".txt", "-iid.txt")
        else:
            log_name = log_name.replace(".txt", "-noniid.txt")
        shutil.move(
            f"/home/hans/WorkSpace/AdaDefense-FL/{log_name}",
            f"{save_path}/{log_name}",
        )


class AdaDefenseObj(object):
    def __init__(
        self,
        model,
        m_zero,
        optimiser,
        scale=0.001,
        beta1=0.9,
        beta2=0.999,
        epislon=1e-8,
        test=False,
    ):
        self.model = model
        self.optimiser = optimiser
        self.m_zero = m_zero
        self.scale = scale
        self.beta1 = beta1
        self.beta2 = beta2
        self.epislon = epislon
        self.m = copy.deepcopy(self.m_zero)
        self.v = copy.deepcopy(self.m_zero)
        if test:
            self.m.requires_grad = False
            self.v.requires_grad = False
        self.t = 0

    def adam(self, g):
        scale = self.scale * (1 - self.beta2 ** (self.t + 1)) ** 0.5 / (1 - self.beta1 ** (self.t + 1))
        # print(f"scale: {scale}")
        result = copy.deepcopy(g)
        # result = copy.deepcopy(self.m_zero)
        param = self.model.named_parameters()
        # for k in self.m.keys(): # all parameters
        for k, v in param:  # only trainable parameters
            if "running_mean" in k or "running_var" in k or "num_batches_tracked" in k:
                continue
            # if "bn" in k:
            #     continue
            self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * g[k]
            self.v[k] = self.beta2 * self.v[k] + (1 - self.beta2) * (g[k] * g[k])
            result[k] = scale * self.m[k] / (torch.sqrt(self.v[k]) + self.epislon)
        return result

    def adagrad(self, g):
        scale = self.scale * (1 - self.beta2 ** (self.t + 1)) ** 0.5 / (1 - self.beta1 ** (self.t + 1))
        # print(f"scale: {scale}")
        result = copy.deepcopy(self.m_zero)
        for k in self.m.keys():
            self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * g[k]
            self.v[k] = self.v[k] + (g[k] * g[k])
            result[k] = scale * self.m[k] / (torch.sqrt(self.v[k]) + self.epislon)
        return result

    def yogi(self, g):
        scale = self.scale * (1 - self.beta2 ** (self.t + 1)) ** 0.5 / (1 - self.beta1 ** (self.t + 1))
        # print(f"scale: {scale}")
        result = copy.deepcopy(self.m_zero)
        for k in self.m.keys():
            self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * g[k]
            self.v[k] = self.v[k] - (1 - self.beta2) * (g[k] * g[k]) * torch.sign(self.v[k] - (g[k] * g[k]))
            result[k] = scale * self.m[k] / (torch.sqrt(self.v[k]) + self.epislon)
        return result

    def compute_gradient(self, g):
        if self.optimiser == "adam":
            new_g = self.adam(g)
        elif self.optimiser == "adagrad":
            new_g = self.adagrad(g)
        elif self.optimiser == "yogi":
            new_g = self.yogi(g)
        else:
            raise ValueError(f"optimiser {self.optimiser} is not supported")
        return new_g

    def ad_weight(self, w, local_w):
        g = copy.deepcopy(local_w)
        result_w = copy.deepcopy(local_w)
        for k in w.keys():
            g[k] = w[k] - local_w[k]
        new_g = self.compute_gradient(g)
        # convert g to w
        for k in w.keys():
            result_w[k] = w[k] - new_g[k]
        return result_w

    def ad_gradient(self, loss, net, create_graph=False):
        g = torch.autograd.grad(loss, net.parameters(), create_graph=create_graph)
        new_dy_dx_dict = copy.deepcopy(self.m_zero)
        idx = 0
        for name, param in net.named_parameters():
            if param.requires_grad:
                new_dy_dx_dict[name] = g[idx]
                idx += 1
        new_g = self.compute_gradient(new_dy_dx_dict)
        new_dy_dx_list = []
        for v in new_g.values():
            new_dy_dx_list.append(v)
        return new_dy_dx_list
