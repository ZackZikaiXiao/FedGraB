# python version 3.7.1
# -*- coding: utf-8 -*-
# 用logits加权的，这个版本有问题
import os
import copy
import numpy as np
import random
import torch
import torch.nn as nn
import pdb
import torch.distributed as dist
import torch.nn.functional as F
from mmdet.utils import get_root_logger
from functools import partial
from torch.nn import init

from tqdm import tqdm
from options import args_parser
from util.update_baseline import *
from util.fedavg import FedAvg, FedAvg_noniid, Weighted_avg_f1
# from util.util import add_noise
from util.dataset import *
from model.build_model import build_model


np.set_printoptions(threshold=np.inf)


class Hook():
    def __init__(self):
        self.m_count = 0    # for debug
        # hook函数中临时变量的保存
        self.input_grad_list = []
        self.output_grad_list = []
        self.gradient = None
        self.gradient_list = []

    def has_gradient(self):
        return self.gradient != None

    def get_gradient(self):
        return self.gradient

    def hook_func_tensor(self, grad):
        grad = copy.deepcopy(grad)
        self.gradient = grad.cpu().numpy().tolist()  # [200, 10] -> [10, 200]
        # print(type(self.gradient))
        # print("tensor hook", self.m_count)
        # print(grad)
        # print(grad.shape)
        self.m_count += 1

    def hook_func_model(self, module, grad_input, grad_output):
        pass
        # print("model hook", )
        # print(module)
        # print('grad_input', grad_input)
        # print('grad_output', grad_output)

    def hook_func_operator(self, module, grad_input, grad_output):
        pass


class EQLv2(nn.Module):
    def __init__(self,
                 use_sigmoid=True,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0,
                 num_classes=10,  # 1203 for lvis v1.0, 1230 for lvis v0.5
                 gamma=12,
                 mu=0.8,
                 alpha=4.0,
                 vis_grad=False,
                 test_with_obj=True,
                 device='cpu'):
        super().__init__()
        self.use_sigmoid = True
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self.num_classes = num_classes
        self.group = True
        self.hook = Hook()  # 用来改梯度的

        # cfg for eqlv2
        self.vis_grad = vis_grad
        self.gamma = gamma
        self.mu = mu
        self.alpha = alpha

        # initial variables
        self.register_buffer('pos_grad', torch.zeros(self.num_classes))
        self.register_buffer('neg_grad', torch.zeros(self.num_classes))
        # At the beginning of training, we set a high value (eg. 100)
        # for the initial gradient ratio so that the weight for pos gradients and neg gradients are 1.
        self.register_buffer('pos_neg', torch.ones(self.num_classes) * 100)
        self.register_buffer('pos_neg', torch.zeros(self.num_classes))
        self.pos_grad = self.pos_grad.to(device)
        self.neg_grad = self.neg_grad.to(device)
        self.pos_neg = self.pos_neg.to(device)
        self.pn_diff = self.pos_neg.to(device)

        self.ce_layer = nn.CrossEntropyLoss()
        self.test_with_obj = test_with_obj

        def _func(x, gamma, mu):
            return 1 / (1 + torch.exp(-gamma * (x - mu)))
        self.map_func = partial(_func, gamma=self.gamma, mu=self.mu)
        logger = get_root_logger()
        logger.info(f"build EQL v2, gamma: {gamma}, mu: {mu}, alpha: {alpha}")

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        self.n_i, self.n_c = cls_score.size()

        self.gt_classes = label
        self.pred_class_logits = cls_score
        # hook_handle = cls_score.register_hook(self.hook_func_tensor)

        def expand_label(pred, gt_classes):
            target = pred.new_zeros(self.n_i, self.n_c)
            target[torch.arange(self.n_i), gt_classes] = 1
            return target
        # target.shape = [20, 10]
        self.target = expand_label(cls_score, label)     # 生成一个target矩阵

        # 论文中的q(t)和r(t)
        # weight是啥啊:每个bce有一个loss，一共有[20, 10]个loss
        self.pos_w, self.neg_w = self.get_weight()
        self.weight = self.pos_w * self.target + self.neg_w * (1 - self.target)

        # 只管tail类别
        # self.weight[torch.arange(self.n_i), :6] = 1

        cls_loss = F.binary_cross_entropy_with_logits(
            cls_score, self.target, reduction='none')
        # cls_loss = torch.sum(cls_loss * self.weight) / self.n_i
        cls_loss = torch.sum(cls_loss) / self.n_i
        hook_handle = cls_score.register_hook(self.hook_func_tensor)
        # self.collect_grad(cls_score.detach(), self.target.detach(), self.weight.detach())
        self.print_for_debug()
        # hook_handle.remove()
        return self.loss_weight * cls_loss

    def hook_func_tensor(self, grad):
        # 更改梯度
        grad *= self.weight
        batch_size = grad.shape[0]
        class_nums = grad.shape[1]
        # # 收集梯度: collect_grad可用，这里不再使用
        target_temp = self.target.detach()
        grad_temp = grad.detach()
        grad_temp = torch.abs(grad_temp)

        # 更新accu grad
        # grad_temp *= self.weight
        pos_grad = torch.sum(grad_temp * target_temp, dim=0)
        neg_grad = torch.sum(grad_temp * (1 - target_temp), dim=0)
        self.pos_grad += pos_grad
        self.neg_grad += neg_grad
        self.pos_neg = self.pos_grad / (self.neg_grad + 1e-20)
        self.pn_diff = torch.abs(self.pos_grad - self.neg_grad)

        # for sample_id in range(batch_size):     # 对于每个样本
        #     for classifier_id in range(class_nums):  # 对于每个分类器
        #         if classifier_id == self.gt_classes[sample_id]: # 正样本
        #             grad[sample_id][classifier_id] *= self.pos_w[classifier_id]               # 加权
        #         else:
        #             grad[sample_id][classifier_id] *= self.neg_w[classifier_id]               # 加权
        # # print("真实的值:")
        # print(grad)
        # grad = self.grad
        pass

    def print_for_debug(self):
        # print("pos", self.pos_grad)
        # print("neg", self.neg_grad)
        # print("ratio", self.pos_neg)
        # print("diff", self.pn_diff)
        # print("pos_w", self.pos_w)
        # print("neg_w", self.neg_w)
        pass

    def collect_grad(self, cls_score, target, weight=None):
        prob = torch.sigmoid(cls_score)
        grad = target * (prob - 1) + (1 - target) * prob
        grad = torch.abs(grad)
        batch_size = grad.shape[0]
        grad = grad / batch_size
        # do not collect grad for objectiveness branch [:-1] why?
        # pos_grad = torch.sum(grad * target * weight, dim=0)[:-1]
        # neg_grad = torch.sum(grad * (1 - target) * weight, dim=0)[:-1]
        pos_grad = torch.sum(grad * target * weight, dim=0)
        neg_grad = torch.sum(grad * (1 - target) * weight, dim=0)

        # dist.all_reduce(pos_

        self.pos_grad += pos_grad
        self.neg_grad += neg_grad
        self.pos_neg = self.pos_grad / (self.neg_grad + 1e-20)

        # print("计算的值：")
        # print("grad", grad)
        # print("pos_grad", self.pos_grad)
        # print("neg_grad", self.neg_grad)
        # print("pos_neg", self.pos_neg)

    def normalization(self, data):
        _range = np.max(data) - np.min(data)
        return (data - np.min(data)) / _range

    def standardization(self, data):
        mu = np.mean(data, axis=0)
        sigma = np.std(data, axis=0)
        return (data - mu) / sigma

    def get_weight(self):
        neg_w = self.map_func(self.pos_neg)
        pos_w = 1 + self.alpha * (1 - neg_w)
        # neg_w = neg_w.view(1, -1).expand(self.n_i, self.n_c)
        # pos_w = pos_w.view(1, -1).expand(self.n_i, self.n_c)
        return pos_w, neg_w


def get_acc_file_path(args):

    rootpath = './temp/'
    if not os.path.exists(rootpath):  # for fedavg, beta = 0,
        os.makedirs(rootpath)

    if args.balanced_global:
        rootpath += 'global_'
    rootpath += 'fl'
    if args.beta > 0:  # set default mu = 1, and set beta = 1 when using fedprox
        #args.mu = 1
        rootpath += "_LP_%.2f" % (args.beta)
    fpath = rootpath + '_acc_{}_{}_cons_frac{}_iid{}_iter{}_ep{}_lr{}_N{}_{}_seed{}_p{}_dirichlet{}_IF{}_Loss{}.txt'.format(
        args.dataset, args.model, args.frac, args.iid, args.rounds, args.local_ep, args.lr, args.num_users, args.num_classes, args.seed, args.non_iid_prob_class, args.alpha_dirichlet, args.IF, args.loss_type)
    return fpath


if __name__ == '__main__':
    # parse args
    args = args_parser()
    # print("STOP")
    # return
    # torch.manual_seed(args.seed)
    # torch.cuda.manual_seed(args.seed)
    # torch.cuda.manual_seed_all(args.seed)
    # np.random.seed(args.seed)
    # random.seed(args.seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    fpath = get_acc_file_path(args)
    f_acc = open(fpath, 'a')
    print(fpath)

    # pdb.set_trace()

    # myDataset containing details and configs about dataset(note: details)
    datasetObj = myDataset(args)
    if args.balanced_global:
        dataset_train, dataset_test, dict_users, dict_localtest = datasetObj.get_balanced_dataset(
            datasetObj.get_args())  # CIFAR10
    else:
        dataset_train, dataset_test, dict_users, dict_localtest = datasetObj.get_imbalanced_dataset(
            datasetObj.get_args())  # IMBALANCEDCIFAR10
        # data_path = './cifar_lt/'
        # trans_val = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                          std=[0.229, 0.224, 0.225])],
        # )
        # dataset_test_lt = IMBALANCECIFAR10(
        #     data_path, imb_factor=args.IF, train=False, download=True, transform=trans_val)
        # dict_globaltest_lt = set()
        # for item in dict_localtest.values():
        #     dict_globaltest_lt = dict_globaltest_lt.union(item)
    print(len(dict_users))
    # pdb.set_trace()
    # torch.manual_seed(args.seed)
    # torch.cuda.manual_seed(args.seed)
    # torch.cuda.manual_seed_all(args.seed)

    # build model
    net = build_model(args)
    # init weight

    init.constant(net.linear.bias, 0.001)

    # copy weights
    # return a dictionary containing a whole state of the module
    weights = net.state_dict()

    # training
    args.frac = 1
    m = max(int(args.frac * args.num_users), 1)  # num_select_clients
    prob = [1/args.num_users for j in range(args.num_users)]

    # add fl training
    epochs = 1000

    net.train()
    # train and update
    if args.lr is None:
        optimizer = torch.optim.SGD(
            net.parameters(), lr=args.lr, momentum=args.momentum)
    else:
        optimizer = torch.optim.SGD(
            net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=0.0001)

    epoch_loss = []
    eqlv2 = EQLv2(device=args.device)
    for epoch in tqdm(range(epochs)):
        # train
        batch_loss = []
        # distri_class_real = [0 for i in range(10)]
        # use/load data from split training set "ldr_train"
        train_loader = torch.utils.data.DataLoader(
            dataset=dataset_train, batch_size=20, shuffle=True)
        for images, labels in train_loader:
            images, labels = images.to(args.device), labels.to(args.device)
            # for i in range(20):
            #     distri_class_real[int(labels[i])] += 1
            labels = labels.long()
            net.zero_grad()
            logits = net(images)
            # criterion = nn.CrossEntropyLoss()
            criterion = eqlv2
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            batch_loss.append(loss.item())

        epoch_loss.append(sum(batch_loss)/len(batch_loss))

        # test
        acc_s2, global_3shot_acc = globaltest(copy.deepcopy(net).to(
            args.device), dataset_test, args, dataset_class=datasetObj)
        global_3shot_acc_mean = (
            global_3shot_acc["head"]+global_3shot_acc["middle"]+global_3shot_acc["tail"]) / 3
        print('epoch %d, global test acc  %.3f \n' % (epoch, acc_s2))
        print('epoch %d, global 3shot acc: [head: %.3f, middle: %.3f, tail: %.3f], mean: %.3f \n' % (
            epoch, global_3shot_acc["head"], global_3shot_acc["middle"], global_3shot_acc["tail"], global_3shot_acc_mean))

    torch.cuda.empty_cache()
