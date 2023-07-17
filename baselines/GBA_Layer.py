# python version 3.7.1
# -*- coding: utf-8 -*-
# 直接mask掉一部分


import os
import copy
from pydoc import classname
import numpy as np
import random
import torch

import pdb

from tqdm import tqdm
from options import args_parser
from util.update_baseline import *
from util.fedavg import FedAvg, FedAvg_noniid, Weighted_avg_f1, weno_aggeration, FedAvg_Rod
# from util.util import add_noise
from util.dataset import *
from util.losses import *
from util.util import shot_split
from model.build_model import build_model
from matplotlib import pyplot as plt

# from tensorboardX import SummaryWriter
# writer = SummaryWriter('runs_tensorboard')

np.set_printoptions(threshold=np.inf)


def get_acc_file_path(args):

    rootpath = './temp/'
    if not os.path.exists(rootpath):  #for fedavg, beta = 0, 
        os.makedirs(rootpath)
 
    if args.balanced_global:
        rootpath+='global_' 
    rootpath += 'fl'
    if args.beta > 0: # set default mu = 1, and set beta = 1 when using fedprox
        #args.mu = 1
        rootpath += "_LP_%.2f" % (args.beta)
    fpath =  rootpath + '_acc_{}_{}_cons_frac{}_iid{}_iter{}_ep{}_lr{}_N{}_{}_seed{}_p{}_dirichlet{}_IF{}_Loss{}.txt'.format(
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
    f_acc = open(fpath,'a')
    print(fpath)

    # pdb.set_trace()

    # myDataset containing details and configs about dataset(note: details)
    datasetObj = myDataset(args)
    if args.balanced_global:
        dataset_train, dataset_test, dict_users, dict_localtest = datasetObj.get_balanced_dataset(datasetObj.get_args())  # CIFAR10
    else:
        dataset_train, dataset_test, dict_users, dict_localtest = datasetObj.get_imbalanced_dataset(datasetObj.get_args())  # IMBALANCEDCIFAR10
        # data_path = './cifar_lt/'
        # trans_val = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                         std=[0.229, 0.224, 0.225])],
        # )
        # dataset_test_lt = IMBALANCECIFAR10(data_path, imb_factor=args.IF,train=False, download=True, transform=trans_val)
    
    
    print(len(dict_users))
    # pdb.set_trace()
    # torch.manual_seed(args.seed)
    # torch.cuda.manual_seed(args.seed)
    # torch.cuda.manual_seed_all(args.seed)
    block_expansion = 1
    num_classes = 100
    # build model, 这个是全局模型，对于server而言，g_backnone和g_classifier即使architecture也是weights
    g_backbone = build_model(args)    # 全局模型中的feature extractor部分
    g_classifier = nn.Linear(512*block_expansion, num_classes).to(args.device) # 全局模型中的classifier部分
    # copy weights
    # w_backnone = g_backbone.state_dict()  # return a dictionary containing a whole state of the module

    # training
    args.frac = 1
    m = max(int(args.frac * args.num_users), 1) #num_select_clients 
    prob = [1/args.num_users for j in range(args.num_users)]

    # classifier init，分为global classifier和local classifier
    # resnet中forward(self, x, latent_output=False)中的false改为true
    # local的两个linear
    g_linears = [nn.Linear(512*block_expansion, num_classes) for i in range(args.num_users)]
    for i in range(args.num_users):
        g_linears[i] = g_linears[i].to(args.device)
    # pid init
    g_pid_losses = [PIDLOSS(device=args.device, num_classes=100, pidmask=["head", "middle"], class_activation = False) for i in range(args.num_users)] # 每一个client都有一个pidloss
    # 算出head, middle和tail
    # client_distribution = datasetObj.training_set_distribution
    for idx in range(args.num_users):
        # three_shot_dict, _ = shot_split(client_distribution[idx], threshold_3shot=[75, 95])
        g_pid_losses[idx].get_3shotclass(head_class=[], middle_class=[], tail_class=[])   # 最开始不mask
        # p_pid_losses[idx].get_3shotclass(head_class=[], middle_class=[], tail_class=[])

    # weno init
    # weno_obj = WENO(class_num = 10)
    # global pid init
    for idx in range(args.num_users):
        g_pid_losses[idx].get_3shotclass(head_class=[i for i in range(0, 27)], middle_class=[i for i in range(27, 28)], tail_class=[i for i in range(28, 100)])
        # g_pid_losses[idx].get_3shotclass(head_class=[i for i in range(0, 27)], middle_class=[i for i in range(27, 75)], tail_class=[i for i in range(75, 101)])
        # 注意下面两个的顺序，类激活将没有mask的class初始化为关闭，根据label打开
        g_pid_losses[idx].apply_3shot_mask()  # 应用3shot mask
        g_pid_losses[idx].apply_class_activation()    #应用类激活
    # torch.save(w_glob, "./output/w_glob_" + str(rnd) + ".pkl")

    # g_backbone = torch.load("/home/zikaixiao/zikai/aaFL/fl_gba_cifar100/output/g_backbone/g_backbone_990.pth")
    # g_classifier = torch.load("/home/zikaixiao/zikai/aaFL/fl_gba_cifar100/output/g_classifier/g_classifier_990.pth")
    # g_backbone.to(args.device)
    # g_classifier.to(args.device)
    # acc_s2, global_3shot_acc = globaltest_GBA_Layer(backbone = copy.deepcopy(g_backbone).to(args.device), classifier = copy.deepcopy(g_classifier).to(args.device), test_dataset = dataset_test, args = args, dataset_class = datasetObj)
    # print('global test acc  %.3f \n'%(acc_s2))
    # print('global 3shot acc: [head: %.3f, middle: %.3f, tail: %.3f] \n'%(global_3shot_acc["head"], global_3shot_acc["middle"], global_3shot_acc["tail"]))
    
    
    # add fl training
    for rnd in tqdm(range(1000, args.rounds+1000)):
        backbone_w_locals, linear_w_locals_1, loss_locals = [], [], []
        idxs_users = np.random.choice(range(args.num_users), m, replace=False, p=prob)
        for idx in range(args.num_users):  # training over the subset, in fedper, all clients train
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])

            backbone_w_local, linear_w_local, loss_local = local.update_weights_GBA_Layer(net=copy.deepcopy(g_backbone).to(args.device), seed=args.seed, epoch=args.local_ep, GBA_Loss = g_pid_losses[idx], GBA_Layer = g_linears[idx])
            # backbone_w_local, linear_w_local, loss_local = local.update_weights_GBA_Layer(net=copy.deepcopy(g_backbone).to(args.device), seed=args.seed, epoch=args.local_ep, GBA_Loss = nn.CrossEntropyLoss(), GBA_Layer = g_linears[idx])


            backbone_w_locals.append(copy.deepcopy(backbone_w_local))  # store every updated model
            linear_w_locals_1.append(copy.deepcopy(linear_w_local))
            loss_locals.append(copy.deepcopy(loss_local))
            # if idx == 2:
            #     break

        dict_len = [len(dict_users[idx]) for idx in idxs_users]
        # print(linear_w_locals)
        # w_glob = weno_aggeration(w_locals, dict_len, datasetObj)
        backbone_w_avg, linear_w_avg = FedAvg_Rod(backbone_w_locals, linear_w_locals_1, dict_len)      # 聚合


        g_backbone.load_state_dict(copy.deepcopy(backbone_w_avg))
        g_classifier.load_state_dict(copy.deepcopy(linear_w_avg))
        if rnd % 1 == 0:
            acc_s2, global_3shot_acc = globaltest_GBA_Layer(backbone = copy.deepcopy(g_backbone).to(args.device), classifier = copy.deepcopy(g_classifier).to(args.device), test_dataset = dataset_test, args = args, dataset_class = datasetObj)
            # torch.save(g_backbone, "./output/g_backbone/" + "g_backbone_" + str(rnd) + ".pth")
            # torch.save(g_classifier, "./output/g_classifier/" + "g_classifier_" + str(rnd) + ".pth")
            print('round %d, global test acc  %.3f \n'%(rnd, acc_s2))
            print('round %d, global 3shot acc: [head: %.3f, middle: %.3f, tail: %.3f] \n'%(rnd, global_3shot_acc["head"], global_3shot_acc["middle"], global_3shot_acc["tail"]))
        g_linears = [copy.deepcopy(g_classifier) for i in range(args.num_users)]
        # f_acc.write('round %d, local average test acc  %.4f \n'%(rnd, avg_local_acc))
        # f_acc.write('round %d, local macro average F1 score  %.4f \n'%(rnd, avg_f1_macro))
        # f_acc.write('round %d, local weighted average F1 score  %.4f \n'%(rnd, avg_f1_weighted))
        # f_acc.write('round %d, global test acc  %.4f \n'%(rnd, acc_s2))
        # f_acc.write('round %d, global 3shot acc: [head: %.3f, middle: %.3f, tail: %.3f] \n'%(rnd, global_3shot_acc["head"], global_3shot_acc["middle"], global_3shot_acc["tail"]))
        # f_acc.write('round %d, average 3shot acc: [head: %.3f, middle: %.3f, tail: %.3f] \n'%(rnd, avg3shot_acc["head"], avg3shot_acc["middle"], avg3shot_acc["tail"]))
        # f_acc.flush()
        # print('round %d, local average test acc  %.3f \n'%(rnd, avg_local_acc))
        # print('round %d, local macro average F1 score  %.3f \n'%(rnd, avg_f1_macro))
        # print('round %d, local weighted average F1 score  %.3f \n'%(rnd, avg_f1_weighted))
        # print('round %d, average 3shot acc: [head: %.3f, middle: %.3f, tail: %.3f] \n'%(rnd, avg3shot_acc["head"], avg3shot_acc["middle"], avg3shot_acc["tail"]))

       
    # torch.cuda.empty_cache()
