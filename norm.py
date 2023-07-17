# python version 3.7.1
# -*- coding: utf-8 -*-

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,2,5,6"

import copy
import numpy as np
import random
import torch
import torch.nn as nn

import pdb

from tqdm import tqdm
from options import args_parser
from util.update_baseline import LocalUpdate, globaltest, localtest
from util.fedavg import *
# from util.util import add_noise
from util.dataset import *
from model.build_model import build_model



np.set_printoptions(threshold=np.inf)
def pnorm(weights, p):
    normB = torch.norm(weights, 2, 1)
    ws = weights.clone()
    for i in range(weights.size(0)):
        ws[i] = ws[i] / torch.pow(normB[i], p)
    return ws


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

    # build model
    netglob = build_model(args)
    netglob = torch.load("/home/zikaixiao/zikai/aaFL/fl_gba_cifar100/output/if100/netglob_400.pth", map_location='cpu')
    netglob.to(args.device)

    # netglob = nn.DataParallel(netglob, device_ids=device_ids)
    netglob = nn.DataParallel(netglob, device_ids=[0, 1, 2, 3])
    # copy weights
    w_glob = netglob.state_dict()  # return a dictionary containing a whole state of the module
    fc_weights = w_glob['module.linear.weight']
    p=1.3
    ws = pnorm(fc_weights, p)
    w_glob['module.linear.weight'] = ws
    netglob.load_state_dict(w_glob)

    # training
    args.frac = 1
    m = max(int(args.frac * args.num_users), 1) #num_select_clients 
    prob = [1/args.num_users for j in range(args.num_users)]

    acc_s2, global_3shot_acc = globaltest(copy.deepcopy(netglob).to(args.device), dataset_test, args, dataset_class = datasetObj)

    print('round %d, global test acc  %.3f \n'%(0, acc_s2))
    # print('round %d, average 3shot acc: [head: %.3f, middle: %.3f, tail: %.3f] \n'%(rnd, avg3shot_acc["head"], avg3shot_acc["middle"], avg3shot_acc["tail"]))
    print('round %d, global 3shot acc: [head: %.3f, middle: %.3f, tail: %.3f] \n'%(0, global_3shot_acc["head"], global_3shot_acc["middle"], global_3shot_acc["tail"]))
