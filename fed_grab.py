
import os
import copy
from pydoc import classname
import numpy as np
import random
import torch

import pdb

from tqdm import tqdm
from options import args_parser
from util.update_baseline import LocalUpdate, globaltest, localtest
from util.fedavg import FedAvg, FedAvg_noniid, Weighted_avg_f1, weno_aggeration, FedAvg_Rod
# from util.util import add_noise
from util.dataset import *
from util.losses import *
from util.util import shot_split
from model.build_model import build_model
from matplotlib import pyplot as plt

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

    args = args_parser()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


    fpath = get_acc_file_path(args)
    f_acc = open(fpath,'a')
    print(fpath)


    datasetObj = myDataset(args)
    if args.balanced_global:
        dataset_train, dataset_test, dict_users, dict_localtest = datasetObj.get_balanced_dataset(datasetObj.get_args())  # CIFAR10
    else:
        dataset_train, dataset_test, dict_users, dict_localtest = datasetObj.get_imbalanced_dataset(datasetObj.get_args())  # IMBALANCEDCIFAR10

    
    print(len(dict_users))

    block_expansion = 1
    num_classes = 10
    g_backbone = build_model(args)   
    g_classifier = nn.Linear(512*block_expansion, num_classes).to(args.device) 


    # training
    args.frac = 1
    m = max(int(args.frac * args.num_users), 1) #num_select_clients 
    prob = [1/args.num_users for j in range(args.num_users)]


    g_linears = [nn.Linear(512*block_expansion, num_classes) for i in range(args.num_users)]
    for i in range(args.num_users):
        g_linears[i] = g_linears[i].to(args.device)

    g_pid_losses = [PIDLOSS(device=args.device, pidmask=["head", "middle"], class_activation = False) for i in range(args.num_users)] # 每一个client都有一个pidloss

    for idx in range(args.num_users):
        g_pid_losses[idx].get_3shotclass(head_class=[], middle_class=[], tail_class=[])   

    for idx in range(args.num_users):
        g_pid_losses[idx].get_3shotclass(head_class=[], middle_class=[], tail_class=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        g_pid_losses[idx].apply_3shot_mask()  
        g_pid_losses[idx].apply_class_activation()   


    for rnd in tqdm(range(args.rounds)):
        backbone_w_locals, linear_w_locals, loss_locals = [], [], []
        idxs_users = np.random.choice(range(args.num_users), m, replace=False, p=prob)
        for idx in range(args.num_users):  
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            backbone_w_local, linear_w_local, loss_local = local.update_weights_pid(net=copy.deepcopy(g_backbone).to(args.device), seed=args.seed, epoch=args.local_ep, GBA_Loss = g_pid_losses[idx], GBA_Layer = g_linears[idx])
            backbone_w_locals.append(copy.deepcopy(backbone_w_local))  
            linear_w_locals.append(copy.deepcopy(linear_w_local))
            loss_locals.append(copy.deepcopy(loss_local))


        dict_len = [len(dict_users[idx]) for idx in idxs_users]

        backbone_w_avg, linear_w_avg = FedAvg_Rod(backbone_w_locals, linear_w_locals, dict_len)  

        print("round:", rnd)

        g_backbone.load_state_dict(copy.deepcopy(backbone_w_avg))
        g_classifier.load_state_dict(copy.deepcopy(linear_w_avg))
        acc_s2, global_3shot_acc = globaltest(backbone = copy.deepcopy(g_backbone).to(args.device), classifier = copy.deepcopy(g_classifier).to(args.device), test_dataset = dataset_test, args = args, dataset_class = datasetObj)
        
        g_linears = [copy.deepcopy(g_classifier) for i in range(args.num_users)]

        print('round %d, global test acc  %.3f \n'%(rnd, acc_s2))
        print('round %d, global 3shot acc: [head: %.3f, middle: %.3f, tail: %.3f] \n'%(rnd, global_3shot_acc["head"], global_3shot_acc["middle"], global_3shot_acc["tail"]))
    torch.cuda.empty_cache()
