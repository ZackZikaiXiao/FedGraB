# python version 3.7.1
# -*- coding: utf-8 -*-

import os
import copy
import numpy as np
import random
import torch

import pdb

from tqdm import tqdm
from options import args_parser
from util.update_baseline import LocalUpdate, globaltest, localtest
from util.fedavg import FedAvg, FedAvg_noniid
# from util.util import add_noise
from util.dataset import get_dataset, get_global_dataset
from model.build_model import build_model

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

    if args.balanced_global:
        dataset_train, dataset_test, dict_users, dict_localtest = get_global_dataset(args)  # CIFAR10
    else:
        dataset_train, dataset_test, dict_users, dict_localtest = get_dataset(args)  # IMBALANCEDCIFAR10

    
    
    print(len(dict_users))
    # pdb.set_trace()
    # torch.manual_seed(args.seed)
    # torch.cuda.manual_seed(args.seed)
    # torch.cuda.manual_seed_all(args.seed)

    # build model
    netglob = build_model(args)
    # copy weights
    w_glob = netglob.state_dict()  # return a dictionary containing a whole state of the module

    # training
    args.frac = 1
    m = max(int(args.frac * args.num_users), 1) #num_select_clients 
    prob = [1/args.num_users for j in range(args.num_users)]

   
    # add fl training
    for rnd in tqdm(range(args.rounds)):
        w_locals, loss_locals = [], []
        idxs_users = np.random.choice(range(args.num_users), m, replace=False, p=prob)
                
        for idx in range(args.num_users):  # training over the subset, in fedper, all clients train
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w_local, loss_local = local.update_weights(net=copy.deepcopy(netglob).to(args.device), seed=args.seed, w_g=netglob.to(args.device), epoch=args.local_ep)
            w_locals.append(copy.deepcopy(w_local))  # store every updated model
            loss_locals.append(copy.deepcopy(loss_local))

        dict_len = [len(dict_users[idx]) for idx in idxs_users]
        w_glob = FedAvg_noniid(w_locals, dict_len)

        # Local test 
        acc_list = []
        for i in range(args.num_users):
            netglob.load_state_dict(copy.deepcopy(w_locals[i]))
            # print('copy sucess')
            acc_local = localtest(copy.deepcopy(netglob).to(args.device), dataset_test, args,idxs=dict_localtest[i])
            # print('local test success')
            acc_list.append(acc_local)
        avg_local_acc = sum(acc_list)/len(acc_list)
        # print('Calculate the local average acc')
        netglob.load_state_dict(copy.deepcopy(w_glob))

        f_acc.write("round %d, local average test acc  %.4f \n"%(rnd, avg_local_acc))
        # print(avg_local_acc)
        acc_s2 = globaltest(copy.deepcopy(netglob).to(args.device), dataset_test, args)
        f_acc.write("round %d, global test acc  %.4f \n"%(rnd, acc_s2))
        f_acc.flush()
        print('round %d, local average test acc  %.3f \n'%(rnd, avg_local_acc))
        print('round %d, global test acc  %.3f \n'%(rnd, acc_s2))
    torch.cuda.empty_cache()
