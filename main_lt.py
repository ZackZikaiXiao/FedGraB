# python version 3.7.1
# -*- coding: utf-8 -*-

import os
import copy
import numpy as np
import random
import torch
import  torch.nn as nn
import pdb

from tqdm import tqdm
from options import args_parser
from util.update_baseline import LocalUpdate, globaltest, localtest
from util.fedavg import FedAvg, FedAvg_noniid, Weighted_avg_f1
# from util.util import add_noise
from util.dataset import *
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

    # myDataset containing details and configs about dataset(note: details)
    datasetObj = myDataset(args)
    if args.balanced_global:
        dataset_train, dataset_test, dict_users, dict_localtest = datasetObj.get_balanced_dataset(datasetObj.get_args())  # CIFAR10
    else:
        dataset_train, dataset_test, dict_users, dict_localtest = datasetObj.get_imbalanced_dataset(datasetObj.get_args())  # IMBALANCEDCIFAR10

    
    
    print(len(dict_users))
    # pdb.set_trace()
    # torch.manual_seed(args.seed)
    # torch.cuda.manual_seed(args.seed)
    # torch.cuda.manual_seed_all(args.seed)

    # build model
    net = build_model(args)
    # copy weights
    weights = net.state_dict()  # return a dictionary containing a whole state of the module

    # training
    args.frac = 1
    m = max(int(args.frac * args.num_users), 1) #num_select_clients 
    prob = [1/args.num_users for j in range(args.num_users)]

   
    # add fl training
    epochs = 1000

    net.train()
    # train and update
    if args.lr is None:
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
    else:
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)

    epoch_loss = []
    for epoch in tqdm(range(epochs)):
        # train
        batch_loss = []
        # use/load data from split training set "ldr_train"
        train_loader = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=20, shuffle=True)
        for images, labels in train_loader:
            images, labels = images.to(args.device), labels.to(args.device)
            labels = labels.long()
            net.zero_grad()
            logits = net(images)
            criterion = nn.CrossEntropyLoss()
            loss = criterion(logits, labels)               

            loss.backward()
            optimizer.step()

            batch_loss.append(loss.item())

        epoch_loss.append(sum(batch_loss)/len(batch_loss))

        # test
        acc_s2, global_3shot_acc = globaltest(copy.deepcopy(net).to(args.device), dataset_test, args, dataset_class = datasetObj)
        print('epoch %d, global test acc  %.3f \n'%(epoch, acc_s2))
        print('epoch %d, global 3shot acc: [head: %.3f, middle: %.3f, tail: %.3f] \n'%(epoch, global_3shot_acc["head"], global_3shot_acc["middle"], global_3shot_acc["tail"]))

    torch.cuda.empty_cache()
