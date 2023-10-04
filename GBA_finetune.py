# python version 3.7.1
# -*- coding: utf-8 -*-
# 直接mask掉一部分


import contextlib
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
from util.fedavg import FedAvg, FedAvg_noniid, Weighted_avg_f1, weno_aggeration
# from util.util import add_noise
from util.dataset import *
from util.losses import *
from util.util import shot_split
from model.build_model import build_model
from matplotlib import pyplot as plt
import sys

import time 

np.set_printoptions(threshold=np.inf)

class DummyFile:
    def __init__(self, file):
        if file is None:
            file = sys.stderr
        self.file = file

    def write(self, x):
        if len(x.rstrip()) > 0:
            tqdm.write(x, file=self.file)

@contextlib.contextmanager
def redirect_stdout(file=None):
    if file is None:
        file = sys.stderr
    old_stdout = file
    sys.stdout = DummyFile(file)
    yield
    sys.stdout = old_stdout

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
   

class GBA_Shift():
    def __init__(self):
        # self.mask_list = [["head", "middle", "tail"], ["head", "tail"], ["head"], ["head", "middle"]]
        # self.mask_list = [["middle", "tail"], ["head", "middle", "tail"], ["head", "tail"], ["head"], ["head", "middle"]]
        self.mask_list = [["middle", "tail"], ["tail"], ["head", "middle", "tail"], ["head", "tail"], ["head"], ["head", "middle"]]
        self.itr = len(self.mask_list)*1000000 - 1
    
    def shift(self, pid_losses, args):
        # with redirect_stdout():
        print("itr:{}, shift to{}".format(self.itr, self.mask_list[self.itr % len(self.mask_list)]))
        for idx in range(args.num_users):
            pid_losses[idx].clear()
            pid_losses[idx].set_mask(self.mask_list[self.itr % len(self.mask_list)])
            # 注意下面两个的顺序，类激活将没有mask的class初始化为关闭，根据label打开
            pid_losses[idx].apply_3shot_mask()  # 应用3shot mask
            pid_losses[idx].apply_class_activation()    #应用类激活
        self.itr -= 1

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
    gba_shift = GBA_Shift()
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
    netglob = build_model(args)
    # netglob.load_state_dict(torch.load("/home/zikaixiao/zikai/aaFL/flpid_cifar100_baseline/output/model_parameter_fednova_lr0.005306.pkl"))
    # netglob.load_state_dict(torch.load("/home/zikaixiao/zikai/aaFL/fl_gba_cifar100/output/model_parameter_normal_207.pkl"))
    # netglob = torch.load("/home/zikaixiao/zikai/aaFL/fl_gba_cifar100/output/if100/netglob_499.pth", map_location='cpu')
    netglob = torch.load("/home/zikaixiao/zikai/aaFL/fl_gba_cifar100/output/if100/netglob_499.pth")
    
    acc_s2, global_3shot_acc = globaltest(copy.deepcopy(netglob).to(args.device), dataset_test, args, dataset_class = datasetObj)
    print('round %d, global test acc  %.3f \n'%(-1, acc_s2))
    print('round %d, global 3shot acc: [head: %.3f, middle: %.3f, tail: %.3f] \n'%(-1, global_3shot_acc["head"], global_3shot_acc["middle"], global_3shot_acc["tail"]))

    # copy weights
    w_glob = netglob.state_dict()  # return a dictionary containing a whole state of the module

    # training
    args.frac = 1
    m = max(int(args.frac * args.num_users), 1) #num_select_clients 
    prob = [1/args.num_users for j in range(args.num_users)]
    global_acc = [0]
    # pid init·
    pid_losses = [PIDLOSS(device=args.device, pidmask=["head", "middle"], num_classes=100, class_activation = False, dist = datasetObj.global_train_distribution) for i in range(args.num_users)] # 每一个client都有一个pidloss
    # 算出head, middle和tail
    # client_distribution = datasetObj.training_set_distribution
    for idx in range(args.num_users):
        # three_shot_dict, _ = shot_split(client_distribution[idx], threshold_3shot=[75, 95])
        # 听说用全局的分布来pid特别牛逼？
        pid_losses[idx].get_3shotclass(head_class=[], middle_class=[], tail_class=[])   # 最开始不mask

    # 更新3shot类别，进行pid mask
    # weno_obj.visualization(rnd)
    for idx in range(args.num_users):
        # pid_losses[idx].get_3shotclass(head_class=[i for i in range(0, 1)], middle_class=[i for i in range(1, 2)], tail_class=[i for i in range(2, 100)])
        pid_losses[idx].get_3shotclass([], [], tail_class=[i for i in range(0, 100)])

        # 注意下面两个的顺序，类激活将没有mask的class初始化为关闭，根据label打开
        pid_losses[idx].apply_3shot_mask()  # 应用3shot mask
        pid_losses[idx].apply_class_activation()    #应用类激活
    # torch.save(w_glob, "./output/w_glob_" + str(rnd) + ".pkl")
    # add fl training
    for rnd in range(args.rounds):
        # start_time = time.time()            ####
        w_locals, loss_locals = [], []
        idxs_users = np.random.choice(range(args.num_users), m, replace=False, p=prob)
        for idx in range(args.num_users):  # training over the subset, in fedper, all clients train
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            # w_local, loss_local = local.update_weights(net=copy.deepcopy(netglob).to(args.device), seed=args.seed, net_glob=netglob.to(args.device), epoch=args.local_ep)
            w_local, loss_local = local.update_weights_GBA_Finetune(net=copy.deepcopy(netglob).to(args.device), seed=args.seed, epoch=args.local_ep, pidloss=pid_losses[idx])
            w_locals.append(copy.deepcopy(w_local))  # store every updated model
            loss_locals.append(copy.deepcopy(loss_local))
            # if idx == 2:
            #     break

        # end_time = time.time()
        # with redirect_stdout():
        #     print("train耗时:{:.2f}秒".format(end_time - start_time))

        dict_len = [len(dict_users[idx]) for idx in idxs_users]
        
        # w_glob = weno_aggeration(w_locals, dict_len, datasetObj)
        w_glob = FedAvg_noniid(w_locals, dict_len)      # 聚合


        # # Local test 
        # acc_list = []
        # f1_macro_list = []
        # f1_weighted_list = []
        # acc_3shot_local_list = []       #####################
        # for i in range(args.num_users):
        #     netglob.load_state_dict(copy.deepcopy(w_locals[i]))
        #     # print('copy sucess')
        #     acc_local, f1_macro, f1_weighted, acc_3shot_local = localtest(copy.deepcopy(netglob).to(args.device), dataset_test, dataset_class = datasetObj, idxs=dict_localtest[i], user_id = i)
        #     # print('local test success')
        #     acc_list.append(acc_local)
        #     f1_macro_list.append(f1_macro)
        #     f1_weighted_list.append(f1_weighted)
        #     acc_3shot_local_list.append(acc_3shot_local) ###################

        # # save model and para to "./output"
        # # torch.save(netglob, "./output/netglob.pth")
        # # for i in range(args.num_users):
        # #     torch.save(w_locals[i], "./output/" + "w_local_" + str(i) + ".pth")
        # #     # netglob.load_state_dict(copy.deepcopy(w_locals[i]))

        # # start:calculate acc_3shot_local
        # avg3shot_acc={"head":0, "middle":0, "tail":0}
        # divisor = {"head":0, "middle":0, "tail":0}
        # for i in range(len(acc_3shot_local_list)):
        #     avg3shot_acc["head"] += acc_3shot_local_list[i]["head"][0]
        #     avg3shot_acc["middle"] += acc_3shot_local_list[i]["middle"][0]
        #     avg3shot_acc["tail"] += acc_3shot_local_list[i]["tail"][0]
        #     divisor["head"] += acc_3shot_local_list[i]["head"][1]
        #     divisor["middle"] += acc_3shot_local_list[i]["middle"][1]
        #     divisor["tail"] += acc_3shot_local_list[i]["tail"][1]
        # avg3shot_acc["head"] /= divisor["head"]
        # avg3shot_acc["middle"] /= divisor["middle"]
        # avg3shot_acc["tail"] /= divisor["tail"]
        # # end 
        
        # # start: calculate 3shot of each client
        # # # three_shot_client = [{"head":0, "middle":0, "tail":0} for i in range(len(acc_3shot_local_list))]
        # for i in range(len(acc_3shot_local_list)):
        #     acclist = []
        #     if acc_3shot_local_list[i]["head"][1] == True:
        #         acclist.append(acc_3shot_local_list[i]["head"][0])
        #     else:
        #         acclist.append(0)

        #     if acc_3shot_local_list[i]["middle"][1] == True:
        #         acclist.append(acc_3shot_local_list[i]["middle"][0])
        #     else:
        #         acclist.append(0)
                
        #     if acc_3shot_local_list[i]["tail"][1] == True:
        #         acclist.append(acc_3shot_local_list[i]["tail"][0])
        #     else:
        #         acclist.append(0)
        #     print("3shot of client {}:head:{}, middle:{}, tail{}".format(i, acclist[0], acclist[1], acclist[2]))
        # # end
        # avg_local_acc = sum(acc_list)/len(acc_list)
        # # print('Calculate the local average acc')
        # avg_f1_macro = Weighted_avg_f1(f1_macro_list,dict_len=dict_len)
        # avg_f1_weighted = Weighted_avg_f1(f1_weighted_list,dict_len)

        # start_time = time.time()

        netglob.load_state_dict(copy.deepcopy(w_glob))
        acc_s2, global_3shot_acc = globaltest(copy.deepcopy(netglob).to(args.device), dataset_test, args, dataset_class = datasetObj)
        global_acc.append(acc_s2)

        # 不用shift
        # if global_acc[-1] < global_acc[-2]:     # 越训练越不行了= =
        #     gba_shift.shift(pid_losses, args)


        # end_time = time.time()
        # with redirect_stdout():
        #     print("test耗时:{:.2f}秒".format(end_time - start_time))


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

        # with redirect_stdout():
        print('round %d, global test acc  %.3f \n'%(rnd, acc_s2))
        print('round %d, global 3shot acc: [head: %.3f, middle: %.3f, tail: %.3f] \n'%(rnd, global_3shot_acc["head"], global_3shot_acc["middle"], global_3shot_acc["tail"]))
    torch.cuda.empty_cache()
