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
from util.fedavg import *
# from util.util import add_noise
from util.dataset import *
from model.build_model import build_model
from torch.optim.optimizer import Optimizer, required

np.set_printoptions(threshold=np.inf)

class FedNova(Optimizer):
    r"""Implements federated normalized averaging (FedNova).

    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        ratio (float): relative sample size of client
        gmf (float): global/server/slow momentum factor
        mu (float): parameter for proximal local SGD
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)

    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf

    .. note::
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.

        Considering the specific case of Momentum, the update can be written as

        .. math::
                  v = \rho * v + g \\
                  p = p - lr * v

        where p, g, v and :math:`\rho` denote the parameters, gradient,
        velocity, and momentum respectively.

        This is in contrast to Sutskever et. al. and
        other frameworks which employ an update of the form

        .. math::
             v = \rho * v + lr * g \\
             p = p - v

        The Nesterov version is analogously modified.
    """

    def __init__(self, params, ratio, gmf, mu = 0, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, variance=0):
        
        self.gmf = gmf
        self.ratio = ratio
        self.momentum = momentum
        self.mu = mu
        self.local_normalizing_vec = 0
        self.local_counter = 0
        self.local_steps = 0


        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov, variance=variance)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(FedNova, self).__init__(params, defaults)


    def __setstate__(self, state):
        super(FedNova, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"

        loss = None
        if closure is not None:
            loss = closure()

        # scale = 1**self.itr

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data

                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                
                param_state = self.state[p]
                if 'old_init' not in param_state:
                    param_state['old_init'] = torch.clone(p.data).detach()

                local_lr = group['lr']

                # apply momentum updates
                if momentum != 0:
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                # apply proximal updates
                if self.mu != 0:
                    d_p.add_(self.mu, p.data - param_state['old_init'])

                # update accumalated local updates
                if 'cum_grad' not in param_state:
                    param_state['cum_grad'] = torch.clone(d_p).detach()
                    param_state['cum_grad'].mul_(local_lr)

                else:
                    param_state['cum_grad'].add_(local_lr, d_p)

                p.data.add_(-local_lr, d_p)

        # compute local normalizing vector a_i
        if self.momentum != 0:
            self.local_counter = self.local_counter * self.momentum + 1
            self.local_normalizing_vec += self.local_counter
        
        self.etamu = local_lr * self.mu
        if self.etamu != 0:
            self.local_normalizing_vec *= (1 - self.etamu)
            self.local_normalizing_vec += 1

        if self.momentum == 0 and self.etamu == 0:
            self.local_normalizing_vec += 1
        
        self.local_steps += 1

        return loss

    def average(self, weight=0, tau_eff=0):
        if weight == 0:
            weight = self.ratio
        if tau_eff == 0:
            if self.mu != 0:
                tau_eff_cuda = torch.tensor(self.local_steps*self.ratio).cuda()
            else:
                tau_eff_cuda = torch.tensor(self.local_normalizing_vec*self.ratio).cuda()
            tau_eff = tau_eff_cuda.item()

        param_list = []
        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                scale = tau_eff/self.local_normalizing_vec
                param_state['cum_grad'].mul_(weight*scale)
                param_list.append(param_state['cum_grad'])

        for group in self.param_groups:
            lr = group['lr']
            for p in group['params']:
                param_state = self.state[p]

                if self.gmf != 0:
                    if 'global_momentum_buffer' not in param_state:
                        buf = param_state['global_momentum_buffer'] = torch.clone(param_state['cum_grad']).detach()
                        buf.div_(lr)
                    else:
                        buf = param_state['global_momentum_buffer']
                        buf.mul_(self.gmf).add_(1/lr, param_state['cum_grad'])
                    param_state['old_init'].sub_(lr, buf)
                else:
                    param_state['old_init'].sub_(param_state['cum_grad'])
                
                p.data.copy_(param_state['old_init'])
                param_state['cum_grad'].zero_()

                # Reinitialize momentum buffer
                if 'momentum_buffer' in param_state:
                    param_state['momentum_buffer'].zero_()
        
        self.local_counter = 0
        self.local_normalizing_vec = 0
        self.local_steps = 0


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
    # copy weights
    w_glob = netglob.state_dict()  # return a dictionary containing a whole state of the module

    # training
    args.frac = 1
    m = max(int(args.frac * args.num_users), 1) #num_select_clients 
    prob = [1/args.num_users for j in range(args.num_users)]

    # client_number_ratio = [datasetObj.training_set_distribution[i].sum() for i in range(len(dict_localtest))]
    # client_number_ratio = [client_number_ratio[i] / sum(client_number_ratio) for i in range(len(client_number_ratio))]
    optimizer_nova = FedNova(netglob.parameters(), lr=args.lr, mu=0, gmf=0, ratio=1, momentum=args.momentum)
    
    # add fl training
    for rnd in tqdm(range(args.rounds)):
        w_locals, loss_locals = [], []
        idxs_users = np.random.choice(range(args.num_users), m, replace=False, p=prob)
                
        for idx in range(args.num_users):  # training over the subset, in fedper, all clients train
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w_local, loss_local = local.update_weights_fednova(net=netglob.to(args.device), seed=args.seed, epoch=args.local_ep, opitizer_nova=optimizer_nova)
            w_locals.append(copy.deepcopy(w_local))  # store every updated model
            loss_locals.append(copy.deepcopy(loss_local))


        dict_len = [len(dict_users[idx]) for idx in idxs_users]
        w_glob = FedAvg_noniid(w_locals, dict_len)
        # w_glob = classifier_avg(w_locals, dict_len, rnd)
        # w_glob = weno_aggeration(w_locals, dict_len, datasetObj, beta = 0.0005, round = rnd, internal = 25)

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
        netglob.load_state_dict(copy.deepcopy(w_glob))
        acc_s2, global_3shot_acc = globaltest(copy.deepcopy(netglob).to(args.device), dataset_test, args, dataset_class = datasetObj)
        
        optimizer_nova.average()

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
        print('round %d, global test acc  %.3f \n'%(rnd, acc_s2))
        # print('round %d, average 3shot acc: [head: %.3f, middle: %.3f, tail: %.3f] \n'%(rnd, avg3shot_acc["head"], avg3shot_acc["middle"], avg3shot_acc["tail"]))
        print('round %d, global 3shot acc: [head: %.3f, middle: %.3f, tail: %.3f] \n'%(rnd, global_3shot_acc["head"], global_3shot_acc["middle"], global_3shot_acc["tail"]))
    torch.cuda.empty_cache()
