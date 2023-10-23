import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from util import *
from functools import partial
from torch.nn.modules.loss import _Loss
import random

class LDAMLoss(nn.Module):

    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)

        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(
            self.m_list[None, :], index_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m

        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s*output, target, weight=self.weight)


class PIDLOSS(nn.Module):
    def __init__(self,
                 use_sigmoid=True,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0,
                 num_classes=10,
                 gamma=12,
                 mu=0.8,
                 alpha=4.0,
                 pidmask=["head"],
                 vis_grad=False,
                 test_with_obj=True,
                 device='cpu',
                 class_activation = False):
        super().__init__()
        self.use_sigmoid = True
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self.num_classes = num_classes
        self.group = True
        self.hook = Hook()
        self.controllers = [PID() for _ in range(self.num_classes)]
        self.pidmask = pidmask
        self.class_activation = class_activation
        self.class_acti_mask = None


        self.vis_grad = vis_grad
        self.gamma = gamma
        self.mu = mu
        self.alpha = alpha


        self.register_buffer('pos_grad', torch.zeros(self.num_classes))
        self.register_buffer('neg_grad', torch.zeros(self.num_classes))


        self.register_buffer('pos_neg', torch.ones(self.num_classes) * 100)
        self.register_buffer('pn_diff', torch.zeros(self.num_classes))
        self.pos_grad = self.pos_grad.to(device)
        self.neg_grad = self.neg_grad.to(device)
        self.pos_neg = self.pos_neg.to(device)
        self.pn_diff = self.pn_diff.to(device)

        self.ce_layer = nn.CrossEntropyLoss()
        self.test_with_obj = test_with_obj

        def _func(x):
            return (10 / 9) / ((1 / 9) + torch.exp(-0.5 * x))
        self.map_func = partial(_func)

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


        def expand_label(pred, gt_classes):
            target = pred.new_zeros(self.n_i, self.n_c)
            target[torch.arange(self.n_i), gt_classes] = 1
            return target

        self.target = expand_label(cls_score, label)



        self.pos_w, self.neg_w = self.get_weight(self.target)
        self.weight = self.pos_w * self.target + self.neg_w * (1 - self.target)


        if self.class_activation:
            if self.class_acti_mask == None:
                self.class_acti_mask = cls_score.new_ones(self.n_i, self.n_c)
                for i in range(self.n_c):
                    if "head" not in self.pidmask and i in self.head_class:    
                        self.class_acti_mask[torch.arange(self.n_i), i] = 0
                    if "middle" not in self.pidmask and i in self.middle_class:
                        self.class_acti_mask[torch.arange(self.n_i), i] = 0
                    if "tail" not in self.pidmask and i in self.tail_class: 
                        self.class_acti_mask[torch.arange(self.n_i), i] = 0
            else:
                for i in range(label.shape[0]):       
                    one_class = label[i]
                    if "head" not in self.pidmask and one_class in self.head_class:    

                        self.class_acti_mask[torch.arange(self.n_i), one_class] = 1
                        self.controllers[one_class].open()
                    if "middle" not in self.pidmask and one_class in self.middle_class:
                        self.class_acti_mask[torch.arange(self.n_i), one_class] = 1
                        self.controllers[one_class].open()

                    if "tail" not in self.pidmask and one_class in self.tail_class: 
                        self.class_acti_mask[torch.arange(self.n_i), one_class] = 1
                        self.controllers[one_class].open()

            self.weight *= self.class_acti_mask[0:self.n_i, :]

        cls_loss = F.binary_cross_entropy_with_logits(
            cls_score, self.target, reduction='none')

        cls_loss = torch.sum(cls_loss) / self.n_i
        hook_handle = cls_score.register_hook(self.hook_func_tensor)



        return self.loss_weight * cls_loss

    def hook_func_tensor(self, grad):

        a = 1

        batchsize = grad.shape[0]
        classes_num = grad.shape[1]


        tail_length = len(self.tail_class)
        img_max = 1
        prob_dist = []
        for cls_idx in range(tail_length):
            prob = img_max * (0.1**(cls_idx / (tail_length - 1.0)))
            prob_dist.append(prob)

        select_record = []
        tail_id = 0
        for c_id in range(classes_num):
            if c_id in self.tail_class:
                if random_unit(prob_dist[tail_id]) == True:

                    self.weight[torch.arange(batchsize), c_id] = 1
                    tail_id += 1
                    select_record.append(c_id)
                else:
                    continue           

        grad *= self.weight


        batch_size = grad.shape[0]
        class_nums = grad.shape[1]

        target_temp = self.target.detach()
        grad_temp = grad.detach()
        grad_temp = torch.abs(grad_temp)

        for c_id in range(classes_num):
            if c_id in select_record:
                grad_temp[torch.arange(batchsize), c_id] = 0

        pos_grad = torch.sum(grad_temp * target_temp, dim=0)
        neg_grad = torch.sum(grad_temp * (1 - target_temp), dim=0)


        self.pos_grad += pos_grad
        self.neg_grad += neg_grad
        self.pos_neg = self.pos_grad / (self.neg_grad + 1e-20)

        self.pn_diff = self.pos_grad - self.neg_grad
        

    def hook_func_tensor_bak(self, grad):

        a = 1

        batchsize = grad.shape[0]
        classes_num = grad.shape[1]



        grad *= self.weight


        batch_size = grad.shape[0]
        class_nums = grad.shape[1]

        target_temp = self.target.detach()
        grad_temp = grad.detach()
        grad_temp = torch.abs(grad_temp)


        pos_grad = torch.sum(grad_temp * target_temp, dim=0)
        neg_grad = torch.sum(grad_temp * (1 - target_temp), dim=0)


        self.pos_grad += pos_grad
        self.neg_grad += neg_grad
        self.pos_neg = self.pos_grad / (self.neg_grad + 1e-20)

        self.pn_diff = self.pos_grad - self.neg_grad


    def get_3shotclass(self, head_class, middle_class, tail_class):
        self.head_class = head_class
        self.middle_class = middle_class
        self.tail_class = tail_class

    def apply_3shot_mask(self):


        if "head" in self.pidmask:
            for i in self.head_class:

                self.controllers[i].reset()
                self.controllers[i].close()
        else:
             for i in self.head_class:
                self.controllers[i].reset()
                self.controllers[i].open()

        if "middle" in self.pidmask:
            for i in self.middle_class:

                self.controllers[i].reset()
                self.controllers[i].close()
        else:
             for i in self.middle_class:
                self.controllers[i].reset()
                self.controllers[i].open()

        if "tail" in self.pidmask:
            for i in self.tail_class:

                self.controllers[i].reset()
                self.controllers[i].close()
        else:
            for i in self.tail_class:
                self.controllers[i].reset()
                self.controllers[i].open()

    def apply_class_activation(self):
        if self.class_activation:

            if "head" not in self.pidmask:
                for i in self.head_class:

                    self.controllers[i].reset()
                    self.controllers[i].close()

            if "middle" not in self.pidmask:
                for i in self.middle_class:

                    self.controllers[i].reset()
                    self.controllers[i].close()

            if "tail" not in self.pidmask:
                for i in self.tail_class:

                    self.controllers[i].reset()
                    self.controllers[i].close()
   

    def get_weight(self, target):
        pos_w = target.new_zeros(self.num_classes)
        neg_w = target.new_zeros(self.num_classes)
        debug = 11
        for i in range(self.num_classes):
            pid_out = self.controllers[i].PID_calc(self.pn_diff[i], 0)

            if 0 - self.pn_diff[i] > 0:
                pos_w[i] = self.map_func(pid_out)
                neg_w[i] = self.map_func(-pid_out)
            else:
                pos_w[i] = self.map_func(pid_out)
                neg_w[i] = self.map_func(-pid_out)

        debug = 12




        return pos_w, neg_w

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



        pos_grad = torch.sum(grad * target * weight, dim=0)
        neg_grad = torch.sum(grad * (1 - target) * weight, dim=0)



        self.pos_grad += pos_grad
        self.neg_grad += neg_grad
        self.pos_neg = self.pos_grad / (self.neg_grad + 1e-20)

    def normalization(self, data):
        _range = np.max(data) - np.min(data)
        return (data - np.min(data)) / _range

    def standardization(self, data):
        mu = np.mean(data, axis=0)
        sigma = np.std(data, axis=0)
        return (data - mu) / sigma


class Hook():
    def __init__(self):
        self.m_count = 0

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
        self.gradient = grad.cpu().numpy().tolist()
        self.m_count += 1

    def hook_func_model(self, module, grad_input, grad_output):
        pass

    def hook_func_operator(self, module, grad_input, grad_output):
        pass


class PID():
    def __init__(self):
        self.mode = "PID_DELTA"
        self.Kp = 10
        self.Ki = 0.01
        self.Kd = 0.1

        self.max_out = 100
        self.max_iout = 100

        self.set = 0
        self.current_value = 0

        self.out = 0
        self.Pout = 0
        self.Iout = 0
        self.Dout = 0

        self.Dbuf = [0, 0, 0]

        self.error = [0, 0, 0]
        self.m_open = False


    def reset(self):
        self.current_value = 0
        self.out = 0
        self.Pout = 0
        self.Iout = 0
        self.Dout = 0

        self.Dbuf = [0, 0, 0]

        self.error = [0, 0, 0]
        self.m_open = False


    def open(self):
        self.m_open = True
        
    
    def close(self):
        self.m_open = False
        

    def is_open(self):
        return self.m_open

    def PID_calc(self, current_value, set_value):


        if self.m_open == False:
            return torch.Tensor([0.])


        self.error[2] = self.error[1]
        self.error[1] = self.error[0]

        self.set_value = set_value
        self.current_value = current_value

        self.error[0] = set_value - current_value

        if self.mode == "PID_POSITION":


            self.Pout = self.Kp * self.error[0]

            self.Iout += self.Ki * self.error[0]

            self.Dbuf[2] = self.Dbuf[1]
            self.Dbuf[1] = self.Dbuf[0]

            self.Dbuf[0] = (self.error[0] - self.error[1])

            self.Dout = self.Kd * self.Dbuf[0]

            self.LimitMax(self.Iout, self.max_iout)

            self.out = self.Pout + self.Iout + self.Dout

            self.LimitMax(self.out, self.max_out)

        elif self.mode == "PID_DELTA":



            self.Pout = self.Kp * (self.error[0] - self.error[1])

            self.Iout = self.Ki * self.error[0]

            self.Dbuf[2] = self.Dbuf[1]
            self.Dbuf[1] = self.Dbuf[0]

            self.Dbuf[0] = self.error[0] - 2.0 * self.error[1] + self.error[2]
            self.Dout = self.Kd * self.Dbuf[0]

            self.out += self.Pout + self.Iout + self.Dout

            self.LimitMax(self.out, self.max_out)

        return self.out

    def LimitMax(self, input, max):
        if input > max:
            input = max
        elif input < -max:
            input = -max



class FocalLoss(nn.Module):
    """
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)^gamma*log(pt)
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index, should be specific when alpha is float
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """
 
    def __init__(self, num_class, alpha=None, gamma=2, balance_index=-1, smooth=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.num_class = num_class
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
        self.size_average = size_average
 
        if self.alpha is None:
            self.alpha = torch.ones(self.num_class, 1)
        elif isinstance(self.alpha, (list, np.ndarray)):
            assert len(self.alpha) == self.num_class
            self.alpha = torch.FloatTensor(alpha).view(self.num_class, 1)
            self.alpha = self.alpha / self.alpha.sum()
        elif isinstance(self.alpha, float):
            alpha = torch.ones(self.num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[balance_index] = self.alpha
            self.alpha = alpha
        else:
            raise TypeError('Not support alpha type')
 
        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')
 
    def forward(self, input, target):
        logit = F.softmax(input, dim=1)
 
        if logit.dim() > 2:

            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        target = target.view(-1, 1)
 



 
        epsilon = 1e-10
        alpha = self.alpha
        if alpha.device != input.device:
            alpha = alpha.to(input.device)
 
        idx = target.cpu().long()
        one_hot_key = torch.FloatTensor(target.size(0), self.num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)
 
        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth, 1.0 - self.smooth)
        pt = (one_hot_key * logit).sum(1) + epsilon
        logpt = pt.log()
 
        gamma = self.gamma
 
        alpha = alpha[idx]
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt
 
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


def random_unit(p: float):
    if p == 0:
        return False
    if p == 1:
        return True

    R = random.random()
    if R < p:
        return True
    else:
        return False