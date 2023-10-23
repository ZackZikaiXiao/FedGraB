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



class WENO():
    def __init__(self, class_num):
        self.class_num = class_num  # 类别数量
        self.class_wise_num = [0 for i in range(class_num)]     # 每个类别的累计样本总数，十个类别则长度为十
        self.assump_num = 10000     # 假设有global有10000个样本
        self.count = 0
        self.class_wise_num_for_3shot_calc = [] # 用来实时计算3shot类别
    # 根据weno估计每类别的样本数量和，返回3shot类别
    def weno_estimation(self, weight):
        self.count += 1
        # # fedavg的weight
        weno_w = copy.deepcopy(weight) 
        # # 合并feature extractor(当然连同classifier一起合并了)
        # for k in avg_w.keys():        
        #     avg_w[k] = avg_w[k] * dict_len[0] 
        #     for i in range(1, len(w)):
        #         avg_w[k] += w[i][k] * dict_len[i]
        #         #w_avg[k] += w[i][k]
        #     #w_avg[k] = w_avg[k] / len(w)
        #     avg_w[k] = avg_w[k] / sum(dict_len)
        # # 计算weightnorm
        weno_prop = torch.norm(weno_w["linear.weight"], p=2, dim=1)
        # 归一化
        weno_prop = self.normalization(weno_prop)
        # 打成正数
        one_weno_min = min(weno_prop)
        if one_weno_min < 0:
            for i in range(self.class_num):
                weno_prop[i] -= one_weno_min
        # 算成占比形式
        one_weno_sum = sum(weno_prop)
        for i in range(self.class_num):
            weno_prop[i] /= one_weno_sum

        # 转化为数量的形式并加到累计的class_wise_num上
        for id_cls in range(self.class_num):
            self.class_wise_num[id_cls] += (weno_prop[id_cls] * self.assump_num)

        # 计算3shot class
        self.class_wise_num_for_3shot_calc = [0 for i in range(self.class_num)]
        for i in range(self.class_num):
            self.class_wise_num_for_3shot_calc[i] = self.class_wise_num[i] / self.count
        three_shot_dict, _ = self.shot_split(self.class_wise_num_for_3shot_calc, threshold_3shot=[75, 95])

        print("每类预估数量", self.class_wise_num_for_3shot_calc)
        print("three_shot_dict: ", three_shot_dict)
        return three_shot_dict
    

    def shot_split(self, class_dtribution, threshold_3shot=[75, 95]):
            threshold_3shot = threshold_3shot  # percentage
            class_distribution = copy.deepcopy(class_dtribution)
            # num2classid2accumu_map[0]:number, num2classid2accumu_map[1]:class, num2classid2accumu_map[2]:cumulative number(percentage)
            map_num2classid2accumu = [[],[],[]]
            for classid in range(len(class_dtribution)):
                map_num2classid2accumu[0].append(class_distribution[classid])
                map_num2classid2accumu[1].append(classid)
            for i in range(len(map_num2classid2accumu[0])):
                for j in range(0,len(map_num2classid2accumu[0]) - i - 1):
                    if map_num2classid2accumu[0][j] < map_num2classid2accumu[0][j+1]:
                        map_num2classid2accumu[0][j], map_num2classid2accumu[0][j+1] = map_num2classid2accumu[0][j+1], map_num2classid2accumu[0][j]
                        map_num2classid2accumu[1][j], map_num2classid2accumu[1][j+1] = map_num2classid2accumu[1][j+1], map_num2classid2accumu[1][j]
            map_num2classid2accumu[2] = (np.cumsum(np.array(map_num2classid2accumu[0]), axis = 0) / sum(map_num2classid2accumu[0]) * 100).tolist()
            three_shot_dict = {"head":[], "middle":[], "tail":[]}   # containg the class id of head, middle and tail respectively
            three_shot_dict["head"] = map_num2classid2accumu[1][0:3]
            three_shot_dict["middle"] = map_num2classid2accumu[1][3:7]
            three_shot_dict["tail"] = map_num2classid2accumu[1][7:10]

            return three_shot_dict, map_num2classid2accumu

if __name__ == '__main__':
    wenoObj = WENO(10)
    three_shot_dict, _ = wenoObj.shot_split([233.33401879169074, 562.4485581279324, 0.0, 374.04921243391567, 631.9994807947138, 1090.6361017254749, 1311.1757275371456, 1680.9905318445149, 1913.4196257865085, 2201.946742958103], threshold_3shot=[50, 90])
    print(three_shot_dict)
    three_shot_dict, _ = wenoObj.shot_split([2196.6407197720323, 3845.5849282240865, 493.0641880923965, 596.8127633139422, 210.4596936057591, 670.4915154717304, 538.2452939568532, 177.6783504240507, 804.6148332270465, 466.4077139121017], threshold_3shot=[50, 90])
    print(three_shot_dict)
    three_shot_dict, _ = wenoObj.shot_split([900, 1000, 780, 600, 700, 500, 550, 430, 100, 200], threshold_3shot=[50, 90])
    print(three_shot_dict)