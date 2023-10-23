# @Time: 2022/7/26
# @Author: zikai
# @File: search the best result in model test output file

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 所有文件搜索
# # Input
# basefolder = '/home/zikaixiao/zikai/aaFL/fl_framework_lss/temp' # path
# keyword = "weight"


# # basefolder = os.path.join(basefolder_pre,'fl_framework_lss', 'temp')
# filelist = os.listdir(basefolder)
# print(filelist)
# filelist.sort()

# for file in filelist:   # for each file
#     context = []
#     fpath = os.path.join(basefolder,file)
#     f = open(fpath,'r')
#     for line in f.readlines():
#         if keyword in line:     # only the line containing keyword will be added to context
#             context.append(line)
#         else:
#             continue
#     evaluationList = []
#     for i in range(len(context)):
#         evaluationList.append(float(context[i].split(" ")[-2]))
#     print("File \"{0}\" containing keyword \"{1}\" reaches to \"{2}\"".format(file, keyword, max(evaluationList)))

  # 根据avg 3shot找best
# keyword = "global 3shot acc"
# # fpath = "/home/zikaixiao/zikai/aaFL/fl_framework_lss/前向后向都没有weight_if100.log"
# # fpath = "/home/zikaixiao/zikai/aaFL/fl_framework_lss/weightmask[0,6]_if100.log"
# fpath = "/home/zikaixiao/zikai/aaFL/fl_framework_lss/todo1_global_lt_3shot_balance.log"
# # fpath = "/home/zikaixiao/zikai/aaFL/fl_framework_lss/eql参数8.log"
# context = []
# # fpath = os.path.join(basefolder,file)
# f = open(fpath,'r')
# for line in f.readlines():
#     if keyword in line:     # only the line containing keyword will be added to context
#         context.append(line)
#     else:
#         continue
# evaluationList = []
# heads = []
# middles = []
# tails = []
# for i in range(len(context)):
#     heads.append(float(context[i].split(" ")[-6][:-1]))
#     middles.append(float(context[i].split(" ")[-4][:-1]))
#     tails.append(float(context[i].split(" ")[-2][:-1]))
#     evaluationList.append((heads[-1] + middles[-1] + tails[-1]) / 3)
# # search the top acc line
# loc = 0
# max_item = 0
# for i in range(len(evaluationList)):
#     if evaluationList[i] > max_item:
#         max_item = evaluationList[i] 
#         loc = i
    
# print("File \"{0}\" containing keyword \"{1}\" reaches to \"{2}\"".format(fpath.split('/')[-1], keyword, max(evaluationList)))
# print(context[loc])


# 根据总的acc找
keyword = "global test"
# fpath = "/home/zikaixiao/zikai/aaFL/fl_framework_lss/前向后向都没有weight_if100.log"
# fpath = "/home/zikaixiao/zikai/aaFL/fl_framework_lss/weightmask[0,6]_if100.log"
fpath = "/home/zikaixiao/zikai/aaFL/aaFL213/pfl_pid/GBA_Layer_if100_alpha05_全局DPA_3.log"
# fpath = "/home/zikaixiao/zikai/aaFL/fl_framework_lss/eql参数8.log"
context = []
# fpath = os.path.join(basefolder,file)
f = open(fpath,'r')
for line in f.readlines():
    if keyword in line:     # only the line containing keyword will be added to context
        context.append(line)
    else:
        continue
evaluationList = []
heads = []
middles = []
tails = []
for i in range(len(context)):
    evaluationList.append(float(context[i].split(" ")[-2]))
# search the top acc line
loc = 0
max_item = 0
for i in range(len(evaluationList)):
    if evaluationList[i] > max_item:
        max_item = evaluationList[i] 
        loc = i
    
print("File \"{0}\" containing keyword \"{1}\" reaches to \"{2}\"".format(fpath.split('/')[-1], keyword, max(evaluationList)))
print(context[loc])
