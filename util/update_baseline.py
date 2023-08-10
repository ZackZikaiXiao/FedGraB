# python version 3.7.1
# -*- coding: utf-8 -*-

from cProfile import label
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import numpy as np
from util.util import shot_split


import sklearn.metrics as metrics
from util.losses import FocalLoss



# 通过cient的id来划分local longtail的数据
class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class LocalUpdate(object):
    def __init__(self, args, dataset, idxs):
        self.args = args
        # self.loss_func = self.get_loss()  # loss function -- cross entropy
        # self.loss_func = nn.CrossEntropyLoss() # loss function -- cross entropy
        self.ldr_train, self.ldr_test = self.train_test(dataset, list(idxs))

    def get_loss(self):
        if self.args.loss_type == 'CE':
            return nn.CrossEntropyLoss()
        elif self.args.loss_type == 'focal':
            return FocalLoss(gamma=1).cuda(self.args.gpu)

    def train_test(self, dataset, idxs):
        # split training set, validation set and test set
        train = DataLoader(DatasetSplit(dataset, idxs),
                   batch_size=self.args.local_bs, shuffle=True,
                   num_workers=4, pin_memory=True)
        test = DataLoader(dataset, batch_size=128, num_workers=4, pin_memory=True)
        return train, test

    def update_weights(self, net, seed, net_glob, epoch, mu=1, lr=None):
        net_glob = net_glob

        net.train()
        # train and update
        if lr is None:
            optimizer = torch.optim.SGD(
                net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        else:
            optimizer = torch.optim.SGD(
                net.parameters(), lr=lr, momentum=self.args.momentum)

        epoch_loss = []
        for iter in range(epoch):
            batch_loss = []
            # label_debug = [0 for i in range(100)]       ######
            # use/load data from split training set "ldr_train"
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(
                    self.args.device), labels.to(self.args.device)
                # for label in labels:
                    # label_debug[label] += 1         #########
                labels = labels.long()
                net.zero_grad()
                logits = net(images)
                criterion = self.get_loss()
                loss = criterion(logits, labels)

                if self.args.beta > 0:
                    if batch_idx > 0:
                        w_diff = torch.tensor(0.).to(self.args.device)
                        for w, w_t in zip(net_glob.parameters(), net.parameters()):
                            w_diff += torch.pow(torch.norm(w - w_t), 2)
                        w_diff = torch.sqrt(w_diff)
                        loss += self.args.beta * mu * w_diff

                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            # print(label_debug)
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

    # 专为获取梯度信息设计的local update函数
    def update_weights_gasp_grad(self, net, seed, net_glob, client_id, epoch, gradBag, mu=1, lr=None):
        hookObj,  gradAnalysor = gradBag.get_client(client_id)
        net_glob = net_glob
        net.train()
        # train and update
        if lr is None:
            optimizer = torch.optim.SGD(
                net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        else:
            optimizer = torch.optim.SGD(
                net.parameters(), lr=lr, momentum=self.args.momentum)

        epoch_loss = []
        for iter in range(epoch):
            batch_loss = []
            # use/load data from split training set "ldr_train"
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(
                    self.args.device), labels.to(self.args.device)

                labels = labels.long()
                net.zero_grad()
                logits = net(images)
                criterion = self.get_loss()
                loss = criterion(logits, labels)
                hook_handle = logits.register_hook(
                    hookObj.hook_func_tensor)  # hook抓取梯度
                if self.args.beta > 0:
                    if batch_idx > 0:
                        w_diff = torch.tensor(0.).to(self.args.device)
                        for w, w_t in zip(net_glob.parameters(), net.parameters()):
                            w_diff += torch.pow(torch.norm(w - w_t), 2)
                        w_diff = torch.sqrt(w_diff)
                        loss += self.args.beta * mu * w_diff
                loss.backward()

                if hookObj.has_gradient():
                    # 输入一个batch的梯度和label
                    gradAnalysor.update(
                        hookObj.get_gradient(), labels.cpu().numpy().tolist())
                optimizer.step()
                hook_handle.remove()
                batch_loss.append(loss.item())
                gradAnalysor.print_for_debug()

            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        gradBag.load_in(client_id, hookObj, gradAnalysor)
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), gradBag
# 用pid算法进行local update
    def update_weights_fednova(self, net, seed, epoch, mu=1, lr=None, opitizer_nova=None):

        net.train()
        # train and update

        optimizer = opitizer_nova

        epoch_loss = []
        for iter in range(epoch):
            batch_loss = []
            # use/load data from split training set "ldr_train"
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(
                    self.args.device), labels.to(self.args.device)

                labels = labels.long()
                net.zero_grad()
                logits = net(images)
                criterion = self.get_loss()
                loss = criterion(logits, labels)

                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)
    def update_weights_GBA_Loss(self, net, seed, epoch, pidloss, mu=1, lr=None):
        
        net.train()
        # train and update
        if lr is None:
            optimizer = torch.optim.SGD(
                net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        else:
            optimizer = torch.optim.SGD(
                net.parameters(), lr=lr, momentum=self.args.momentum)

        epoch_loss = []
        for iter in range(epoch):
            batch_loss = []
            # use/load data from split training set "ldr_train"
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(
                    self.args.device), labels.to(self.args.device)

                labels = labels.long()
                net.zero_grad()
                logits = net(images)
                criterion = pidloss
                loss = criterion(logits, labels)

                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)


    def update_weights_GBA_Finetune(self, net, seed, epoch, pidloss, mu=1, lr=None):
        # 冻结一部分层
        count = 0
        for p in net.parameters():
            if count >= 105:        # 105; 108, 现在不frozen，所以改成0了
                break
            p.requires_grad = False
            count += 1

        filter(lambda p: p.requires_grad, net.parameters())
        net.train()
        # train and update
        if lr is None:
            optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, net.parameters()), lr=self.args.lr, momentum=self.args.momentum)
        else:
            optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=self.args.momentum)

        epoch_loss = []
        for iter in range(epoch):
            batch_loss = []
            # use/load data from split training set "ldr_train"
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(
                    self.args.device), labels.to(self.args.device)

                labels = labels.long()
                net.zero_grad()
                logits = net(images)
                criterion = pidloss
                # criterion = nn.CrossEntropyLoss()
                loss = criterion(logits, labels)

                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())
                # print(criterion.pn_diff[70])
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)
    
    def update_weights_fedir(self, net, seed, net_glob, epoch, criterion, mu=1, lr=None):
        net_glob = net_glob

        net.train()
        # train and update
        if lr is None:
            optimizer = torch.optim.SGD(
                net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        else:
            optimizer = torch.optim.SGD(
                net.parameters(), lr=lr, momentum=self.args.momentum)

        epoch_loss = []
        for iter in range(epoch):
            batch_loss = []
            # use/load data from split training set "ldr_train"
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(
                    self.args.device), labels.to(self.args.device)

                labels = labels.long()
                net.zero_grad()
                logits = net(images)
                # criterion = self.get_loss()
                loss = criterion(logits, labels)

                if self.args.beta > 0:
                    if batch_idx > 0:
                        w_diff = torch.tensor(0.).to(self.args.device)
                        for w, w_t in zip(net_glob.parameters(), net.parameters()):
                            w_diff += torch.pow(torch.norm(w - w_t), 2)
                        w_diff = torch.sqrt(w_diff)
                        loss += self.args.beta * mu * w_diff

                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)
    
    # 用pid算法进行local update
    def update_weights_GBA_Layer(self, net, seed, epoch, GBA_Loss, GBA_Layer, mu=1, lr=None):

        net.train()
        GBA_Layer.train()
        # train and update
        if lr is None:
            backbone_optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        else:
            backbone_optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=self.args.momentum)

        epoch_loss = []
        for iter in range(epoch):
            batch_loss = []
            # label_sum = [0 for i in range(10)]
            # use/load data from split training set "ldr_train"
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)

                labels = labels.long()
                # for i in labels:
                #     label_sum[i] += 1

                # init
                net.zero_grad()
                feat = net(images, latent_output=True)
                logits = GBA_Layer(feat)
                loss = GBA_Loss(logits, labels) 
                loss.backward()

                # torch.nn.utils.clip_grad_norm_(net.parameters(), 10)

                max_grad = max(p.grad.data.abs().max() for p in net.parameters() if p.grad is not None)
                print('Max gradient:', max_grad)
                
                

                backbone_optimizer.step()
                
                # print(GBA_Loss.pn_diff[52])
                # loss
                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        # print("-----------------------------------------------------------------------")
        return net.state_dict(), GBA_Layer.state_dict(), sum(epoch_loss) / len(epoch_loss)
# global dataset is balanced

    def update_weights_GBA_Layer_finetune(self, net, seed, epoch, GBA_Loss, GBA_Layer_1, GBA_Layer_2, mu=1, lr=None):

        net.train()
        GBA_Layer_1.train()
        GBA_Layer_2.train()
        # train and update
        if lr is None:
            optimizer = torch.optim.SGD(list(net.parameters()) + list(GBA_Layer_1.parameters()), lr=self.args.lr, momentum=self.args.momentum)
            optimizer_gba = torch.optim.SGD(GBA_Layer_2.parameters(), lr=self.args.lr, momentum=self.args.momentum)  # 创建新的优化器，只包含GBA_Layer_2的参数
        else:
            optimizer = torch.optim.SGD(list(net.parameters()) + list(GBA_Layer_1.parameters()), lr=lr, momentum=self.args.momentum)
            optimizer_gba = torch.optim.SGD(GBA_Layer_2.parameters(), lr=lr, momentum=self.args.momentum)  # 创建新的优化器，只包含GBA_Layer_2的参数

        epoch_loss = []
        ce_loss = nn.CrossEntropyLoss()
        for iter in range(epoch):
            batch_loss = []
            # label_sum = [0 for i in range(10)]
            # use/load data from split training set "ldr_train"
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)

                labels = labels.long()
                # for i in labels:
                #     label_sum[i] += 1

                # init
                # 首先使用ce_loss进行反向传播并更新所有参数
                net.zero_grad()
                feat = net(images, True)
                logits_1 = GBA_Layer_1(feat)
                logits_2 = GBA_Layer_2(logits_1)
                loss = ce_loss(logits_2, labels) 
                loss.backward()
                optimizer.step()

                # 然后使用GBA_Loss仅更新GBA_Layer_2的参数
                logits_1_detach = logits_1.detach()
                logits_2_detach = GBA_Layer_2(logits_1_detach)
                GBA_Loss_value = GBA_Loss(logits_2_detach, labels)  # 在计算GBA_Loss之前调用.detach() 
                optimizer_gba.zero_grad()  # 清零GBA_Layer_2的梯度
                GBA_Loss_value.backward()  # 使用GBA_Loss进行反向传播
                optimizer_gba.step()  # 更新GBA_Layer_2的参数
                
                # print(GBA_Loss.pn_diff[52])
                # loss
                # batch_loss.append(loss.item())
                batch_loss.append(loss.item() + GBA_Loss_value.item())

            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        # print("-----------------------------------------------------------------------")
        return net.state_dict(), GBA_Layer_1.state_dict(), GBA_Layer_2.state_dict(), sum(epoch_loss) / len(epoch_loss)
    

def globaltest(net, test_dataset, args, dataset_class=None):
    global_test_distribution = dataset_class.global_test_distribution
    three_shot_dict, _ = shot_split(global_test_distribution, threshold_3shot=[75, 95])
    correct_3shot = {"head": 0, "middle": 0, "tail": 0}
    total_3shot = {"head": 0, "middle": 0, "tail": 0}
    acc_3shot_global = {"head": None, "middle": None, "tail": None}
    net.eval()
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=100, shuffle=False)
    # 监视真实情况下所有样本的类别分布
    total_class_label = [0 for i in range(args.num_classes)]
    predict_true_class = [0 for i in range(args.num_classes)]
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(args.device)
            labels = labels.to(args.device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # class-wise acc calc
            for i in range(len(labels)):
                total_class_label[int(labels[i])] += 1      # total
                if predicted[i] == labels[i]:
                    predict_true_class[int(labels[i])] += 1

            # start: cal 3shot metrics
            for label in labels:
                if label in three_shot_dict["head"]:
                    total_3shot["head"] += 1
                elif label in three_shot_dict["middle"]:
                    total_3shot["middle"] += 1
                else:
                    total_3shot["tail"] += 1
            for i in range(len(predicted)):
                if predicted[i] == labels[i] and labels[i] in three_shot_dict["head"]:   # 预测正确且在head中
                    correct_3shot["head"] += 1
                # 预测正确且在middle中
                elif predicted[i] == labels[i] and labels[i] in three_shot_dict["middle"]:
                    correct_3shot["middle"] += 1
                # 预测正确且在tail中
                elif predicted[i] == labels[i] and labels[i] in three_shot_dict["tail"]:
                    correct_3shot["tail"] += 1      # 在tail中
            # end
    acc_class_wise = [predict_true_class[i] / total_class_label[i] for i in range(args.num_classes)]
    acc = correct / total
    acc_3shot_global["head"] = correct_3shot["head"] / \
        (total_3shot["head"] + 1e-10)
    acc_3shot_global["middle"] = correct_3shot["middle"] / \
        (total_3shot["middle"] + 1e-10)
    acc_3shot_global["tail"] = correct_3shot["tail"] / \
        (total_3shot["tail"] + 1e-10)
    return acc, acc_3shot_global

# global dataset is balanced
def globaltest_GBA_Layer(backbone, classifier, test_dataset, args, dataset_class = None):
    global_test_distribution = dataset_class.global_test_distribution
    three_shot_dict, _ = shot_split(global_test_distribution, threshold_3shot=[75, 95])
    correct_3shot = {"head":0, "middle":0, "tail":0}    #######
    total_3shot = {"head":0, "middle":0, "tail":0} 
    acc_3shot_global = {"head":None, "middle":None, "tail":None}
    backbone.eval()
    classifier.eval()
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=100, shuffle=False)
    # 监视真实情况下所有样本的类别分布
    distri_class_real = [0 for i in range(100)]
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(args.device)
            labels = labels.to(args.device)
            feat = backbone(images)
            outputs = classifier(feat)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # start: cal 3shot metrics
            for label in labels:
                distri_class_real[int(label)] += 1      # 监视真实情况下所有样本的类别分布
                if label in three_shot_dict["head"]:
                    total_3shot["head"] += 1
                elif label in three_shot_dict["middle"]:
                    total_3shot["middle"] += 1
                else:
                    total_3shot["tail"] += 1
            for i in range(len(predicted)):
                if predicted[i] == labels[i] and labels[i] in three_shot_dict["head"]:   # 预测正确且在head中
                    correct_3shot["head"] += 1
                elif predicted[i] == labels[i] and labels[i] in three_shot_dict["middle"]:   # 预测正确且在middle中
                    correct_3shot["middle"] += 1
                elif predicted[i] == labels[i] and labels[i] in three_shot_dict["tail"]:   # 预测正确且在tail中
                    correct_3shot["tail"] += 1      # 在tail中
            # end 

    acc = correct / total
    acc_3shot_global["head"] = correct_3shot["head"] / (total_3shot["head"] + 1e-10)
    acc_3shot_global["middle"] = correct_3shot["middle"] / (total_3shot["middle"] + 1e-10)
    acc_3shot_global["tail"] = correct_3shot["tail"] / (total_3shot["tail"] + 1e-10)
    return acc, acc_3shot_global


def globaltest_GBA_Integrate(backbone, classifier, test_dataset, args, dataset_class = None):
    global_test_distribution = dataset_class.global_test_distribution
    three_shot_dict, _ = shot_split(global_test_distribution, threshold_3shot=[75, 95])
    correct_3shot = {"head":0, "middle":0, "tail":0}    #######
    total_3shot = {"head":0, "middle":0, "tail":0} 
    acc_3shot_global = {"head":None, "middle":None, "tail":None}
    backbone.eval()
    classifier.eval()
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=100, shuffle=False)
    # 监视真实情况下所有样本的类别分布
    distri_class_real = [0 for i in range(100)]
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(args.device)
            labels = labels.to(args.device)
            feat = backbone(images)
            outputs = classifier(feat)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # start: cal 3shot metrics
            for label in labels:
                distri_class_real[int(label)] += 1      # 监视真实情况下所有样本的类别分布
                if label in three_shot_dict["head"]:
                    total_3shot["head"] += 1
                elif label in three_shot_dict["middle"]:
                    total_3shot["middle"] += 1
                else:
                    total_3shot["tail"] += 1
            for i in range(len(predicted)):
                if predicted[i] == labels[i] and labels[i] in three_shot_dict["head"]:   # 预测正确且在head中
                    correct_3shot["head"] += 1
                elif predicted[i] == labels[i] and labels[i] in three_shot_dict["middle"]:   # 预测正确且在middle中
                    correct_3shot["middle"] += 1
                elif predicted[i] == labels[i] and labels[i] in three_shot_dict["tail"]:   # 预测正确且在tail中
                    correct_3shot["tail"] += 1      # 在tail中
            # end 

    acc = correct / total
    acc_3shot_global["head"] = correct_3shot["head"] / (total_3shot["head"] + 1e-10)
    acc_3shot_global["middle"] = correct_3shot["middle"] / (total_3shot["middle"] + 1e-10)
    acc_3shot_global["tail"] = correct_3shot["tail"] / (total_3shot["tail"] + 1e-10)
    return acc, acc_3shot_global


def localtest(net, test_dataset, dataset_class, idxs, user_id):
    from sklearn.metrics import f1_score
    import copy
    args = dataset_class.get_args()
    net.eval()
    test_loader = torch.utils.data.DataLoader(DatasetSplit(
        test_dataset, idxs), batch_size=args.local_bs, shuffle=False)

    # get overall distribution
    # class_distribution = [0 for _ in range(10000)]  # 10000 >
    # for images, labels in test_loader:
    #     labels = labels.tolist()
    class_distribution_dict = {}

    class_distribution = dataset_class.local_test_distribution[user_id]
    three_shot_dict, _ = shot_split(
        class_distribution, threshold_3shot=[75, 95])
    # three_shot_dict: {"head":[], "middle":[], "tail":[]}   # containg the class id of head, middle and tail respectively

    ypred = []
    ytrue = []
    acc_3shot_local = {"head": None, "middle": None, "tail": None}

    with torch.no_grad():
        correct = 0
        total = 0
        correct_3shot = {"head": 0, "middle": 0, "tail": 0}
        total_3shot = {"head": 0, "middle": 0, "tail": 0}
        for images, labels in test_loader:
            # inference
            images = images.to(args.device)
            labels = labels.to(args.device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)

            # calc total metrics
            total += labels.size(0)     # numble of all samples
            # numble of correct predictions
            correct += (predicted == labels).sum().item()
            predicted = predicted.tolist()
            gts = copy.deepcopy(labels)
            gts = gts.tolist()
            ypred.append(predicted)
            ytrue.append(gts)
            # f1 = f1_score(y_true=labels,y_pred=predicted)
            # print(f1)
            # all_f1.append(f1)

            # start: cal 3shot metrics
            for label in labels:
                if label in three_shot_dict["head"]:
                    total_3shot["head"] += 1
                elif label in three_shot_dict["middle"]:
                    total_3shot["middle"] += 1
                else:
                    total_3shot["tail"] += 1
            for i in range(len(predicted)):
                if predicted[i] == labels[i] and labels[i] in three_shot_dict["head"]:   # 预测正确且在head中
                    correct_3shot["head"] += 1
                # 预测正确且在middle中
                elif predicted[i] == labels[i] and labels[i] in three_shot_dict["middle"]:
                    correct_3shot["middle"] += 1
                # 预测正确且在tail中
                elif predicted[i] == labels[i] and labels[i] in three_shot_dict["tail"]:
                    correct_3shot["tail"] += 1      # 在tail中
            # end

    ypred = sum(ypred, [])
    ytrue = sum(ytrue, [])
    # print(ypred)
    # print(ytrue)
    f1_macro = f1_score(y_true=ytrue, y_pred=ypred, average='macro')
    f1_weighted = f1_score(y_true=ytrue, y_pred=ypred, average='weighted')
    # print(f1)
    # import pdb;pdb.set_trace()
    acc = correct / total

    # start: calc acc_3shot_local
    # acc_3shot_local["head"] = [0, False],False代表无效，平均的时候分母减1
    # 分布不为0，如果没有head，则返回-1，-1不参与平均计算
    acc_3shot_local["head"] = [0, False] if total_3shot["head"] == 0 else [
        (correct_3shot["head"] / total_3shot["head"]), True]
    acc_3shot_local["middle"] = [0, False] if total_3shot["middle"] == 0 else [
        (correct_3shot["middle"] / total_3shot["middle"]), True]
    acc_3shot_local["tail"] = [0, False] if total_3shot["tail"] == 0 else [
        (correct_3shot["tail"] / total_3shot["tail"]), True]
    # end

    # print("F1: "+ str(np.mean(f1)))
    return acc, f1_macro, f1_weighted, acc_3shot_local


def calculate_metrics(pred_np, seg_np):
    # pred_np: B,N
    # seg_np: B,N
    b = len(pred_np)
    all_f1 = []
    all_sensitivity = []
    all_specificity = []
    all_ppv = []
    all_npv = []
    for i in range(b):

        f1 = metrics.f1_score(seg_np[i], pred_np[i], average='macro')

        # confusion_matrix = metrics.confusion_matrix(seg_np[i], pred_np[i])  # n_class * n_class(<=17)
        # FP = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)   # n_class，
        # FN = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)   # n_class，
        # TP = np.diag(confusion_matrix)                                  # n_class，
        # TN = confusion_matrix.sum() - (FP + FN + TP)                    # n_class，

        # TPR = []
        # PPV = []
        # for j in range(len(TP)):
        #     if (TP[j] + FN[j]) == 0:
        #         TPR.append(1)
        #     else:
        #         TPR.append(TP[j] / (TP[j] + FN[j]))
        # for j in range(len(TP)):
        #     if (TP[j] + FP[j]) == 0:
        #         PPV.append(1)
        #     else:
        #         PPV.append(TP[j] / (TP[j] + FP[j]))
        # # # Sensitivity, hit rate, recall, or true positive rate
        # # TPR = TP / (TP + FN)
        # # Specificity or true negative rate
        # TNR = TN / (TN + FP)
        # # # Precision or positive predictive value
        # # PPV = TP / (TP + FP)
        # # Negative predictive value
        # NPV = TN / (TN + FN)

        all_f1.append(f1)
        # all_ppv.append(np.mean(PPV))
        # all_npv.append(np.mean(NPV))
        # all_sensitivity.append(np.mean(TPR))
        # all_specificity.append(np.mean(TNR))
    # return all_f1, all_ppv, all_npv, all_sensitivity, all_specificity  # B,
    return all_f1
