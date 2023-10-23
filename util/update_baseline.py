


from cProfile import label
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import numpy as np 
from util.util import shot_split


import sklearn.metrics as metrics
from util.losses import FocalLoss



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


        self.ldr_train, self.ldr_test = self.train_test(dataset, list(idxs))

    def get_loss(self):
        if self.args.loss_type == 'CE':
            return nn.CrossEntropyLoss()
        elif self.args.loss_type == 'focal':
            return FocalLoss(gamma=1).cuda(self.args.gpu)



    def train_test(self, dataset, idxs):

        train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        test = DataLoader(dataset, batch_size=128)
        return train, test

    def update_weights(self, net, seed, net_glob, epoch, mu=1, lr=None):
        net_glob = net_glob

        net.train()

        if lr is None:
            optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        else:
            optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=self.args.momentum)

        epoch_loss = []
        for iter in range(epoch):
            batch_loss = []

            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)

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
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def update_weights_pid(self, net, seed, epoch, GBA_Loss, GBA_Layer, mu=1, lr=None):

        net.train()
        GBA_Layer.train()

        if lr is None:
            backbone_optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        else:
            backbone_optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=self.args.momentum)

        epoch_loss = []
        for iter in range(epoch):
            batch_loss = []


            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)

                labels = labels.long()




                net.zero_grad()
                feat = net(images)
                logits = GBA_Layer(feat)
                loss = GBA_Loss(logits, labels) 
                loss.backward()
                backbone_optimizer.step()
                

                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return net.state_dict(), GBA_Layer.state_dict(), sum(epoch_loss) / len(epoch_loss)



    def update_weights_gasp_grad(self, net, seed, net_glob, client_id, epoch, gradBag, mu=1, lr=None):
        hookObj,  gradAnalysor = gradBag.get_client(client_id)
        net_glob = net_glob
        net.train()

        if lr is None:
            optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        else:
            optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=self.args.momentum)

        epoch_loss = []
        for iter in range(epoch):
            batch_loss = []

            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)

                labels = labels.long()
                net.zero_grad()
                logits = net(images)
                criterion = self.get_loss()
                loss = criterion(logits, labels)               
                hook_handle = logits.register_hook(hookObj.hook_func_tensor) 
                if self.args.beta > 0:
                    if batch_idx > 0:
                        w_diff = torch.tensor(0.).to(self.args.device)
                        for w, w_t in zip(net_glob.parameters(), net.parameters()):
                            w_diff += torch.pow(torch.norm(w - w_t), 2)
                        w_diff = torch.sqrt(w_diff)
                        loss += self.args.beta * mu * w_diff
                loss.backward()
            
                if hookObj.has_gradient():
                    gradAnalysor.update(hookObj.get_gradient(), labels.cpu().numpy().tolist()) 
                optimizer.step()
                hook_handle.remove()
                batch_loss.append(loss.item())
                gradAnalysor.print_for_debug()

            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        gradBag.load_in(client_id, hookObj, gradAnalysor)
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), gradBag


def globaltest(backbone, classifier, test_dataset, args, dataset_class = None):
    global_test_distribution = dataset_class.global_test_distribution
    three_shot_dict, _ = shot_split(global_test_distribution, threshold_3shot=[75, 95])
    three_shot_dict["head"] = [0, 1]
    three_shot_dict["middle"] = [2, 3, 4, 5]
    three_shot_dict["tail"] = [6, 7, 8, 9]
    correct_3shot = {"head":0, "middle":0, "tail":0} 
    total_3shot = {"head":0, "middle":0, "tail":0} 
    acc_3shot_global = {"head":None, "middle":None, "tail":None}
    backbone.eval()
    classifier.eval()
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=100, shuffle=False)

    distri_class_label = [0 for i in range(10)]
    distri_class_correct = [0 for i in range(10)]
    
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


            for i in range(len(labels)):
                label = labels[i]
                distri_class_label[int(label)] += 1 
                if predicted[i] == label:
                    distri_class_correct[label] += 1

            for label in labels:
                if label in three_shot_dict["head"]:
                    total_3shot["head"] += 1
                elif label in three_shot_dict["middle"]:
                    total_3shot["middle"] += 1
                else:
                    total_3shot["tail"] += 1
            for i in range(len(predicted)):
                if predicted[i] == labels[i] and labels[i] in three_shot_dict["head"]: 
                    correct_3shot["head"] += 1
                elif predicted[i] == labels[i] and labels[i] in three_shot_dict["middle"]: 
                    correct_3shot["middle"] += 1
                elif predicted[i] == labels[i] and labels[i] in three_shot_dict["tail"]: 
                    correct_3shot["tail"] += 1 

    acc_per_class = [distri_class_correct[i] / distri_class_label[i] for i in range(10)]
    print("acc_per_class:", acc_per_class)
    acc = correct / total
    acc_3shot_global["head"] = correct_3shot["head"] / (total_3shot["head"] + 1e-10)
    acc_3shot_global["middle"] = correct_3shot["middle"] / (total_3shot["middle"] + 1e-10)
    acc_3shot_global["tail"] = correct_3shot["tail"] / (total_3shot["tail"] + 1e-10)
    return acc, acc_3shot_global

def localtest(net, g_linear, p_linear, test_dataset, dataset_class, idxs, user_id, conv_beta=[1, 1]):
    from sklearn.metrics import f1_score
    import copy
    args = dataset_class.get_args()
    net.eval()
    g_linear.eval()
    p_linear.eval()
    test_loader = torch.utils.data.DataLoader(DatasetSplit(test_dataset,idxs), batch_size=args.local_bs, shuffle=False)





    class_distribution_dict = {}

    class_distribution = dataset_class.local_test_distribution[user_id]
    three_shot_dict, _ = shot_split(class_distribution, threshold_3shot=[75, 95])



    ypred = []
    ytrue = []
    acc_3shot_local = {"head":None, "middle":None, "tail":None}

    with torch.no_grad():
        correct = 0
        total = 0
        correct_3shot = {"head":0, "middle":0, "tail":0} 
        total_3shot = {"head":0, "middle":0, "tail":0} 
        for images, labels in test_loader:

            images = images.to(args.device)
            labels = labels.to(args.device)
            feat = net(images)
            g_output = g_linear(feat)
            p_output = p_linear(feat)
            outputs = conv_beta[0] * g_output + conv_beta[1] * p_output

            _, predicted = torch.max(outputs.data, 1)
            

            total += labels.size(0) 
            correct += (predicted == labels).sum().item() 
            predicted = predicted.tolist()
            gts = copy.deepcopy(labels)
            gts = gts.tolist()
            ypred.append(predicted)
            ytrue.append(gts)





            for label in labels:
                if label in three_shot_dict["head"]:
                    total_3shot["head"] += 1
                elif label in three_shot_dict["middle"]:
                    total_3shot["middle"] += 1
                else:
                    total_3shot["tail"] += 1
            for i in range(len(predicted)):
                if predicted[i] == labels[i] and labels[i] in three_shot_dict["head"]: 
                    correct_3shot["head"] += 1
                elif predicted[i] == labels[i] and labels[i] in three_shot_dict["middle"]: 
                    correct_3shot["middle"] += 1
                elif predicted[i] == labels[i] and labels[i] in three_shot_dict["tail"]: 
                    correct_3shot["tail"] += 1 


    ypred = sum(ypred,[])
    ytrue = sum(ytrue,[])


    f1_macro = f1_score(y_true=ytrue,y_pred=ypred,average='macro')
    f1_weighted = f1_score(y_true=ytrue,y_pred=ypred,average='weighted')


    acc = correct / total




    acc_3shot_local["head"] = [0, False] if total_3shot["head"] == 0 else [(correct_3shot["head"] / total_3shot["head"]), True]
    acc_3shot_local["middle"] = [0, False] if total_3shot["middle"] == 0 else [(correct_3shot["middle"] / total_3shot["middle"]), True]
    acc_3shot_local["tail"] = [0, False] if total_3shot["tail"] == 0 else [(correct_3shot["tail"] / total_3shot["tail"]), True]


    

    return acc,f1_macro,f1_weighted, acc_3shot_local


def calculate_metrics(pred_np, seg_np):


    b = len(pred_np)
    all_f1 = []
    all_sensitivity = []
    all_specificity = []
    all_ppv = []
    all_npv = []
    for i in range(b):
        
        f1 = metrics.f1_score(seg_np[i], pred_np[i], average='macro')




























        all_f1.append(f1)





    return all_f1




