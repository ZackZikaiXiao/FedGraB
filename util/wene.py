
class WENO():
    def __init__(self, class_num):
        self.class_num = class_num  
        self.class_wise_num = [0 for i in range(class_num)]   
        self.assump_num = 10000   
        self.count = 0
        self.class_wise_num_for_3shot_calc = [] 

    def weno_estimation(self, weight):
        self.count += 1

        weno_w = copy.deepcopy(weight) 

        weno_prop = torch.norm(weno_w["linear.weight"], p=2, dim=1)

        weno_prop = self.normalization(weno_prop)

        one_weno_min = min(weno_prop)
        if one_weno_min < 0:
            for i in range(self.class_num):
                weno_prop[i] -= one_weno_min

        one_weno_sum = sum(weno_prop)
        for i in range(self.class_num):
            weno_prop[i] /= one_weno_sum

        for id_cls in range(self.class_num):
            self.class_wise_num[id_cls] += (weno_prop[id_cls] * self.assump_num)

        self.class_wise_num_for_3shot_calc = [0 for i in range(self.class_num)]
        for i in range(self.class_num):
            self.class_wise_num_for_3shot_calc[i] = self.class_wise_num[i] / self.count
        three_shot_dict, _ = self.shot_split(self.class_wise_num_for_3shot_calc, threshold_3shot=[75, 95])

        print("Estimated quantity for each category", self.class_wise_num_for_3shot_calc)
        print("three_shot_dict: ", three_shot_dict)
        return three_shot_dict
    
    def visualization(self, round):
        fig = plt.figure(dpi = 500)
        x = range(len(self.class_wise_num_for_3shot_calc))
        plt.plot(x, self.class_wise_num_for_3shot_calc, label='weight_norm')
        plt.savefig("./class_wise_num/" +"round"+str(round))
        plt.close(fig)

    def normalization(self, list):
        list = list.cpu()
        list = np.array(list)
        list = (list - list.mean()) / list.std()
        list = list.tolist()
        return list

    def shot_split(self, class_dtribution, threshold_3shot=[75, 95]):
        threshold_3shot = threshold_3shot  # percentage
        class_distribution = copy.deepcopy(class_dtribution)
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
        

        cut1 = 0
        cut2 = 0
        accu_range_auxi = [0] + map_num2classid2accumu[2]
        accu_range = copy.deepcopy(accu_range_auxi)
        for i in range(1, len(accu_range)):
            accu_range[i] = [accu_range_auxi[i-1], accu_range_auxi[i]]
        del accu_range[0]
        for i in range(len(accu_range)):
            if threshold_3shot[0] > accu_range[i][0] and threshold_3shot[0] <= accu_range[i][1]:
                cut1 = i
            if threshold_3shot[1] > accu_range[i][0] and threshold_3shot[1] <= accu_range[i][1]:
                cut2 = i

        for i in range(len(map_num2classid2accumu[1])):
            if i <= cut1:
                three_shot_dict["head"].append(map_num2classid2accumu[1][i])
            elif i > cut1 and i <= cut2:
                three_shot_dict["middle"].append(map_num2classid2accumu[1][i])
            else:
                three_shot_dict["tail"].append(map_num2classid2accumu[1][i])

        return three_shot_dict, map_num2classid2accumu