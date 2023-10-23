# 画单张图

# import re
# import matplotlib.pyplot as plt

# file_path = '/home/zikaixiao/zikai/aaFL/aaFL213/pfl_pid/GBA_Layer_todo3_kp100_kd2.log'

# def extract_global_test_acc(file_path):
#     global_test_acc_values = []
#     with open(file_path, 'r') as file:
#         for line in file:
#             if "global test acc" in line:
#                 value = float(re.findall("\d+\.\d+", line)[0])
#                 if 0 <= value <= 1:
#                     global_test_acc_values.append(value)
#     return global_test_acc_values

# global_test_acc_values = extract_global_test_acc(file_path)
# print(global_test_acc_values)



# def plot_and_save_global_test_acc(global_test_acc_values):
#     plt.plot(global_test_acc_values)
#     plt.xlabel("Round")
#     plt.ylabel("Global Test Acc")
#     plt.title("Global Test Acc per Round")
#     plt.savefig("global_test_acc_plot.png", dpi=300)
#     plt.close()

# plot_and_save_global_test_acc(global_test_acc_values)




import re
import matplotlib.pyplot as plt

file_path_1 = '/home/zikaixiao/zikai/aaFL/aaFL213/pfl_pid/GBA_Layer_todo3_kp100_kd0.log'
file_path_2 = '/home/zikaixiao/zikai/aaFL/aaFL213/pfl_pid/GBA_Layer_todo3_kp100_kd2.log'

def extract_global_test_acc(file_path):
    global_test_acc_values = []
    with open(file_path, 'r') as file:
        for line in file:
            if "global test acc" in line:
                value = float(re.findall("\d+\.\d+", line)[0])
                if 0 <= value <= 1:
                    global_test_acc_values.append(value)
    return global_test_acc_values

global_test_acc_values_1 = extract_global_test_acc(file_path_1)
print(global_test_acc_values_1)

global_test_acc_values_2 = extract_global_test_acc(file_path_2)
print(global_test_acc_values_2)

def plot_and_save_global_test_acc(global_test_acc_values_1, global_test_acc_values_2):
    plt.plot(global_test_acc_values_1, color='blue', linewidth = 1, label='GBA_Layer_todo3_kp100_kd0.log')
    plt.plot(global_test_acc_values_2, color='red', linewidth = 1, label='GBA_Layer_todo3_kp100_kd2.log')
    plt.xlabel("Round")
    plt.ylabel("Global Test Acc")
    plt.title("Global Test Acc per Round")
    plt.legend(loc='best')
    plt.savefig("global_test_acc_plot.png", dpi=300)
    plt.close()

plot_and_save_global_test_acc(global_test_acc_values_1, global_test_acc_values_2)
