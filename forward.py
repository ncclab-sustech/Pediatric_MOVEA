import numpy as np
# 
import math
from public import glo
#import glo

inf = 9999999999
NUM_ELE = 75
POINT_NUM = 1000



print("loading 0026049")
# grey and white matter
lfm = np.load(r'/mnt/database7/zeming_L/lfm_0026049_2.npy')
print("load grey and white matter",lfm.shape)
#lfm = lfm[[0,1,2,9,11,13,15,17,31,33,36,38,52,54,56,58,60,69,70,71]]
# for train
pos = np.load(r'/mnt/database7/zeming_L/pos_0026049_2.npy')
#pos = np.load('/home/ncclab306/database7/wangm/simnibs_leadfield/pos_269731_mni.npy')
print("load position")
print(pos.shape)

position =[-36,-19,48] #motor 这里取了对侧
# position =[10, -19, 6] #thalamus
# position =[-31, -20, -14] #hippo
# position =[-39, 34, 37] #dlpfc
# position =[-17, 3, -1] #pallidum
# position =[14, -99, -3] #v1


#position =[-36,-19,48] #motor
#position =[10, -19, 6] #thalamus
#position =[-31, -20, -14] #hippo
#position =[-39, 34, 37] #dlpfc
#position =[-17, 3, -1] #pallidum
#position =[14, -99, -3] #v1

#motor
#[-31, -20, -14] #[47, -13, 52] #[-29,-19,-15] #[-29,-19,-15]#LEFT-HIPPO #[25,3,-1] paliidum #[41,-36,66] sensory #[25,42,37]dorsal # v1 [11,-94,0]   # MNI_coords #r_dkpfc [52, 38, 15] #motor [44, -18, 57]
distance = np.zeros(len(pos))
print(position)

for i in range(len(pos)):
    distance[i] = (pos[i, 0] - position[0])**2 + (pos[i, 1] - position[1])**2 + (pos[i, 2] - position[2])**2
rank = np.argsort(distance)
# index = np.argmin(distance)
# TARGET_POSITION = index
# index = np.argsort(distance)
# TARGET_POSITION = index[:POINT_NUM]
TARGET_POSITION = np.where(distance < 100)
TARGET_POSITION = TARGET_POSITION[0]
print(len(TARGET_POSITION))
# print("no is ", index[0])



def tdcs_function1(x):
    x = np.array(x)
    x[abs(x) < 0.01] = 0
    return  1000 / np.average(((np.matmul(lfm[:, TARGET_POSITION, 0].T, x)) ** 2 + (
        np.matmul(lfm[:, TARGET_POSITION, 1].T, x)) ** 2 + (np.matmul(lfm[:, TARGET_POSITION, 2].T, x)) ** 2) ** 0.5) 







# def tdcs_function_global(x):
#     x = np.array(x)
#     x[abs(x) < 0.01] = 0  # 这一步是为了过滤掉非常小的值，减少计算误差
#     field_strength = ((np.matmul(lfm[:, :, 0].T, x)) ** 2 + (np.matmul(lfm[:, :, 1].T, x)) ** 2 + (np.matmul(lfm[:, :, 2].T, x)) ** 2) ** 0.5
#     average_field_strength = np.mean(field_strength)  # 计算全脑的平均场强度
#     return average_field_strength

def tdcs_function2(x):
    x = np.array(x)
    x[abs(x) < 0.01] = 0
    return np.average(((np.matmul(lfm[:, :, 0].T, x)) ** 2 + (
        np.matmul(lfm[:, :, 1].T, x)) ** 2 + (np.matmul(lfm[:, :, 2].T, x)) ** 2) ** 0.5) /1000

x= np.zeros(75)
x[31]=1
x[32]=1
x[20]=-1
x[21]=-1

# results=tdcs_function2(x)
# print(results)
results=tdcs_function1(x)
print(1/results)
print(tdcs_function2(x))
