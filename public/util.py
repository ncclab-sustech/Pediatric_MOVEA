import numpy as np
# 
import math
from public import glo
#import glo

inf = 9999999999
NUM_ELE = 75
POINT_NUM = 1000


if glo.head_model == '0026058_10':
    print("loading 0026058_10")
    # grey and white matter
    lfm = np.load(r'/home/ncclab/Desktop/MOVEA_conduc/data/lfm_0026058_10_2.npy')
    print("load grey and white matter",lfm.shape)
    #lfm = lfm[[0,1,2,9,11,13,15,17,31,33,36,38,52,54,56,58,60,69,70,71]]
    # for train
    pos = np.load(r'/home/ncclab/Desktop/MOVEA_conduc/data/pos_0026058_10_2.npy')
    #pos = np.load('/home/ncclab306/database7/wangm/simnibs_leadfield/pos_269731_mni.npy')
    print("load position")
    print(pos.shape)


if glo.head_model == 'data':
    print("loading data")
    # grey and white matter
    lfm = np.load(r'/media/ncclab/database7/zeming_L/电刺激项目/mopso/lfm_data_2.npy')
    print("load grey and white matter",lfm.shape)
    #lfm = lfm[[0,1,2,9,11,13,15,17,31,33,36,38,52,54,56,58,60,69,70,71]]
    # for train
    pos = np.load(r'/media/ncclab/database7/zeming_L/电刺激项目/mopso/pos_data_2.npy')
    #pos = np.load('/home/ncclab306/database7/wangm/simnibs_leadfield/pos_269731_mni.npy')
    print("load position")
    print(pos.shape)


if glo.head_model == 'pretomind':
    print("loading pretomind")
    # grey and white matter
    lfm = np.load(r'/media/ncclab/database7/zeming_L/电刺激项目/mopso/lfm_pretomid_2.npy')
    print("load grey and white matter",lfm.shape)
    #lfm = lfm[[0,1,2,9,11,13,15,17,31,33,36,38,52,54,56,58,60,69,70,71]]
    # for train
    pos = np.load(r'/media/ncclab/database7/zeming_L/电刺激项目/mopso/pos_pretomid_2.npy')
    #pos = np.load('/home/ncclab306/database7/wangm/simnibs_leadfield/pos_269731_mni.npy')
    print("load position")
    print(pos.shape)

if glo.head_model == 'CJH':
    print("loading CJH")
    # grey and white matter
    lfm = np.load(r'/media/ncclab/database7/zeming_L/电刺激项目/mopso/lfm_CJH_2.npy')
    print("load grey and white matter",lfm.shape)
    #lfm = lfm[[0,1,2,9,11,13,15,17,31,33,36,38,52,54,56,58,60,69,70,71]]
    # for train
    pos = np.load(r'/media/ncclab/database7/zeming_L/电刺激项目/mopso/pos_CJH_2.npy')
    #pos = np.load('/home/ncclab306/database7/wangm/simnibs_leadfield/pos_269731_mni.npy')
    print("load position")
    print(pos.shape)

if glo.head_model == 'zpl':
    print("loading zpl")
    # grey and white matter
    lfm = np.load(r'/media/ncclab/database7/zeming_L/电刺激项目/mopso/lfm_zpl_2.npy')
    print("load grey and white matter",lfm.shape)
    #lfm = lfm[[0,1,2,9,11,13,15,17,31,33,36,38,52,54,56,58,60,69,70,71]]
    # for train
    pos = np.load(r'/media/ncclab/database7/zeming_L/电刺激项目/mopso/pos_zpl_2.npy')
    #pos = np.load('/home/ncclab306/database7/wangm/simnibs_leadfield/pos_269731_mni.npy')
    print("load position")
    print(pos.shape)







# position = [42,37,31]#[-17, 3, -1] #pallidum

# #[-31, -20, -14] #[47, -13, 52] #[-29,-19,-15] #[-29,-19,-15]#LEFT-HIPPO #[25,3,-1] paliidum #[41,-36,66] sensory #[25,42,37]dorsal # v1 [11,-94,0]   # MNI_coords #r_dkpfc [52, 38, 15] #motor [44, -18, 57]
# distance = np.zeros(len(pos))
# print(position)

# for i in range(len(pos)):
#     distance[i] = (pos[i, 0] - position[0])**2 + (pos[i, 1] - position[1])**2 + (pos[i, 2] - position[2])**2

# # index = np.argmin(distance)
# # TARGET_POSITION = index
# AVOID_POSITION = np.where(distance < 4 )
# AVOID_POSITION = AVOID_POSITION[0]
# print(len(AVOID_POSITION))


position = glo.position
glo.position = np.array([-36,-19,48])
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
# print("position is ", position)
# print('index is ', pos[index[:POINT_NUM]])
# print('distance1', distance[index[0]])
# print('distance2', distance[index[100]])
# print('distance2', distance[index[300]])
# print('distance2', distance[index[600]])
# print('distance2', distance[index[POINT_NUM]])


def tis_function5(x1, x2, x3, x4,x5):
    eam = np.zeros(len(TARGET_POSITION))
    electrode1 = x1
    electrode2 = x2
    stimulation1 = np.zeros(NUM_ELE)
    stimulation1[electrode1] = 1 + 1/x5
    stimulation1[electrode2] = -1 - 1/x5
    e1 = np.array([np.matmul(lfm[:, TARGET_POSITION, 0].T, stimulation1), np.matmul(lfm[:, TARGET_POSITION, 1].T, stimulation1),np.matmul(lfm[:, TARGET_POSITION, 2].T, stimulation1)]).T /1000
    #print(np.max(e1))
    electrode3 = x3
    electrode4 = x4
    stimulation2 = np.zeros(NUM_ELE)
    # stimulation2[electrode3] = 1.5 - x[0]
    # stimulation2[electrode4] = -1.5 + x[0]
    stimulation2[electrode3] = 1 - 1/x5
    stimulation2[electrode4] = -1 + 1/x5
    e2 = np.array([np.matmul(lfm[:, TARGET_POSITION, 0].T, stimulation2), np.matmul(lfm[:, TARGET_POSITION, 1].T, stimulation2),np.matmul(lfm[:, TARGET_POSITION, 2].T, stimulation2)]).T / 1000
    for i in range(len(e1)):
        l_x = np.sqrt(e1[i].dot(e1[i]))
        l_y = np.sqrt(e2[i].dot(e2[i]))
        point = e1[i].dot(e2[i])
        cos_ = point / (l_x * l_y)
        # print(cos_)
        if cos_ <= 0:  # one of the fields must be flipped when a > 90 degrees
            e1[i] = -e1[i]
            cos_ = -cos_
        if l_y < l_x:
            if l_y < l_x * cos_:
                eam[i] = 2 * l_y
            else:
                eam[i] = 2 * np.linalg.norm(np.cross(e2[i], (e1[i] - e2[i]))) / np.linalg.norm(e1[i] - e2[i])
        else:
            if l_x < l_y * cos_:
                eam[i] = 2 * l_x
            else:
                eam[i] = 2 * np.linalg.norm(np.cross(e1[i], (e2[i] - e1[i]))) / np.linalg.norm(e2[i] - e1[i])
    # print(np.max(eam))
    return 1/np.mean(abs(eam)) # ,eam[TARGET_AVOID] ,dis


def tis_function_avoid(x):
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    x4 = x[3]
    x5 = x[4]
    eam = np.zeros(len(TARGET_POSITION))
    electrode1 = x1
    electrode2 = x2
    stimulation1 = np.zeros(NUM_ELE)
    stimulation1[electrode1] = 1 + 1/x5
    stimulation1[electrode2] = -1 - 1/x5
    e1 = np.array([np.matmul(lfm[:, TARGET_POSITION, 0].T, stimulation1), np.matmul(lfm[:, TARGET_POSITION, 1].T, stimulation1),np.matmul(lfm[:, TARGET_POSITION, 2].T, stimulation1)]).T /1000
    #print(np.max(e1))
    electrode3 = x3
    electrode4 = x4
    stimulation2 = np.zeros(NUM_ELE)
    # stimulation2[electrode3] = 1.5 - x[0]
    # stimulation2[electrode4] = -1.5 + x[0]
    stimulation2[electrode3] = 1 - 1/x5
    stimulation2[electrode4] = -1 + 1/x5
    e2 = np.array([np.matmul(lfm[:, TARGET_POSITION, 0].T, stimulation2), np.matmul(lfm[:, TARGET_POSITION, 1].T, stimulation2),np.matmul(lfm[:, TARGET_POSITION, 2].T, stimulation2)]).T / 1000
    for i in range(len(e1)):
        l_x = np.sqrt(e1[i].dot(e1[i]))
        l_y = np.sqrt(e2[i].dot(e2[i]))
        point = e1[i].dot(e2[i])
        cos_ = point / (l_x * l_y)
        # print(cos_)
        if cos_ <= 0:  # one of the fields must be flipped when a > 90 degrees
            e1[i] = -e1[i]
            cos_ = -cos_
        if l_y < l_x:
            if l_y < l_x * cos_:
                eam[i] = 2 * l_y
            else:
                eam[i] = 2 * np.linalg.norm(np.cross(e2[i], (e1[i] - e2[i]))) / np.linalg.norm(e1[i] - e2[i])
        else:
            if l_x < l_y * cos_:
                eam[i] = 2 * l_x
            else:
                eam[i] = 2 * np.linalg.norm(np.cross(e1[i], (e2[i] - e1[i]))) / np.linalg.norm(e2[i] - e1[i])
    target1 = 1/np.mean(abs(eam))


    eam = np.zeros(len(AVOID_POSITION))
    electrode1 = x1
    electrode2 = x2
    stimulation1 = np.zeros(NUM_ELE)
    stimulation1[electrode1] = 1 + 1 / x5
    stimulation1[electrode2] = -1 - 1 / x5
    e1 = np.array(
        [np.matmul(lfm[:, AVOID_POSITION, 0].T, stimulation1), np.matmul(lfm[:, AVOID_POSITION, 1].T, stimulation1),
         np.matmul(lfm[:, AVOID_POSITION, 2].T, stimulation1)]).T / 1000
    # print(np.max(e1))
    electrode3 = x3
    electrode4 = x4
    stimulation2 = np.zeros(NUM_ELE)
    # stimulation2[electrode3] = 1.5 - x[0]
    # stimulation2[electrode4] = -1.5 + x[0]
    stimulation2[electrode3] = 1 - 1 / x5
    stimulation2[electrode4] = -1 + 1 / x5
    e2 = np.array(
        [np.matmul(lfm[:, AVOID_POSITION, 0].T, stimulation2), np.matmul(lfm[:, AVOID_POSITION, 1].T, stimulation2),
         np.matmul(lfm[:, AVOID_POSITION, 2].T, stimulation2)]).T / 1000
    for i in range(len(e1)):
        l_x = np.sqrt(e1[i].dot(e1[i]))
        l_y = np.sqrt(e2[i].dot(e2[i]))
        point = e1[i].dot(e2[i])
        cos_ = point / (l_x * l_y)
        # print(cos_)
        if cos_ <= 0:  # one of the fields must be flipped when a > 90 degrees
            e1[i] = -e1[i]
            cos_ = -cos_
        if l_y < l_x:
            if l_y < l_x * cos_:
                eam[i] = 2 * l_y
            else:
                eam[i] = 2 * np.linalg.norm(np.cross(e2[i], (e1[i] - e2[i]))) / np.linalg.norm(e1[i] - e2[i])
        else:
            if l_x < l_y * cos_:
                eam[i] = 2 * l_x
            else:
                eam[i] = 2 * np.linalg.norm(np.cross(e1[i], (e2[i] - e1[i]))) / np.linalg.norm(e2[i] - e1[i])
    target2 = np.mean(abs(eam))
    return [target1, target2]

def fixed_ti_function5(x1, x2, x3, x4,x5):

    eam = np.zeros(len(pos))
    electrode1 = x1
    electrode2 = x2
    stimulation1 = np.zeros(NUM_ELE)
    stimulation1[electrode1] = 1 + 1 / x5
    stimulation1[electrode2] = -1 - 1 / x5
    e1 = np.array([np.matmul(lfm[:, :, 0].T, stimulation1), np.matmul(lfm[:, :, 1].T, stimulation1),
                   np.matmul(lfm[:, :, 2].T, stimulation1)]).T / 1000
    electrode3 = x3
    electrode4 = x4
    stimulation2 = np.zeros(NUM_ELE)
    stimulation2[electrode3] = 1 - 1 / x5
    stimulation2[electrode4] = -1 + 1 / x5
    e2 = np.array([np.matmul(lfm[:, :, 0].T, stimulation2), np.matmul(lfm[:, :, 1].T, stimulation2),
                   np.matmul(lfm[:, :, 2].T, stimulation2)]).T / 1000
    for i in range(len(e1)):
        l_x = np.sqrt(e1[i].dot(e1[i]))
        l_y = np.sqrt(e2[i].dot(e2[i]))
        point = e1[i].dot(e2[i])
        cos_ = point / (l_x * l_y)
        # print(cos_)
        if cos_ <= 0:  # one of the fields must be flipped when a > 90 degrees
            e1[i] = -e1[i]
            cos_ = -cos_
        if l_y < l_x:
            if l_y < l_x * cos_:
                eam[i] = 2 * l_y
            else:
                eam[i] = 2 * np.linalg.norm(np.cross(e2[i], (e1[i] - e2[i]))) / np.linalg.norm(e1[i] - e2[i])
        else:
            if l_x < l_y * cos_:
                eam[i] = 2 * l_x
            else:
                eam[i] = 2 * np.linalg.norm(np.cross(e1[i], (e2[i] - e1[i]))) / np.linalg.norm(e2[i] - e1[i])
    # print(np.max(eam))

    return np.array([1 / np.average(np.abs(eam[TARGET_POSITION])), np.mean(eam)])  # ,eam[TARGET_AVOID]


def tis_constraint6(x):
    lst = [math.ceil(x[i] * NUM_ELE) for i in [2, 3, 4, 5]]
    set_lst = set(lst)
    if len(set_lst) == len(lst) and x[0] + x[1] <= 1 and x[0] > 0.1 and x[1] > 0.1:
        return True
    else:
        return False

def tis_function6(x):
    #print(x)
    eam = np.zeros(len(pos))
    electrode1 = int(round(x[2] * 74))
    electrode2 = int(round(x[3] * 74))
    stimulation1 = np.zeros(NUM_ELE)
    stimulation1[electrode1] = 2 * x[0]
    stimulation1[electrode2] = -(2 * x[0])
    e1 = np.array([np.matmul(lfm[:, :, 0].T, stimulation1), np.matmul(lfm[:, :, 1].T, stimulation1),np.matmul(lfm[:, :, 2].T, stimulation1)]).T /1000
    #print(np.max(e1))
    electrode3 = int(round(x[4] * 74))
    electrode4 = int(round(x[5] * 74))
    print(electrode1,electrode2,electrode3,electrode4,x[0],x[1])
    stimulation2 = np.zeros(NUM_ELE)
    # stimulation2[electrode3] = 1.5 - x[0]
    # stimulation2[electrode4] = -1.5 + x[0]
    stimulation2[electrode3] = 2 * x[1]
    stimulation2[electrode4] = -(2 * x[1])
    e2 = np.array([np.matmul(lfm[:, :, 0].T, stimulation2), np.matmul(lfm[:, :, 1].T, stimulation2),np.matmul(lfm[:, :, 2].T, stimulation2)]).T / 1000
    # t1 = e1 + e2
    # t2 = e1 - e2
    #
    #
    # for i in range(len(t1)):
    #     eam[i] = np.sqrt(t1[i].dot(t1[i])) - np.sqrt(t2[i].dot(t2[i]))
    for i in range(len(e1)):
        l_x = np.sqrt(e1[i].dot(e1[i]))
        l_y = np.sqrt(e2[i].dot(e2[i]))
        point = e1[i].dot(e2[i])
        cos_ = point/(l_x*l_y)
        #print(cos_)
        if cos_ <= 0:   # one of the fields must be flipped when a > 90 degrees
            e1[i] = -e1[i]
            cos_ = -cos_
        if l_y < l_x:
            if l_y < l_x * cos_:
                eam[i] = 2 * l_y
            else:
                eam[i] = 2 * np.linalg.norm(np.cross(e2[i], (e1[i] - e2[i]))) / np.linalg.norm(e1[i] - e2[i])
        else:
            if l_x < l_y * cos_:
                eam[i] = 2 * l_x
            else:
                eam[i] = 2 * np.linalg.norm(np.cross(e1[i], (e2[i] - e1[i]))) / np.linalg.norm(e2[i] - e1[i])
    #print(np.max(eam))


    return np.array([1 / np.average(np.abs(eam[TARGET_POSITION])), np.mean(eam)])  # ,eam[TARGET_AVOID]
#return np.array([1 / np.average(np.abs(eam[TARGET_POSITION])), np.mean(eam)])

# calculate 1/E
def function1(x):
    x = np.array(x)
    x[abs(x) < 0.01] = 0
    # x.reshape(1, DNA_SIZE)
    return 1000/((np.matmul(lfm[:, TARGET_POSITION, 0].T, x)) ** 2 + (np.matmul(lfm[:, TARGET_POSITION, 1].T, x)) ** 2 + (np.matmul(lfm[:, TARGET_POSITION, 2].T, x)) ** 2)[0] ** 0.5

def function1_s(x):
    x = np.array(x)

    # x.reshape(1, DNA_SIZE)
    return np.abs(1000/(np.matmul(lfm[:, TARGET_POSITION, 0].T, x))[0])

# def tdcs_function2(s):

#     field = ((np.matmul(lfm[:, :, 0].T, s)) ** 2 + (np.matmul(lfm[:, :, 1].T, s)) ** 2 + (
#         np.matmul(lfm[:, :, 2].T, s)) ** 2) ** 0.5
#     return np.average(np.abs(field)) / 1000


def tdcs_function2(x):
    x = np.array(x)
    x[abs(x) < 0.01] = 0
    return np.average(((np.matmul(lfm[:, :, 0].T, x)) ** 2 + (
        np.matmul(lfm[:, :, 1].T, x)) ** 2 + (np.matmul(lfm[:, :, 2].T, x)) ** 2) ** 0.5) /1000

def tdcs_function1(x):
    x = np.array(x)
    x[abs(x) < 0.01] = 0
    return 1000 / np.average(((np.matmul(lfm[:, TARGET_POSITION, 0].T, x)) ** 2 + (
        np.matmul(lfm[:, TARGET_POSITION, 1].T, x)) ** 2 + (np.matmul(lfm[:, TARGET_POSITION, 2].T, x)) ** 2) ** 0.5)


def tdcs_function1_avoid(x):
    x = np.array(x)
    x[abs(x) < 0.01] = 0
    return 1000 / np.average(((np.matmul(lfm[:, AVOID_POSITION, 0].T, x)) ** 2 + (
        np.matmul(lfm[:, AVOID_POSITION, 1].T, x)) ** 2 + (np.matmul(lfm[:, AVOID_POSITION, 2].T, x)) ** 2) ** 0.5)



# calculate R
def function2(s):
    
    s = np.array(s)
    field = ((np.matmul(lfm[:, :, 0].T, s)) ** 2 + (np.matmul(lfm[:, :, 1].T, s)) ** 2 + (
        np.matmul(lfm[:, :, 2].T, s)) ** 2) ** 0.5
    field_r = 0
    field_all = np.sum(abs(field))

    r = 0
    idx = 0
    while field_r < 1 / 2 * field_all:
        r = distance[rank[idx]]
        field_r += abs(field[rank[idx]])
        idx = idx + 1
    # print("r", r)
    return r


def function2_s(s):
    # s = np.array(s)
    # s[abs(s) < 0.01] = 0
    field = np.matmul(lfm[:, :, 0].T, s)
    field_r = 0
    field_all = np.sum(abs(field))

    r = 0
    idx = 0
    while field_r < 1 / 2 * field_all:
        r = distance[rank[idx]]
        field_r += abs(field[rank[idx]])
        idx = idx + 1
    # print("r", r)
    return r

def tdcs_function3(s):
    field = np.matmul(lfm[:, :, 0].T, s)
    field_r = 0
    field_all = np.sum(abs(field))
    r = 0
    idx = 0
    while field_r <   0.5* field_all:
        r = distance[rank[idx]]
        field_r += abs(field[rank[idx]])
        idx = idx + 1
    # print("r", r)
    return r**0.5

def tdcs_function4(s):
    number = 0
    field = np.matmul(lfm[:, :, 0].T, s)
    max1 = field[np.argmin(distance)]
    #max1 = np.max(field)
    for i in field:
        if i > 0.5 * max1:
            number = number + 1
    return number/len(field)

def function0(s):
    return np.sum(np.abs(s) > 0.01)


def function2roastt(s):
    field = ((np.matmul(lfm[:, 0, :], s)) ** 2 + (np.matmul(lfm[:, 1, :], s)) ** 2 + (
        np.matmul(lfm[:, 2, :], s)) ** 2) ** 0.5
    r = 0
    field_target = function1(s)
    # print("target:", field_target)
    for f in field:
        if f > 0.5 * field_target:
            r = r + 1
    # print("r", r**(1/3))
    return r



# satisfy constraints' domains others
def function3(x):
    x = np.array(x)
    x[abs(x) < 0.01] = 0

    result = abs(np.sum(x))
    if result <= 1:
        return 0
    else:
        return result - 1


def function4(x):
    x = np.array(x)
    x[abs(x) < 0.01] = 0
    result = np.sum(abs(x)) + abs(np.sum(x))
    if result <= 4:
        return 0
    else:
        return result - 4


def h(x):
    print("constraint violation:", function3(x) + 1/4 * function4(x))
    return function3(x) + 1 / 4 * function4(x)


def h_function(subpopulation):
    h_values = [h(subpopulation[i]) for i in range(len(subpopulation))]
    return subpopulation[np.argmin(h_values)]


def asf_function(subpopulation, w):
    function1_values = [function1(subpopulation[i]) for i in range(len(subpopulation))]
    function2_values = [function2(subpopulation[i]) for i in range(len(subpopulation))]
    function1_values_min = min(function1_values)
    function2_values_min = min(function2_values)
    flag = 0
    value = inf
    for i in range(len(subpopulation)):
        temp1 = (function1_values[i] - function1_values_min) / w[0]
        temp2 = (function2_values[i] - function2_values_min) / w[1]
        if value > max(temp1, temp2):
            flag = i
            value = max(temp1, temp2)
    return subpopulation[flag]


def divide(subpopulation):
    U = []
    V = []
    V1 = []
    for solution,idx in enumerate(subpopulation):
        if h(solution) > 0 :
            U.append(idx)
        else : V.append(idx)
    for idxv in V:
        for idxu in U:
            if function1(subpopulation[idxv]) < function1(subpopulation[idxu]) or function2(subpopulation[idxv]) < function2(subpopulation[idxu]):
                V1.append(idxv)
                break
    return U,V1

def near_function(subpopulation):
    U,V1 = divide(subpopulation)
    flag = 0
    value = inf
    for i in V1:
        for j in U:
            near_value = [function1(subpopulation[i]) - function1(subpopulation[j])]**2 + [function2(subpopulation[i]) - function2(subpopulation[j])]**2
            if value > near_value:
                value = near_value
                flag = i
    return subpopulation[flag]


def inverse_asf_function(subpopulation, w):
    function1_values = [function1(subpopulation[i]) for i in range(len(subpopulation))]
    function2_values = [function2(subpopulation[i]) for i in range(len(subpopulation))]
    function1_values_min = max(function1_values)
    function2_values_min = max(function2_values)
    flag = 0
    value = inf
    for i in range(len(subpopulation)):
        temp1 = (function1_values[i] - function1_values_min) / w[0]
        temp2 = (function2_values[i] - function2_values_min) / w[1]
        if value > min(temp1, temp2):
            flag = i
            value = min(temp1, temp2)
    return subpopulation[flag]


def group(population,reference):
    direction_w = [function1(i)/function2(i) for i in reference]
    direction_p = [function1(i)/function2(i) for i in population]
    group_value = [np.argmin([abs(direction_p[i] - j) for j in direction_w]) for i in range(direction_p)]
    return group_value


def fast_non_dominated_sort(values1, values2, values3):
    S = [[] for i in range(0, len(values1))]  # S[i] i支配的集合
    front = [[]]  # front[i],ranki的集合
    n = [0 for i in range(0, len(values1))]  # n[i]支配i的个数
    rank = [0 for i in range(0, len(values1))]  # rank[i] i的rank
    #  计算了每个值的n，将n=0的值
    for p in range(0, len(values1)):
        S[p] = []
        n[p] = 0
        for q in range(0, len(values1)):
            # if (values1[p] > values1[q] and values2[p] > values2[q] and values3[p] >= values3[q]) or (values1[p] >= values1[q] and values2[p] > values2[q] and values3[p] >= values3[q]) or (values1[p] > values1[q] and values2[p] >= values2[q] and values3[p] >= values3[q]):
            if (values1[q] >= values1[p] and values2[q] > values2[p] and values3[p] == values3[q] == 0) or (
                    values1[q] > values1[p] and values2[q] >= values2[p] and values3[p] == values3[q] == 0) or (
                     values3[q] > values3[p]):
                if q not in S[p]:
                    S[p].append(q)  # [p]的value支配[q]的value
            elif (values1[p] >= values1[q] and values2[p] > values2[q] and values3[p] == values3[q] == 0) or (
                    values1[p] > values1[q] and values2[p] >= values2[q] and values3[p] == values3[q] == 0) or (
                    values3[p] > values3[q]):
                n[p] = n[p] + 1  # [q]的value支配[p]的value
        if n[p] == 0:  # [p]的value支配所有[q]的value
            rank[p] = 0  # p‘s rank = 0
            if p not in front[0]:
                front[0].append(p)  # p join in rank0

    i = 0
    while (front[i] != []):  # new rank != 0  which means there exists number
        Q = []
        for p in front[i]:  # 当前rank的每个值
            for q in S[p]:  # 对值的S集合里的每个q
                n[q] = n[q] - 1  # q的n-1, 因为当前的集合已经被加入了RANK
                if (n[q] == 0):  # 如果n=0 means q没有支配他的解了
                    rank[q] = i + 1  # q 的rank 设为i+1
                    if q not in Q:
                        Q.append(q)
        i = i + 1
        front.append(Q)  # i+1 的rank的集合设置为q的集合

    del front[len(front) - 1]  # 删去最后一个空front
    return front  # front就是2维度的第1维度是rank,第二维度是对应rank的解(索引)的集合


def index_of(a,list):
    for i in range(0,len(list)):
        if list[i] == a:
            return i
    return -1


def sort_by_values(list1, values):
    sorted_list = []
    while(len(sorted_list)!=len(list1)):
        if index_of(min(values),values) in list1:  # index_of(min(values),values)获得values中最小值的索引,如果这个最小值在当前的rank里
            sorted_list.append(index_of(min(values),values))  # sorted_list获得最小值
        values[index_of(min(values),values)] = np.inf  # 将value的最小值替换为math.inf
    return sorted_list


# Function to calculate crowding distance  front这里指某一个rank
def crowding_distance(values1, values2, front):
    distance = [0 for i in range(0, len(front))]
    sorted1 = sort_by_values(front, values1[:])
    sorted2 = sort_by_values(front, values2[:])
    distance[0] = 4444444444444444
    distance[len(front) - 1] = 4444444444444444  # 最边上的点设置拥挤度
    for k in range(1, len(front) - 1):
        distance[k] = distance[k] + (values1[sorted1[k + 1]] - values2[sorted1[k - 1]]) / (
                    max(values1) - min(values1))
    for k in range(1, len(front) - 1):
        distance[k] = distance[k] + (values1[sorted2[k + 1]] - values2[sorted2[k - 1]]) / (
                    max(values2) - min(values2))
    return distance


if __name__ == "__main__":
    import scipy.io as io
    # Main program starts here
    pop_size = 140
    max_gen = 100
    DNA_SIZE = 125
    # Initialization
    min_x = -1
    max_x = 1
    solution = np.random.uniform(min_x, max_x, size=(pop_size, DNA_SIZE)).tolist()
    # solution = []
    # for i in range(13,27,1):
    #     a = io.loadmat(r"/home/ncclab306/zfs-pool/wangmo/tes/matlab0."+str(i)+".mat")
    #     a = a['ans']
    #     a = a.reshape(1,-1)  # 1,125
    #     solution.append(a[0])
    # solution = np.repeat(np.array(solution),10,axis=0).tolist()
    print(function0(solution[0]))
    print(1/function1(solution[0]))
    print(function2(solution[0]))
    print(function3(solution[0]))
    print(function4(solution[0]))
    print(h(solution[0]))


