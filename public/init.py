#encoding: utf-8
import random
import numpy as np
from public import pareto,NDsort,glo
import scipy.io as io

def init_designparams(particals,in_min,in_max):
    in_dim = len(in_max)     #输入参数维度
    print(in_dim)
    print(glo.prior)
    solution = np.zeros(75)
    solution[glo.prior[0]] = 1
    solution[glo.prior[1]] = 1
    solution[glo.prior[2]] = -1
    solution[glo.prior[3]] = -1

    print(solution)
    solution=np.repeat([solution], particals-5, axis=0)
    # hippo
    # solution[0, 17] = 1
    # solution[0, 28] = 1
    # solution[0, 30] = -1
    # solution[0, 31] = -1
    # solution[0, 47] = 1
    # solution[0, 58] = 1
    # solution[0, 14] = -1
    # solution[0, 25] = -1
    in_temp = np.random.uniform(-5, 5, (5, in_dim))
    in_temp[-1 > in_temp] = 0
    in_temp[in_temp > 1] = 0
    in_temp = np.vstack([in_temp,solution])
    print(in_temp)
    return in_temp


def init_v(particals,v_max,v_min):
    v_dim = len(v_max)     #输入参数维度
    v_ = np.random.uniform(0,1,(particals,v_dim))*(v_max-v_min)+v_min

    # v_ = np.zeros((particals,v_dim))
    return v_

def init_pbest(in_,fitness_):
    return in_,fitness_

def init_archive(in_,fitness_):

    # FrontValue_1_index = NDsort.NDSort(fitness_, in_.shape[0])[0]==1
    # FrontValue_1_index = np.reshape(FrontValue_1_index,(-1,))
    #
    # curr_archiving_in=in_[FrontValue_1_index]
    # curr_archiving_fit=fitness_[FrontValue_1_index]

    pareto_c = pareto.Pareto_(in_,fitness_)
    curr_archiving_in,curr_archiving_fit = pareto_c.pareto()
    return curr_archiving_in,curr_archiving_fit


