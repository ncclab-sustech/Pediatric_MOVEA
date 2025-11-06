"""Demo.
max f = (-1+x1+((6-x2)*x2-2)*x2)**2+(-1+x1+((x2+2)*x2-10)*x2)**2
s.t.  x∈{1.1, 1, 0, 3, 5.5, 7.2, 9}
"""
import numpy as np
from public import glo
import geatpy as ea
from public import util


class MyProblem(ea.Problem):  # 继承Problem父类

    def __init__(self):
        name = 'MyProblem'  # 初始化name（函数名称，可以随意设置）
        M = 1  # 初始化M（目标维数）
        maxormins = [1]  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        self.var_set = np.arange(0,75,1) # 设定一个集合，要求决策变量的值取自于该集合
        Dim = 4  # 初始化Dim（决策变量维数）
        varTypes = [1] * Dim  # 初始化varTypes（决策变量的类型，元素为0表示对应的变量是连续的；1表示是离散的）
        lb = [0, 0,0,0]  # 决策变量下界
        ub = [74, 74,74,74]  # 决策变量上界
        lbin = [1] * Dim  # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [1] * Dim  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self,
                            name,
                            M,
                            maxormins,
                            Dim,
                            varTypes,
                            lb,
                            ub,
                            lbin,
                            ubin)

    def evalVars(self, Vars):  # 目标函数
        Vars = Vars.astype(np.int32)  # 强制类型转换确保元素是整数
        x1 = self.var_set[Vars[:, [0]]]  # 得到所有的x1组成的列向量
        x2 = self.var_set[Vars[:, [1]]]  # 得到所有的x2组成的列向量
        x3 = self.var_set[Vars[:, [2]]]  # 得到所有的x1组成的列向量
        x4 = self.var_set[Vars[:, [3]]]  # 得到所有的x2组成的列向量
        r = np.zeros(len(x1)).tolist()
        for i in range(len(x1)):
            lst = [int(x1[i]),int(x2[i]),int(x3[i]),int(x4[i])]
            set_lst = set(lst)
            if len(set_lst) == len(lst):
                x = np.zeros(75)
                x[x1[i]] = 1
                x[x2[i]] = 1
                x[x3[i]] = -1
                x[x4[i]] = -1
                r[i] = [util.tdcs_function1(x)]
                #r[i] = [tis_function6(x1[i], x2[i], x3[i], x4[i])]
            else:
                r[i] = [10000]
        #print(r)
        return np.array(r)
