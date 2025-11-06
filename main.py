#encoding: utf-8
import numpy as np
from Mopso import *
from public import  P_objective

 
def main():

    particals = 100 #粒子群的数量
    cycle_ = 1000 #迭代次数
    mesh_div = 10 #网格等分数量
    thresh = 100 #外部存档阀值

    Problem = "TES"
    M = 2
    # Population, Boundary, Coding = P_objective.P_objective("init", Problem, M, particals)
    print("init")
    _, Boundary, _ = P_objective.P_objective("init", Problem, M, particals)
    max_ = Boundary[0]
    min_ = Boundary[1]

    print("start")
    mopso_ = Mopso(particals,max_,min_,thresh,mesh_div) #粒子群实例化
    pareto_in,pareto_fitness = mopso_.done(cycle_) #经过cycle_轮迭代后，pareto边界粒子
    np.savetxt("./img_txt/pareto_in_n.txt",pareto_in)#保存pareto边界粒子的坐标
    np.savetxt("./img_txt/pareto_fitness_n.txt",pareto_fitness) #打印pareto边界粒子的适应值
    print ("\n","pareto边界的坐标保存于：/img_txt/pareto_inn.txt")
    print ("pareto边界的适应值保存于：/img_txt/pareto_fitnessn.txt")
    print ("\n,迭代结束,over")

 
if __name__ == "__main__":
    main()
