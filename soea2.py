import numpy as np
import geatpy as ea  # import geatpy
import sys
import argparse
from public import glo




def argdet():
 if len(sys.argv) <= 9:
     args = myargs()
     return args
 else:
     print('Cannot recognize the inputs!')
     print("-i data -opt optimizer -dim dimension")
     exit()

def myargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', '-t', default="tdcs", help='stimulation method')
    parser.add_argument('--position', '-p', default='motor', help='target location')#跟glo对应？ #motor
    parser.add_argument('--head', '-m', default='0026058_10', help='head model name')#这里要改输入
    parser.add_argument('--gen', '-g', default= 0 , help='max epochs')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    print("start")

    args = argdet()
    glo.head_model = args.head
    glo.type = args.type
    glo.name = args.position
    #  实例化问题对象
    # 定义outFunc()函数
    if args.position == 'hippo':
        glo.position = np.array([-31, -20, -14])
    elif args.position == 'pallidum':
        glo.position = np.array([-17, 3, -1])
    elif args.position == 'thalamus':
        glo.position = np.array([10, -19, 6])
    # if args.position == 'sensory':
    #     glo.position = [41,-36,66]
    # if args.position == 'dorsal':
    #     glo.position = [25,42,37]
    elif args.position == 'v1':
        glo.position = np.array( [14,-99,-3])#     [10,-92,2])#   #[11, -84, 5])
    elif args.position == 'dlpfc':
        glo.position = np.array([-39, 34, 37])     #[-39, 34, 37]    #[35,39,31]
    elif args.position == 'motor':
        #glo.position = np.array([47, -13, 52])
        glo.position = np.array([-36,-19,48])
    else:
        print("!!!!!!!!!error roi!!!!!!!!!!!!")

    from soea2_problem import MyProblem  # 导入自定义问题接口

    if args.type == 'ti':
        problem = MyProblem_ti.MyProblem()
        #problem = myproblem_ti_gpu.MyProblem()
        gen = 100
    else:
        problem = MyProblem()
        gen = 600
    if int(args.gen) != 0:
        gen = int(args.gen)

    algorithm = ea.soea_DE_rand_1_bin_templet(
        problem,
        ea.Population(Encoding='RI', NIND=100),
        MAXGEN=200,  # 最大进化代数。
        logTras=1,  # 表示每隔多少代记录一次日志信息，0表示不记录。         
        #trappedValue=1e-1,  # 单目标优化陷入停滞的判断阈值。
        maxTrappedCount=10)  # 进化停滞计数器最大上限值。
    algorithm.mutOper.F = 0.5  # 差分进化中的参数F。
    algorithm.recOper.XOVR = 0.2  # 差分进化中的参数Cr。
    # 求解
    res = ea.optimize(algorithm,
                      verbose=True,
                      drawing=10,
                      outputMsg=False,
                      drawLog=False,
                      saveFlag=False)
    print(res)

    # print(res['optPop'])
    # print(res['Vars'])
    # print(res['Vars'][0,0])
    prior = np.array(res['Vars'][0])
    glo.prior = prior

    from Mopso import *
    from public import P_objective

    particals = 100  # 粒子群的数量
    cycle_ = 500 # 迭代次数
    mesh_div = 10 # 网格等分数量
    thresh = 200  # 外部存档阀值

    Problem = "TES"
    M = 2
    # Population, Boundary, Coding = P_objective.P_objective("init", Problem, M, particals)
    print("init")
    _, Boundary, _ = P_objective.P_objective("init", Problem, M, particals)
    max_ = Boundary[0]
    min_ = Boundary[1]

    print("start")
    mopso_ = Mopso(particals, max_, min_, thresh, mesh_div)  # 粒子群实例化
    pareto_in, pareto_fitness = mopso_.done(cycle_)  # 经过cycle_轮迭代后，pareto边界粒子
    np.savetxt("./MOVEA_conduc/pareto1_in_"+ args.position+"_"+ args.head +".txt", pareto_in)  # 保存pareto边界粒子的坐标
    np.savetxt("./MOVEA_conduc/pareto1_fitness_"+ args.position+"_"+ args.head +".txt", pareto_fitness)  # 打印pareto边界粒子的适应值
    print("\n", "pareto_position：/500cycle/1105pareto1111_in_"+ args.position+"_"+ args.head +".txt")
    print("pareto)value：/500cycle/1105pareto1111_fitness_"+ args.position+"_"+ args.head +".txt")
    print("\n,over")
