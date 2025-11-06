#encoding: utf-8
import numpy as np
import os
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
#from mpl_toolkits.mplot3d import Axes3D
import time
class Plot_pareto:
    def __init__(self):
        self.start_time = time.time()

    def show(self,in_,fitness_,archive_in,archive_fitness,i,m=2):

        #共3个子图，第1、2/子图绘制输入坐标与适应值关系，第3图展示pareto边界的形成过程

        # fig = plt.figure('第'+str(i+1)+'次迭代',figsize = (17,5))
        # ax1 = fig.add_subplot(131, projection='3d')
        # ax1.set_xlabel('input_x1')
        # ax1.set_ylabel('input_x2')
        # ax1.set_zlabel('fitness_y1')
        # ax1.plot_surface(self.x1,self.x2,self.y1,alpha = 0.6)
        # ax1.scatter(in_[:,0],in_[:,1],fitness_[:,0],s=20, c='blue', marker=".")
        # ax1.scatter(archive_in[:,0],archive_in[:,1],archive_fitness[:,0],s=50, c='red', marker=".")
        # ax2 = fig.add_subplot(132, projection='3d')
        # ax2.set_xlabel('input_x1')
        # ax2.set_ylabel('input_x2')
        # ax2.set_zlabel('fitness_y2')
        # ax2.plot_surface(self.x1,self.x2,self.y2,alpha = 0.6)
        # ax2.scatter(in_[:,0],in_[:,1],fitness_[:,1],s=20, c='blue', marker=".")
        # ax2.scatter(archive_in[:,0],archive_in[:,1],archive_fitness[:,1],s=50, c='red', marker=".")
        fig = plt.figure('第' + str(i + 1) + '次迭代')
        # data = np.array([[0.13, 0.137, 38.5, 42.8, 40.6],
        #                  [0.14, 0.147, 39.2, 43.9, 41.5],
        #                  [0.15, 0.157, 40.0, 45.2, 42.7],
        #                  [0.16, 0.167, 40.8, 47.2, 44.4],
        #                  [0.17, 0.178, 42.0, 49.3, 46.4],
        #                  [0.18, 0.187, 43.0, 51.0, 48.2],
        #                  [0.19, 0.197, 44.2, 52.6, 49.0],
        #                  [0.20, 0.206, 45.7, 54.6, 50.7],
        #                  [0.21, 0.216, 47.9, 55.5, 51.6],
        #                  [0.22, 0.227, 51.1, 56.2, 51.7],
        #                  [0.23, 0.235, 55.8, 58.8, 55.6],
        #                  [0.24, 0.250, 60.5, 59.8, 56.7],
        #                  [0.25, 0.264, 65.9, 65.3, 60.6],
        #                  [0.26, 0.283, 68.4, 68.3, 60.2]])
        #
        # plt.scatter(data[:, 0], data[:, 2], c='r', label='single_direction')
        # plt.scatter(data[:, 1], data[:, 3], c='b', label='all_direction')
        # plt.scatter(data[:, 0], data[:, 4], c='y', label='roast_method')
        #plt.scatter([1 / function1_s(i) for i in gt_data], [function2_s(i) for i in gt_data], s=10, c='black',
        #            marker=".")
        plt.legend(loc='best', fontsize=16, markerscale=0.5)
        plt.title("trade-off_for_electric_and_focality", fontdict={'size': 20})
        plt.xlabel("Electric_Field(V/m)", fontdict={'size': 16})
        plt.ylabel("Half-Max_Radius(mm)", fontdict={'size': 16})

        # ax3.set_xlim((0,1))
        # ax3.set_ylim((0,1))
        # for i in range(len(fitness_)):
        #     if fitness_[i,0] <100 or fitness_[i,1] <100:
        #         ax3.scatter(fitness_[i,0],fitness_[i,1],s=10, c='blue', marker=".")
        # for i in range(len(archive_fitness)):
        #     if archive_fitness[i,0] <100 or archive_fitness[i,1] <100:
        #         ax3.scatter(archive_fitness[i,0],archive_fitness[i,1],s=30, c='red', marker=".",alpha = 1.0)
        #plt.scatter(fitness_[:, 0], fitness_[:, 1], s=10, c='g', marker=".")
        plt.scatter(1/archive_fitness[:, 0], archive_fitness[:, 1], s=30, c='black', marker=".", alpha=1.0, label='MOPSO')

        plt.savefig('./pic3/86_'+str(i)+'_result.png')
        #plt.scatter(1/fitness_[:, 0], fitness_[:, 1], s=10, c='g', marker=".")
        #plt.savefig('./pic/16all_' + str(i) + '_result.png')
        # plt.savefig('./img_txt/'+str(i+1)+'.png')
        # print ('第'+str(i+1)+'次迭代的图片保存于 img_txt 文件夹')
        # print ('第'+str(i+1)+'次迭代, time consuming: ',np.round(time.time() - self.start_time, 2), "s")
        #plt.ion()
        # plt.close()

'''

        plt.figure('第' + str(i + 1) + '次迭代', figsize=(10, 10), dpi=100)
        ax = plt.axes(projection='3d')  # 设置三维轴
        ax.scatter3D(fitness_[:, 1], fitness_[:, 0], fitness_[:, 2], s=10, c='blue', marker=".")  # 三个数组对应三个维度（三个数组中的数一一对应）
        ax.scatter3D(archive_fitness[:, 1], archive_fitness[:, 0], archive_fitness[:, 2], s=30, c='red', marker=".", alpha=1.0)
        # plt.xticks(range(11))  # 设置 x 轴坐标
        plt.rcParams.update({'font.size': 20})
        plt.xlabel('intensity:1/E')
        plt.ylabel('focality:R', rotation=38)  # y 轴名称旋转 38 度
        ax.set_zlabel('constraint violation')  # 因为 plt 不能设置 z 轴坐标轴名称，所以这里只能用 ax 轴来设置（当然，x 轴和 y 轴的坐标轴名称也可以用 ax 设置）
        plt.show()
        plt.ion()
'''


