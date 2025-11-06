import matplotlib.pyplot as plt
import numpy as np

def cal_rank(x,gap):
    index = [0,len(x)-1]
    temp = x[0]
    for i in range(len(x)):
        if x[i] - temp > gap:
            index.append(i)
            temp = x[i]
    index = np.unique(index)
    return index

def rank_dense(x):
    
    #print(x)
    length = np.zeros(len(x))
    for i in range(len(x)):
        if i == 0 or i == len(x)-1:
            length[i] = 999
        else: 
            length[i] = x[i+1] - x[i-1]
    return -length
        
GAP =0.00
MODEL ='hcp4'
# for tacs img_txt is node == 1000,for ti img_txt/2
#plt.figure(figsize=(20, 10), dpi=100)
t,sub = plt.subplots(2,1,figsize=(20, 10), dpi=100)

plt.subplot(121)
gt_data = []
f2 = open(r"img_txt2/pareto_fitness_motor_"+MODEL+".txt", "r")
lines = f2.readlines()
for line3 in lines:
    #print(line3)
    cur = line3.strip().split(" ")
    cur = list(map(float, cur))
    gt_data.append(cur)
data = np.array(gt_data)
index1 = np.argsort(1/data[:,0])

index = cal_rank(1 / data[index1, 0],GAP)
parameter = np.polyfit(1 / data[index, 0],data[index, 1],2)
#plt.scatter(1/data[index, 0], data[index, 1],  c='#F73030', marker='1', label='Motor : '+ str(np.around(parameter[0],decimals=2))+'*X^2'+str(np.around(parameter[1],decimals=2))+'*X+'+str(np.around(parameter[2],decimals=2)))
plt.scatter(1/data[index1[index], 0], data[index1[index], 1], s=30, c='#F73030', marker='v', label='Motor : '+ str(np.around(parameter[0],decimals=2))+'*X^2+'+str(np.around(parameter[1],decimals=2))+'*X+'+str(np.around(parameter[2],decimals=2)))

p = np.poly1d(parameter)
plt.plot(1/data[index1,0],p(1/data[index1,0]),c='#F73030')


gt_data = []
f2 = open(r"img_txt2/pareto_fitness_dlpfc_"+MODEL+".txt", "r")
lines = f2.readlines()
for line3 in lines:
    #print(line3)
    cur = line3.strip().split(" ")
    cur = list(map(float, cur))
    gt_data.append(cur)
data = np.array(gt_data)
index1 = np.argsort(1/data[:,0])
index = cal_rank(1 / data[index1, 0],GAP)
parameter = np.polyfit(1 / data[index, 0],data[index, 1],2)
plt.scatter(1/data[index1[index], 0], data[index1[index], 1], s=30,  c='#548235', marker='v', label='DLPFC : '+ str(np.around(parameter[0],decimals=2))+'*X^2+'+str(np.around(parameter[1],decimals=2))+'*X+'+str(np.around(parameter[2],decimals=2)))
p = np.poly1d(parameter)
plt.plot(1/data[index1,0],p(1/data[index1,0]),c='#548235')


gt_data = []
f2 = open(r"img_txt2/pareto_fitness_v1_"+MODEL+".txt", "r")
lines = f2.readlines()
for line3 in lines:
    #print(line3)
    cur = line3.strip().split(" ")
    cur = list(map(float, cur))
    gt_data.append(cur)
data = np.array(gt_data)
index1 = np.argsort(1/data[:,0])
index = cal_rank(1 / data[index1, 0],GAP)
parameter = np.polyfit(1 / data[index, 0],data[index, 1],2)
plt.scatter(1/data[index1[index], 0], data[index1[index], 1], s=30,  c='#C55A11', marker='v', label='V1 : '+ str(np.around(parameter[0],decimals=2))+'*X^2+'+str(np.around(parameter[1],decimals=2))+'*X+'+str(np.around(parameter[2],decimals=2)))
p = np.poly1d(parameter)
plt.plot(1/data[index1,0],p(1/data[index1,0]),c='#C55A11')
#
gt_data = []
f2 = open(r"img_txt2/pareto_fitness_hippo_"+MODEL+".txt", "r")
lines = f2.readlines()
for line3 in lines:
    #print(line3)
    cur = line3.strip().split(" ")
    cur = list(map(float, cur))
    gt_data.append(cur)
data = np.array(gt_data)
index1= np.argsort(1/data[:,0])
index = cal_rank(1 / data[index1, 0],GAP)
parameter = np.polyfit(1 / data[index, 0],data[index, 1],2)
plt.scatter(1/data[index1[index], 0], data[index1[index], 1], s=30,  c='#ef8b8b', marker='v', label='Hippocampus : '+ str(np.around(parameter[0],decimals=2))+'*X^2+'+str(np.around(parameter[1],decimals=2))+'*X+'+str(np.around(parameter[2],decimals=2)))
p = np.poly1d(parameter)
plt.plot(1/data[index1,0],p(1/data[index1,0]),c='#ef8b8b')
#
#
gt_data = []
f2 = open(r"img_txt2/pareto_fitness_pallidum_"+MODEL+".txt", "r")
lines = f2.readlines()
for line3 in lines:
    #print(line3)
    cur = line3.strip().split(" ")
    cur = list(map(float, cur))
    gt_data.append(cur)
data = np.array(gt_data)
index1 = np.argsort(1/data[:,0])
index = cal_rank(1 / data[index1, 0],GAP)
parameter = np.polyfit(1 / data[index, 0],data[index, 1],2)
plt.scatter(1/data[index1[index], 0], data[index1[index], 1],  c='#00b0f0', s=30, marker='v', label='Pallidum : '+ str(np.around(parameter[0],decimals=2))+'*X^2+'+str(np.around(parameter[1],decimals=2))+'*X+'+str(np.around(parameter[2],decimals=2)))
p = np.poly1d(parameter)
plt.plot(1/data[index1,0],p(1/data[index1,0]),c='#00b0f0')

#
#
gt_data = []
f2 = open(r"img_txt2/pareto_fitness_thalamus_"+MODEL+".txt", "r")
lines = f2.readlines()
for line3 in lines:
    #print(line3)
    cur = line3.strip().split(" ")
    cur = list(map(float, cur))
    gt_data.append(cur)
data = np.array(gt_data)

index1 = np.argsort(1/data[:,0])
index = cal_rank(1 / data[index1, 0],GAP)
parameter = np.polyfit(1 / data[index, 0],data[index, 1],2)
plt.scatter(1/data[index1[index], 0], data[index1[index], 1], s=30, c='gold', marker='v', label='Thalamus : '+ str(np.around(parameter[0],decimals=2))+'*X^2+'+str(np.around(parameter[1],decimals=2))+'*X+'+str(np.around(parameter[2],decimals=2)))
p = np.poly1d(parameter)
plt.plot(1/data[index1,0],p(1/data[index1,0]),c='gold')

#
plt.legend(loc='upper left', fontsize=12, markerscale=2)
plt.title('tACS Optimization Results For '+MODEL,fontsize=20)
plt.rcParams.update({'font.size': 15})
plt.xlabel('Target Intensity (V/m)',fontsize=20)
plt.ylabel('Mean Field Norm (V/m)',fontsize=20)  # y 轴名称旋转 38 度
plt.grid(True,linestyle = '--') # 不显示网格线

ax = plt.gca()

# ax.spines['right'].set_color('#ccc')
# ax.spines['top'].set_color('#ccc')
# ax.spines['left'].set_color('#ccc')
# ax.spines['bottom'].set_color('#ccc')
# 设置图像的边框粗细程度
ax.spines['right'].set_linewidth('2.0')
ax.spines['top'].set_linewidth('2.0')
ax.spines['left'].set_linewidth('2.0')
ax.spines['bottom'].set_linewidth('2.0')

plt.xlim(0.09, 0.61, 0.08)
plt.xticks(np.arange(0.1, 0.7, 0.1),size =10)
plt.ylim(0, 0.25, 0.05)
plt.yticks(np.arange(0.05, 0.25, 0.05),size =10)

# plt.xlim(0.1, 0.5, 0.03)
# plt.xticks(np.arange(0.1, 0.5, 0.08),size =10)
# plt.ylim(0, 0.25, 0.05)
# plt.yticks(np.arange(0.05, 0.25, 0.05),size =10)



#-----------------------------------------------#
plt.subplot(122)
gt_data = []
f2 = open(r"/home/ncclab306/zfs-pool/tes/ti/mopso/img_txt_ti/pareto_fitness_motor_"+MODEL+".txt", "r")
lines = f2.readlines()
for line3 in lines:
    #print(line3)
    cur = line3.strip().split(" ")
    cur = list(map(float, cur))
    gt_data.append(cur)
# f2 = open(r"/home/ncclab306/zfs-pool/tes/ti/mopso/img_txt3/pareto_fitness_motor_"+MODEL+".txt", "r")
# lines = f2.readlines()
# for line3 in lines:
#     #print(line3)
#     cur = line3.strip().split(" ")
#     cur = list(map(float, cur))
#     gt_data.append(cur)
data = np.array(gt_data)
index = np.argsort(1/data[:,0])
data = data[index]
index = cal_rank(1 / data[:,0],GAP/2)
parameter = np.polyfit(1 / data[index, 0],data[index, 1],2)
#plt.scatter(1/data[index, 0], data[index, 1],  c='#F73030', marker='1', label='Motor : '+ str(np.around(parameter[0],decimals=2))+'*X^2'+str(np.around(parameter[1],decimals=2))+'*X+'+str(np.around(parameter[2],decimals=2)))
plt.scatter(1/data[index, 0], data[index, 1], s=30,  c='#F73030', marker='o', label='Motor : '+ str(np.around(parameter[0],decimals=2))+'*X^2+'+str(np.around(parameter[1],decimals=2))+'*X+'+str(np.around(parameter[2],decimals=2)))
p = np.poly1d(parameter)
plt.plot(1/data[index,0],p(1/data[index,0]),c='#F73030')


gt_data = []
f2 = open(r"/home/ncclab306/zfs-pool/tes/ti/mopso/img_txt_ti/pareto_fitness_dlpfc_"+MODEL+".txt", "r")
lines = f2.readlines()
for line3 in lines:
    #print(line3)
    cur = line3.strip().split(" ")
    cur = list(map(float, cur))
    gt_data.append(cur)
f2 = open(r"/home/ncclab306/zfs-pool/tes/ti/mopso/img_txt3/pareto_fitness_dlpfc_"+MODEL+".txt", "r")
lines = f2.readlines()
for line3 in lines:
    #print(line3)
    cur = line3.strip().split(" ")
    cur = list(map(float, cur))
    gt_data.append(cur)
data = np.array(gt_data)
index = np.argsort(1/data[:,0])
data = data[index]
index = cal_rank(1 / data[:,0],GAP/2)
parameter = np.polyfit(1 / data[index, 0],data[index, 1],2)
plt.scatter(1/data[index, 0], data[index, 1], s=30,  c='#548235', marker='o', label='DLPFC : '+ str(np.around(parameter[0],decimals=2))+'*X^2+'+str(np.around(parameter[1],decimals=2))+'*X+'+str(np.around(parameter[2],decimals=2)))
p = np.poly1d(parameter)
plt.plot(1/data[index,0],p(1/data[index,0]),c='#548235')


gt_data = []
f2 = open(r"/home/ncclab306/zfs-pool/tes/ti/mopso/img_txt_ti/pareto_fitness_v1_"+MODEL+".txt", "r")
lines = f2.readlines()
for line3 in lines:
    #print(line3)
    cur = line3.strip().split(" ")
    cur = list(map(float, cur))
    gt_data.append(cur)
f2 = open(r"/home/ncclab306/zfs-pool/tes/ti/mopso/img_txt3/pareto_fitness_v1_"+MODEL+".txt", "r")
lines = f2.readlines()
for line3 in lines:
    #print(line3)
    cur = line3.strip().split(" ")
    cur = list(map(float, cur))
    gt_data.append(cur)
data = np.array(gt_data)
index = np.argsort(1/data[:,0])
data = data[index]
index = cal_rank(1 / data[:,0],GAP/2)
parameter = np.polyfit(1 / data[index, 0],data[index, 1],2)
plt.scatter(1/data[index, 0], data[index, 1], s=30,  c='#C55A11', marker='o', label='V1 : '+ str(np.around(parameter[0],decimals=2))+'*X^2+'+str(np.around(parameter[1],decimals=2))+'*X+'+str(np.around(parameter[2],decimals=2)))
p = np.poly1d(parameter)
plt.plot(1/data[index,0],p(1/data[index,0]),c='#C55A11')


#
gt_data = []
f2 = open(r"/home/ncclab306/zfs-pool/tes/ti/mopso/img_txt_ti/pareto_fitness_hippo_"+MODEL+".txt", "r")
lines = f2.readlines()
for line3 in lines:
    #print(line3)
    cur = line3.strip().split(" ")
    cur = list(map(float, cur))
    gt_data.append(cur)
f2 = open(r"/home/ncclab306/zfs-pool/tes/ti/mopso/img_txt3/pareto_fitness_hippo_"+MODEL+".txt", "r")
lines = f2.readlines()
for line3 in lines:
    #print(line3)
    cur = line3.strip().split(" ")
    cur = list(map(float, cur))
    gt_data.append(cur)
data = np.array(gt_data)
index = np.argsort(1/data[:,0])
data = data[index]
index = cal_rank(1 / data[:,0],GAP/2)
parameter = np.polyfit(1 / data[index, 0],data[index, 1],2)
plt.scatter(1/data[index, 0], data[index, 1], s=30,  c='#ef8b8b', marker='o', label='Hippocampus : '+ str(np.around(parameter[0],decimals=2))+'*X^2+'+str(np.around(parameter[1],decimals=2))+'*X+'+str(np.around(parameter[2],decimals=2)))
p = np.poly1d(parameter)
plt.plot(1/data[index,0],p(1/data[index,0]),c='#ef8b8b')



#
gt_data = []
f2 = open(r"/home/ncclab306/zfs-pool/tes/ti/mopso/img_txt_ti/pareto_fitness_pallidum_"+MODEL+".txt", "r")
lines = f2.readlines()
for line3 in lines:
    #print(line3)
    cur = line3.strip().split(" ")
    cur = list(map(float, cur))
    gt_data.append(cur)
f2 = open(r"/home/ncclab306/zfs-pool/tes/ti/mopso/img_txt3/pareto_fitness_pallidum_"+MODEL+".txt", "r")
lines = f2.readlines()
for line3 in lines:
    #print(line3)
    cur = line3.strip().split(" ")
    cur = list(map(float, cur))
    gt_data.append(cur)
data = np.array(gt_data)
index = np.argsort(1/data[:,0])
data = data[index]
index = cal_rank(1 / data[:,0],GAP/2)
parameter = np.polyfit(1 / data[index, 0],data[index, 1],2)
plt.scatter(1/data[index, 0], data[index, 1], s=30,  c='#00b0f0', marker='o', label='Pallidum : '+ str(np.around(parameter[0],decimals=2))+'*X^2+'+str(np.around(parameter[1],decimals=2))+'*X+'+str(np.around(parameter[2],decimals=2)))
p = np.poly1d(parameter)
plt.plot(1/data[index,0],p(1/data[index,0]),c='#00b0f0')



#
gt_data = []
f2 = open(r"/home/ncclab306/zfs-pool/tes/ti/mopso/img_txt_ti/pareto_fitness_thalamus_"+MODEL+".txt", "r")
lines = f2.readlines()
for line3 in lines:
    #print(line3)
    cur = line3.strip().split(" ")
    cur = list(map(float, cur))
    gt_data.append(cur)
data = np.array(gt_data)
index = np.argsort(1/data[:,0])
data = data[index]
index = cal_rank(1 / data[:,0],GAP/2)
parameter = np.polyfit(1 / data[index, 0],data[index, 1],2)
plt.scatter(1/data[index, 0], data[index, 1], s=30,  c='gold', marker='o', label='Thalamus : '+ str(np.around(parameter[0],decimals=2))+'*X^2+'+str(np.around(parameter[1],decimals=2))+'*X+'+str(np.around(parameter[2],decimals=2)))
p = np.poly1d(parameter)
plt.plot(1/data[index,0],p(1/data[index,0]),c='gold')

# #

plt.legend(loc='best', fontsize=12, markerscale=2)
plt.title('TIs Optimization Results For '+MODEL,fontsize=20)
plt.rcParams.update({'font.size': 15})
plt.xlabel('Target Intensity (V/m)',fontsize=20)
#plt.ylabel('Mean Field Norm (V/m)',fontsize=20)  # y 轴名称旋转 38 度

plt.grid(True,linestyle = '--') # 不显示网格线
ax = plt.gca()
ax.yaxis.set_ticks_position('right')
#
plt.xlim(0.09, 0.61, 0.08)
plt.xticks(np.arange(0.1, 0.7, 0.1),size =10)
plt.ylim(0, 0.25, 0.05)
plt.yticks(np.arange(0.05, 0.25, 0.05),size =10)


# ax.spines['right'].set_color('#ccc')
# ax.spines['top'].set_color('#ccc')
# ax.spines['left'].set_color('#ccc')
# ax.spines['bottom'].set_color('#ccc')
# # 设置图像的边框粗细程度
ax.spines['right'].set_linewidth('2.0')
ax.spines['top'].set_linewidth('2.0')
ax.spines['left'].set_linewidth('2.0')
ax.spines['bottom'].set_linewidth('2.0')

t.subplots_adjust(wspace = 0.05)
t.suptitle(MODEL)
t.savefig(MODEL+'_mopso_r2.png')

plt.show()