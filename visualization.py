import os.path

# import simnibs
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from nilearn import plotting, image
from nibabel.affines import apply_affine
from public import glo


ALL_NUM_ELE =  glo.NUM_ELE

if glo.head_model == '0026104':
    print("loading 0026104")
    # grey and white matter
    lfm = np.load(r'/home/ncclab/Desktop/movea/lfm_0026104_2.npy')
    print("load grey and white matter",lfm.shape)
    #lfm = lfm[[0,1,2,9,11,13,15,17,31,33,36,38,52,54,56,58,60,69,70,71]]
    # for train
    pos = np.load(r'/home/ncclab/Desktop/movea/pos_0026104_2.npy')
    #pos = np.load('/home/ncclab306/database7/wangm/simnibs_leadfield/pos_269731_mni.npy')
    print("load position")
    print(pos.shape)


    # If you don't have segmentation result, you can use mri.nii and then delete interpolate from Line107-121
    MASK_PATH = r'/home/ncclab/Desktop/movea/0026046_final_contr.nii.gz'
    mask = nib.load(MASK_PATH)

pos = apply_affine(np.linalg.inv(mask.affine), pos) 
SAVE_PATH = r'/home/ncclab/Desktop/movea/result_{}.nii.gz'



def envelop(e1, e2):
    e1 = np.sqrt(np.sum(e1 * e1, axis=1))
    e2 = np.sqrt(np.sum(e2 * e2, axis=1))
    min_values = np.minimum(e1, e2)
    result = 2 * min_values
    
    return result

def mti(x):
        x = x/2
        x[np.where(abs(x)<0.01)] = 0
        e1 = np.array([np.matmul(lfm[:, :, 0].T, x[:75]), np.matmul(lfm[:, :, 1].T, x[:75]),np.matmul(lfm[:, :, 2].T, x[:75])]).T /1000 # 1000 is to unify the unit from mV/m to V/m
        e2 = np.array([np.matmul(lfm[:, :, 0].T, x[75:]), np.matmul(lfm[:, :, 1].T, x[75:]),np.matmul(lfm[:, :, 2].T, x[75:])]).T /1000
        return envelop(e1,e2)
         

def tis_function6(x):

    electrode1 = int(round(x[2] * 74))
    electrode2 = int(round(x[3] * 74))
    stimulation1 = np.zeros(75)
    stimulation1[electrode1] = 2 * x[0]
    stimulation1[electrode2] = -(2 * x[0])
    e1 = np.array([np.matmul(lfm[:, :, 0].T, stimulation1), np.matmul(lfm[:, :, 1].T, stimulation1),np.matmul(lfm[:, :, 2].T, stimulation1)]).T /1000

    electrode3 = int(round(x[4] * 74))
    electrode4 = int(round(x[5] * 74))
    stimulation2 = np.zeros(75)

    stimulation2[electrode3] = 2 * x[1]
    stimulation2[electrode4] = -(2 * x[1])
    e2 = np.array([np.matmul(lfm[:, :, 0].T, stimulation2), np.matmul(lfm[:, :, 1].T, stimulation2),np.matmul(lfm[:, :, 2].T, stimulation2)]).T /1000
    return envelop(e1,e2) 


def tdcs(s):
    field = ((np.matmul(lfm[:, :, 0].T, s)) ** 2 + (np.matmul(lfm[:, :, 1].T, s)) ** 2 + (
        np.matmul(lfm[:, :, 2].T, s)) ** 2) ** 0.5
    return abs(field)/1000


def visual(selected_rows, name):


    ob = mask.get_fdata()
    print(ob.shape)

    if glo.type == 'ti':
        x = tis_function6(selected_rows)
    if glo.type == 'mti':
        x = mti(selected_rows)
    if glo.type == 'tdcs':
        x = tdcs(selected_rows)
    print(x.shape)
    con = np.zeros(ob.shape)

    for idx, i in enumerate(pos):
        con[int(i[0]), int(i[1]), int(i[2])] = x[idx]

    point = np.array([np.where(con > 0)[0], np.where(con > 0)[1], np.where(con > 0)[2]]).T

    ##some values in eam were overlapped
    v = np.zeros(len(point))
    for i in range(len(point)):
        v[i] = con[point[i, 0], point[i, 1], point[i, 2]]
    point_grid = np.array([np.where(((ob == 2) | (ob == 1)))[0], np.where(((ob == 2) | (ob == 1)))[1],  # here labels are gm and wm 
                           np.where(((ob == 2) | (ob == 1)))[2]]).T  # mask < 4, gm == 1

    print(point_grid.shape)
    print(len(np.where(x > 0)[0]))

    #intrpolate starts
    from scipy.interpolate import griddata

    value = griddata(point, v, point_grid, method='nearest')
    print(value)
    print('value1', value[0])
    print("finish interpolate")
    ob = np.zeros(ob.shape)
    for i in range(len(point_grid)):
        ob[point_grid[i, 0], point_grid[i, 1], point_grid[i, 2]] = value[i]

    print(ob[ob > 0].size)
    print(ob.shape)
    con = ob
    #intrpolate ends

    img = nib.Nifti1Image(con, mask.affine)
    nib.save(img, SAVE_PATH.format(name))
    

    if not os.path.exists(SAVE_PATH.format(name)):
        print('ti model does not exist. ')
    else:
        img = image.load_img(SAVE_PATH.format(name))
        bg = image.load_img(MASK_PATH)
        target_pos = glo.position.copy()
        plotting.plot_roi(img,
                          bg_img=bg,
                          cut_coords=target_pos,
                          black_bg=False,
                          cmap=plt.get_cmap('jet'),
                          output_file=f'result/ti_result_{name}.png')


if __name__ == '__main__':
    # ele = [41, 52, 42, 43]
    # input_ele = input("Enter electrode plan (Example: [5, 59, 55, 10, 0.89]): ")
    # ele = ast.literal_eval(input_ele)
    # input_pos = input("Enter target position (Example: [-1.5,-2,2]): ")
    # glo.position = ast.literal_eval(input_pos)
    # glo.direction = input("Enter field direction (m/x/y/z): ")
    # ele = [34 , 26 , 32 , 22 , 0.8281365131442178]
    glo.NUM_ELE=75
    glo.type = 'tdcs'
    glo.position = [-31, -20, -14]
    visual([1,1,46/74, 2 /74, 74/74 , 23/74], 'test')
