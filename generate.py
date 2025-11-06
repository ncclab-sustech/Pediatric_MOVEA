import numpy  as np


pos = np.load(r'/media/ncclab/database7/wangm/simnibs_leadfield/pos.npy')
lfm = np.load(r'/media/ncclab/database7/wangm/simnibs_leadfield/lfm_hcp4_2.npy')

TARGET_POSITION = [-31, -20, -14] # hippo #[-39, 34, 37] dlpfc 
position = TARGET_POSITION
distance = np.zeros(len(pos))
for i in range(len(pos)):
    distance[i] = (pos[i, 0] - position[0])**2 + (pos[i, 1] - position[1])**2 + (pos[i, 2] - position[2])**2

AVOID_POSITION = np.where(distance < 10**2)
AVOID_POSITION = AVOID_POSITION[0]
print(len(AVOID_POSITION))
print(len(AVOID_POSITION)/len(pos))


NUM_ELE = 75
def tis_function6_focality(x):
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
    return eam[AVOID_POSITION],eam

x = np.array([5.000000000000000000e-01,5.000000000000000000e-01,3.108108108108108003e-01,1.216216216216216284e-01,2.702702702702702853e-01,1.621621621621621712e-01
])

a,b = tis_function6_focality(x)
label = '4_ti_dlpfc'
np.save("all_"+label+".npy",b)
np.save("target_"+label+".npy",a)
print("done")

def tdcs_function_vol(s):
    eam = ((np.matmul(lfm[:, :, 0].T, s)) ** 2 + (np.matmul(lfm[:, :, 1].T, s)) ** 2 + (
        np.matmul(lfm[:, :, 2].T, s)) ** 2) ** 0.5
    return eam[AVOID_POSITION],eam

x = np.array([-1.219389345988404136e-04,1.191243065794727989e-03,2.784017921493447262e-03,-2.488132898346687508e-03,-3.328776557210463838e-03,-2.020358876306665996e-04,-5.960840174932357030e-03,3.965220321144940704e-03,-9.042086018081769982e-03,-1.000000000000000000e+00,-7.350143553424681037e-04,-2.043168004298235099e-03,-6.069031538476927015e-03,-4.771415135080957041e-03,-6.976062977036425611e-03,7.296986793004658564e-03,-2.942365853734028522e-03,-5.187528381155217042e-03,1.765371730754720578e-03,-1.229641966874748301e-03,-1.000000000000000000e+00,8.106945877305194587e-03,1.000000000000000000e+00,1.000000000000000000e+00,-1.505252934292265407e-04,9.622763426712843335e-04,2.325395769057109728e-03,5.898720688432262244e-04,6.832974368564287068e-03,-5.864685183948629456e-03,3.130507776878535715e-03,-1.602938543144547499e-03,-4.915870478163822041e-03,-1.143664016253339785e-03,-2.251078329162737104e-03,-6.199085304791400464e-03,8.228393035269654029e-04,7.206582892773719107e-04,1.818325698150136203e-03,1.501903879689894458e-03,5.255926199133802504e-04,-6.162495392008376753e-03,-1.504548250626682251e-03,-1.835550393457908644e-03,-4.564037415704066611e-03,2.513712496347944060e-03,-2.117041170366143997e-03,3.875172110879210839e-03,1.493928409528826051e-03,2.192292461089938999e-03,-9.795209798337109128e-04,1.962977099875930307e-03,-4.743547607430844207e-03,4.435358274172411693e-03,-2.875912155003892967e-03,-1.524138271437773702e-03,-1.180203867195531858e-03,3.624956328433293073e-03,3.654617137674462737e-04,-6.062583970127437445e-03,1.882984282031295760e-03,-1.944679511483451886e-03,1.371691308412983923e-03,2.186757992820113558e-03,7.454899173823640451e-03,1.488810265418823338e-03,-9.995558376867282047e-05,1.139025496423018771e-02,2.210993820576623650e-03,-4.656391547891852722e-04,-6.534854095331604253e-03,-2.781380992024686319e-03,-3.318853673097213464e-03,3.517135547915625424e-04,5.388973546147824908e-03
])

a,b = tdcs_function_vol(x)
label = '4_tacs_dlpfc'
np.save("all_"+label+".npy",b)
np.save("target_"+label+".npy",a)
print("done")
