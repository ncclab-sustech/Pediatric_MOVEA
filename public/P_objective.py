import numpy as np
#from public.util import *
from public.util import function0, function1_s, function2_s, h,tdcs_function1,tdcs_function1_avoid,tdcs_function2,tdcs_function3,tdcs_function4,tis_constraint6,tis_function6,tis_function_avoid

import multiprocessing

# def fun_(x, a=1):
#
#     cv_value = h(x) * a
#     return [tdcs_function1(x) + cv_value, tdcs_function2(x) + cv_value]

def fun_(x):
    cv_value = h(x)
    #avoid_value = tdcs_function1_avoid(x)
    if cv_value > 0.01:
        result = [100000 * cv_value, 100000 * cv_value]
    # elif avoid_value < 10:
    #     result = [10000 * 1 / avoid_value, 10000 * 1 / avoid_value]
    else:
        intensity = tdcs_function1(x)
        # if intensity > 10:
        #     result = [intensity * intensity, tdcs_function2(x) * intensity]
        # else:
        result = [intensity, tdcs_function2(x)]
    return result


# def fun_(x):
#     if cv_value > 0.01:
#         temp = tdcs_function1(x)
#         if temp > 10 : # 1
#             result = np.array([temp,tdcs_function2(x)]) * temp # 2
#         else:
#             result = np.array([temp,tdcs_function2(x)])
#     return result

def P_objective(Operation,Problem,M,Input,epoch=0):
    [Output, Boundary, Coding] = TES(Operation, Problem, M, Input,epoch)
    #[Output, Boundary, Coding] = P_DTLZ(Operation, Problem, M, Input)
    #if Boundary == []:
    if len(Boundary) ==0:
        return Output
    else:
        return Output, Boundary, Coding


def TES(Operation,Problem,M,Input,epoch):
    Boundary = []
    Coding = ""
    if Operation == "init":
        MaxValue = np.ones((1, 75))
        MinValue = -np.ones((1, 75))
        Population = np.random.uniform(-1, 1, size=(Input, 75))
        Boundary = np.vstack((MaxValue, MinValue))
        Coding = "Real"
        return Population, Boundary, Coding
    elif Operation == "value":
        Population = Input
        FunctionValue = np.zeros((Population.shape[0], M))
        if Problem == "TES":
            FunctionValue[:, 2] = np.array([h(s) for s in Population])
            FunctionValue[:, 1] = np.array([function1_s(s) for s in Population])
            FunctionValue[:, 0] = np.array([function2_s(s) for s in Population])
        if Problem == "TEScv":
            cv = np.array([h(s) for s in Population])
            for i in range(len(Population)):
                if cv[i] > 10e-4:
                    FunctionValue[i, 0] = cv[i] + 10e10
                    FunctionValue[i, 1] = cv[i] + 10e10
                else:
                    FunctionValue[i, 0] = function1_s(Population[i])
                    FunctionValue[i, 1] = function2_s(Population[i])

        if Problem == "TEScv2":
            # cv = np.array([h(s) for s in Population])
            # FunctionValue[:, 0] = np.array([function1_s(s) for s in Population]) + cv * 5
            # FunctionValue[:, 1] = np.array([function2_s(s) for s in Population]) + cv * 5
            p = multiprocessing.Pool(100)
            FunctionValue = np.array(p.map(fun_, Population))
            p.close()
            p.join()

            # cv = np.array([h(s) for s in Population])
            # FunctionValue[:, 0] = np.array([function1_s(s) for s in Population]) + cv * np.sqrt(epoch + 1)
            # FunctionValue[:, 1] = np.array([function2_s(s) for s in Population]) + cv * np.sqrt(epoch + 1)

        return FunctionValue, Boundary, Coding







