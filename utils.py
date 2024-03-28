import numpy as np
from math import sqrt
from scipy import stats

# 一些模型评价参数的计算方法
def rmse(y,f):
    rmse = sqrt(((y - f)**2).mean(axis=0))
    return rmse
def mse(y,f):
    mse = ((y - f)**2).mean(axis=0)
    return mse
def pearson(y,f):
    rp = np.corrcoef(y, f)[0,1]
    return rp
def spearman(y,f):
    rs = stats.spearmanr(y, f)[0]
    return rs
def ci(y,f):
    ind = np.argsort(y)
    y = y[ind]
    f = f[ind]
    i = len(y)-1
    j = i-1
    z = 0.0
    S = 0.0
    while i > 0:
        while j >= 0:
            if y[i] > y[j]:
                z = z+1
                u = f[i] - f[j]
                if u > 0:
                    S = S + 1
                elif u == 0:
                    S = S + 0.5
            j = j - 1
        i = i - 1
        j = i-1
    ci = S/z
    return ci


# 计算蛋白质残基扭转角的方法有很多种，其中一种常用的方法是使用Dihedral角度计算。下面是一个使用Python计算蛋白质残基扭转角的示例代码：

import numpy as np

def calculate_dihedral(p0, p1, p2, p3):
    """
    计算给定四个原子坐标的dihedral角度
    :param p0: 第一个原子的坐标
    :param p1: 第二个原子的坐标
    :param p2: 第三个原子的坐标
    :param p3: 第四个原子的坐标
    :return: dihedral角度（以弧度为单位）
    """
    b0 = -1.0 * (p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2

    # 计算b1和b0的法向量
    b0xb1 = np.cross(b0, b1)
    b1xb2 = np.cross(b1, b2)

    # 计算b0xb1和b1xb2的法向量
    b0xb1_x_b1xb2 = np.cross(b0xb1, b1xb2)

    # 计算b0xb1和b1xb2的夹角
    cos_phi = np.dot(b0xb1, b1xb2) / (np.linalg.norm(b0xb1) * np.linalg.norm(b1xb2))
    sin_phi = np.dot(b0, b0xb1_x_b1xb2) / (np.linalg.norm(b0) * np.linalg.norm(b0xb1_x_b1xb2))

    # 计算dihedral角度
    phi = -np.arctan2(sin_phi, cos_phi)

    return phi


# # 定义原子坐标
# p0 = np.array([1.0, 2.0, 3.0])
# p1 = np.array([4.0, 5.0, 6.0])
# p2 = np.array([7.0, 8.0, 9.0])
# p3 = np.array([10.0, 11.0, 12.0])
#
# # 计算dihedral角度
# dihedral_angle = calculate_dihedral(p0, p1, p2, p3)

# print("蛋白质残基的扭转角度：", np.degrees(dihedral_angle))  # 以角度为单位打印结果

import numpy as np

def calculate_phi_psi(prior_residue, current_residue, next_residue):
    # 获取残基的C、CA、N、C_alpha和C_beta原子的坐标
    prior_C = prior_residue['C'].get_coord()
    current_N = current_residue['N'].get_coord()
    current_CA = current_residue['CA'].get_coord()
    current_C = current_residue['C'].get_coord()
    next_N = next_residue['N'].get_coord()

    # 计算phi角度
    phi = calculate_dihedral(prior_C, current_N, current_CA, current_C)

    # 计算psi角度
    psi = calculate_dihedral(current_N, current_CA, current_C, next_N)

    return phi, psi

def calculate_dihedral(p1, p2, p3, p4):
    b0 = -1.0 * (p2 - p1)
    b1 = p3 - p2
    b2 = p4 - p3

    # 计算b1和b2的单位法向量
    b1 /= np.linalg.norm(b1)
    b2 /= np.linalg.norm(b2)

    # 计算b0和b1的法向量
    b0xb1 = np.cross(b0, b1)

    # 计算b1和b2的法向量
    b1xb2 = np.cross(b1, b2)

    # 计算b0与b1xb2的夹角
    # phi_psi = np.sin(np.degrees(np.arccos(np.dot(b0xb1, b1xb2) / (np.linalg.norm(b0xb1) * np.linalg.norm(b1xb2))))/180*np.pi)

    if -1 <= np.dot(b0xb1, b1xb2)/(np.linalg.norm(b0xb1) * np.linalg.norm(b1xb2)) <= 1:

        phi_psi = np.degrees(np.arccos(np.dot(b0xb1, b1xb2) / (np.linalg.norm(b0xb1) * np.linalg.norm(b1xb2))))
    else:
        # print('cross的两个值',np.dot(b0xb1, b1xb2)/(np.linalg.norm(b0xb1) * np.linalg.norm(b1xb2)))
        if np.dot(b0xb1, b1xb2)/(np.linalg.norm(b0xb1) * np.linalg.norm(b1xb2)) > 1:
            phi_psi = 0
        if np.dot(b0xb1, b1xb2)/(np.linalg.norm(b0xb1) * np.linalg.norm(b1xb2)) < -1:
            phi_psi = 180
        # print(phi_psi)
    return phi_psi


