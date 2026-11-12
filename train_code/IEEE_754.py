import struct
import numpy as np
import math
from sympy import *


def floatToBinary32(value):
    val = '{:032b}'.format(struct.unpack('I', struct.pack('f', value))[0])
    return str(val)


def Binary32Tofloat(B):  # B是二进制字符串
    return struct.unpack('<f', struct.pack('<I', int(B, 2)))[0]
    # return '{:.07f}'.format(struct.unpack('<f', struct.pack('<I', int(B, 2)))[0])


# def findbits(tempB, tempavail, Comprebits, C):
#     # ********************<<输入参数>>****************
#     # 参数1表示：提取的特征比特,
#     # 参数2表示：特征比特对应的标签(1表示篡改,0表示未篡改),
#     # 参数3表示：正确ref(包含0和1)和错误ref(用2表示),
#     # 参数4表示：扩展矩阵其大小为[ref_num,cha_num]
#
#     # ********************<<输出参数>>*****************
#     # 参数1(label)表示：
#     # 参数2(lth)表示：恢复特征比特的个数
#     # 参数3(y)表示：恢复后的特征比特
#
#     if sum(tempavail) == 0:  # 表示特征比特未被篡改,不需要进行恢复
#         label = 0
#         lth = 0
#         y = 0
#         return label, lth, y
#
#     nt = Comprebits.size
#     for i in range(nt):  # 根据参考比特的篡改标记,找出对应扩展矩阵中无效的行,并记录有效行的数目
#         if Comprebits[i] == 2:
#             C[i, :] = 0
#             nt = nt - 1  # 扩展矩阵中有效行的数目
#
#     Y = np.transpose(Comprebits)  # (256,1)  将ref转置
#     length = tempavail.size
#     CC = np.empty(shape=[len(Comprebits), 0])
#     for i in range(length):
#         if tempavail[i] == 1:
#             CC = np.c_[CC, C[:, i]]  # 根据特征比特标签,提取扩展矩阵中无效的列
#         else:
#             Y = np.mod(Y + C[:, i] * tempB[i], 2)  # 正确特征比特乘上扩展矩阵正确列再加上ref的转置然后再对2求余???????????
#     h, w = CC.shape
#     if (w > h):  # 当错误的太多则无解?
#         label = 0
#         lth = 0
#         y = 0
#         return label, lth, y
#
#     for k in range(w):
#         value_i = max(CC[k:h, k])
#         index_i = CC[k:h, k].argmax()
#         if value_i == 0:  # 如果
#             label = 0
#             lth = 0
#             y = 0
#             return label, lth, y
#         index_i = int(index_i + k)
#         CC[k], CC[index_i] = CC[index_i].copy(), CC[k].copy()
#         Y[k], Y[index_i] = Y[index_i].copy(), Y[k].copy()
#         for j in range(int(k + 1), int(h)):
#             if CC[j, k] == 1.0:
#                 line_datas = np.mod(CC[j, :] + CC[k, :], 2)
#                 CC[j, :] = line_datas
#                 Y[j] = np.mod(Y[j] + Y[k], 2)
#     lth = w
#     y = Y[0:lth]
#     for i in range(lth - 2, -1, -1):
#         for kj in range(i + 1, lth):
#             y[i] = np.mod(y[i] + y[kj] * CC[i, kj], 2)
#     y = np.mod(y, 2)
#     label = 1
#     return label, lth, y

# def cha_recovery(tempB, tempavail, Comprebits, C):
#     # ref_error_pos = np.argwhere(Comprebits == 2)
#     # ref_error_pos = np.empty(shape=[0, len(Comprebits)])
#     # ref_true_pos = np.empty(shape=[0, len(Comprebits)])
#     ref_error_pos = []
#     ref_true_pos = []
#     True_ref_bits = []
#     for i in range(len(Comprebits)):
#         if Comprebits[i] == 2:
#             # ref_error_pos = np.c_[ref_error_pos, C[i, :]]
#             ref_error_pos.append(C[i, :])
#         else:
#             # ref_true_pos = np.c_[ref_true_pos, C[i, :]]  # 列不变与cha一致
#             ref_true_pos.append(C[i, :])
#             True_ref_bits.append(Comprebits[i])
#     ref_error_pos = np.array(ref_error_pos)
#     ref_true_pos = np.array(ref_true_pos)
#     True_ref_bits = np.array(True_ref_bits)
#
#     # cha_error_pos = np.empty(shape=[len(tempB), 0])
#     # cha_true_pos = np.empty(shape=[len(tempB), 0])
#     cha_error_pos = []
#     cha_true_pos = []
#     True_cha_bits = []
#     False_cha_bits = []
#     for i in range(len(tempB)):
#         if tempavail[i] == 1:
#             # cha_error_pos = np.c_[cha_error_pos, ref_true_pos[:, i]]
#             cha_error_pos.append(ref_true_pos[:, i])
#             False_cha_bits.append(tempB[i])
#         else:
#             # cha_true_pos = np.c_[cha_true_pos, ref_true_pos[:, i]]  # 行不变与正确的ref一致
#             cha_true_pos.append(ref_true_pos[:, i])
#             True_cha_bits.append(tempB[i])
#
#     cha_error_pos = np.array(cha_error_pos)
#     cha_true_pos = np.array(cha_true_pos)  # 72 * 148
#     True_cha_bits = np.array(True_cha_bits)  # 72
#     False_cha_bits = np.array(False_cha_bits)
#
#     # cha_true_pos_new = np.mod(np.matmul(cha_true_pos, True_cha_bits), 2)
#     cha_true_pos_new = np.mod(np.matmul(True_cha_bits, cha_true_pos), 2)
#     true_bits = np.mod(True_ref_bits - cha_true_pos_new, 2)
#     r, c = true_bits.shape
#     if r == c:
#         False_cha_bits_new = np.linalg.inv(False_cha_bits)
#     else:
#         print('the matrix is Non-reversible...')
#         False_cha_bits_new = np.linalg.pinv(False_cha_bits)  # 求伪逆矩阵
#         lth = 0
#
#     recovery_bits = np.mod(np.matmul(False_cha_bits_new, true_bits), 2)
#     return recovery_bits, lth


def findbits(tempB, tempavail, Comprebits, C):
    #     # ********************<<输入参数>>****************
    #     # 参数1表示：提取的特征比特,
    #     # 参数2表示：特征比特对应的标签(1表示篡改,0表示未篡改),
    #     # 参数3表示：正确ref(包含0和1)和错误ref(用2表示),
    #     # 参数4表示：扩展矩阵其大小为[ref_num,cha_num]
    #
    #     # ********************<<输出参数>>*****************
    #     # 参数1(label)表示：
    #     # 参数2(lth)表示：恢复特征比特的个数
    #     # 参数3(y)表示：恢复后的特征比特
    if sum(tempavail) == 0:
        label = 0
        lth = 0
        y = 0
        return label, lth, y

    nt = Comprebits.size
    for i in range(nt):
        if Comprebits[i] == 2:
            C[i, :] = 0
            nt = nt - 1

    Y = np.transpose(Comprebits)  # (256,1)
    length = tempavail.size
    CC = np.empty(shape=[Comprebits.size, 0])
    for i in range(length):
        if tempavail[i] == 1:
            CC = np.c_[CC, C[:, i]]
        else:
            Y = np.mod(Y + C[:, i] * tempB[i], 2)
    h, w = CC.shape
    if (w > h):
        label = 0
        lth = 0
        y = 0
        return label, lth, y

    for k in range(w):
        value_i = max(CC[k:h, k])
        index_i = CC[k:h, k].argmax()
        if value_i == 0:
            label = 0
            lth = 0
            y = 0
            return label, lth, y
        index_i = int(index_i + k)
        tem_CC = CC[k, :]
        CC[k, :] = CC[index_i, :]
        CC[index_i, :] = tem_CC
        tem_Y = Y[k]
        Y[k] = Y[index_i]
        Y[index_i] = tem_Y
        for j in range(int(k + 1), int(h)):
            if CC[j, k] == 1.0:
                line_datas = np.mod(CC[j, :] + CC[k, :], 2)
                CC[j, :] = line_datas
                tem_data = np.mod(Y[j] + Y[k], 2)
                Y[j] = tem_data
    lth = w
    y = Y[0:lth]
    for i in range(lth - 1, -1, -1):
        for kj in range(i + 1, lth):
            y[i] = np.mod(y[i] + y[kj] * CC[i, kj], 2)
    y = np.mod(y, 2)
    label = 1
    return label, lth, y


def func3(a, b):
    s = a * b
    while a % b != 0:
        a, b = b, (a % b)
    else:
        print("最大公约数是：", b)
        print("最小公倍数是：", s // b)
    return b
