import numpy as np
import itertools
import random
from IEEE_754 import *


def bit_flip(input_weight, block_size, flip_bits, hy_seed):
    # ********************<<后门攻击1:翻转指数位的最高位>>********************
    print('The attack method is bit flip...')
    weight_num1 = len(input_weight[0])
    total_weight = np.array(list(itertools.chain(*input_weight)))
    total_num = len(total_weight)
    block_num = int(np.floor(total_num / block_size))
    real_tamper_range = block_num * block_size
    real_weight = total_weight[0:real_tamper_range]
    rest_weight = total_weight[real_tamper_range:]
    tamper_para_num = flip_bits
    random.seed(hy_seed)
    tamper_block_pos = random.sample(range(0, len(real_weight)), tamper_para_num)
    for para_idx in range(tamper_para_num):
        tem_para = real_weight[tamper_block_pos[para_idx]]
        tem_binary = floatToBinary32(float(tem_para))
        if tem_binary[1] == '0':
            tem_value_attacked = tem_binary[0] + '1' + tem_binary[2:32]
        else:
            tem_value_attacked = tem_binary[0] + '1' + tem_binary[2:32]

        real_weight[tamper_block_pos[para_idx]] = Binary32Tofloat(tem_value_attacked)
        print('The tampered weight is:', Binary32Tofloat(tem_value_attacked))

    if len(rest_weight) == 0:
        real_weight_tem = real_weight
    else:
        real_weight_tem = np.hstack([real_weight, rest_weight])

    attacked_weight = []
    attacked_weight.append(real_weight_tem[0:weight_num1])
    attacked_weight.append(real_weight_tem[weight_num1:total_num])
    return attacked_weight


def ran_block(input_weight, tamper_rate, block_size, mask_path, hy_seed):
    # *****************************按照篡改率选择若干个篡改块，每个篡改块内选择48个参数置为0***************************
    weight_num1 = len(input_weight[0])
    total_weight = np.array(list(itertools.chain(*input_weight)))
    total_num = len(total_weight)
    block_num = int(np.floor(total_num / block_size))
    # 篡改攻击只对进行了水印嵌入参数进行
    real_tamper_range = block_num * block_size
    real_weight = total_weight[0:real_tamper_range]
    rest_weight = total_weight[real_tamper_range:]
    real_mask = np.zeros(block_num, dtype=int)
    tamper_block_num = int(np.floor(block_num * tamper_rate))
    if tamper_block_num == 0:
        tamper_block_num = 1
        print('The paras are too small, and the number of tampered block is set to 1')

    random.seed(hy_seed)
    tamper_block_pos = random.sample(range(0, block_num), tamper_block_num)
    for i in range(tamper_block_num):
        real_mask[tamper_block_pos[i]] = 1
        tem_block_weight = real_weight[tamper_block_pos[i] * block_size:(tamper_block_pos[i] + 1) * block_size]
        random.seed(926)
        tem_pos = random.sample(range(0, block_size), 60)
        for j in range(len(tem_pos)):
            tem_block_weight[tem_pos[j]] = 0
        real_weight[tamper_block_pos[i] * block_size:(tamper_block_pos[i] + 1) * block_size] = tem_block_weight

    np.save('{}.npy'.format(mask_path), real_mask)
    if len(rest_weight) == 0:
        real_weight_tem = real_weight
    else:
        real_weight_tem = np.hstack([real_weight, rest_weight])

    attacked_weight = []
    attacked_weight.append(real_weight_tem[0:weight_num1])
    attacked_weight.append(real_weight_tem[weight_num1:total_num])

    return attacked_weight


def ran_weight(input_weight, tamper_rate, block_size):
    # ******************************<<按照篡改率，随机将若干个参数置为0>>******************************
    weight_num1 = len(input_weight[0])
    total_weight = np.array(list(itertools.chain(*input_weight)))
    total_num = len(total_weight)
    tamper_num = int(np.floor(total_num * tamper_rate))
    block_num = int(np.floor(total_num / block_size))
    real_mask = np.zeros(block_num, dtype=int)
    tamper_pos = random.sample(range(0, total_num), tamper_num)
    for i in range(tamper_num):
        total_weight[tamper_pos[i]] = 0

    for j in range(tamper_num):
        pos_tem = int(np.floor(tamper_pos[j] / block_size))
        real_mask[pos_tem] = 1
    attacked_weight = []
    attacked_weight.append(total_weight[0:weight_num1])
    attacked_weight.append(total_weight[weight_num1:total_num])
    np.save('real_mask.npy', real_mask)
    return attacked_weight


def ran_gauss_dis(input_weight, tamper_rate):
    # ********************************将前两个参数置为0*****************************
    # input_weight[0:2] = 0  # 表示将前面128个参数置为0，表示总共修改了128*32个比特位
    input_weight[0][0] = 0
    input_weight[1][0] = 0
    attacked_weight = input_weight
    print('The tampered weight is:', attacked_weight)
    return attacked_weight


def ran_no(input_weight, tamper_rate):
    # *******************************无篡改*******************************
    attacked_weight = input_weight
    return attacked_weight


def ran_block_FC(input_weight, tamper_rate, block_size, mask_path):
    # *****************************单独篡改FC层按照篡改率选择若干个篡改块，每个篡改块内选择48个参数置为0***************************
    weight_num1 = len(input_weight[0])
    total_weight = input_weight[0]
    total_num = len(total_weight)
    block_num = int(np.floor(total_num / block_size))
    real_mask = np.zeros(block_num, dtype=int)
    tamper_block_num = int(np.floor(block_num * tamper_rate))
    tamper_block_pos = random.sample(range(0, block_num), tamper_block_num)
    for i in range(tamper_block_num):
        real_mask[tamper_block_pos[i]] = 1
        tem_block_weight = total_weight[tamper_block_pos[i] * block_size:(tamper_block_pos[i] + 1) * block_size]
        tem_pos = random.sample(range(0, block_size), 60)
        for j in range(len(tem_pos)):
            tem_block_weight[tem_pos[j]] = 0
        total_weight[tamper_block_pos[i] * block_size:(tamper_block_pos[i] + 1) * block_size] = tem_block_weight

    # mask_path = './tamper_mask/FC/real_mask' + '_' + str(tamper_rate)
    real_mask = np.hstack([real_mask, np.zeros(int(np.floor(len(input_weight[1]) / block_size)), dtype=int)])
    np.save('{}.npy'.format(mask_path), real_mask)
    attacked_weight = []
    attacked_weight.append(total_weight)
    attacked_weight.append(input_weight[1])
    return attacked_weight


def ran_block_Conv1(input_weight, tamper_rate, block_size, mask_path):
    # *****************************单独篡改Conv1层，按照篡改率选择若干个篡改块，每个篡改块内选择60个参数置为0***************************
    weight_num1 = len(input_weight[0])
    # total_weight = np.array(list(itertools.chain(*input_weight)))
    total_weight = input_weight[1]
    total_num = len(total_weight)
    block_num = int(np.floor(total_num / block_size))
    real_mask = np.zeros(block_num, dtype=int)
    tamper_block_num = int(np.floor(block_num * tamper_rate))
    tamper_block_pos = random.sample(range(0, block_num), tamper_block_num)
    for i in range(tamper_block_num):
        real_mask[tamper_block_pos[i]] = 1
        tem_block_weight = total_weight[tamper_block_pos[i] * block_size:(tamper_block_pos[i] + 1) * block_size]
        tem_pos = random.sample(range(0, block_size), 60)
        for j in range(len(tem_pos)):
            tem_block_weight[tem_pos[j]] = 0
        total_weight[tamper_block_pos[i] * block_size:(tamper_block_pos[i] + 1) * block_size] = tem_block_weight

    # mask_path = './tamper_mask/Conv1/real_mask' + '_' + str(tamper_rate)
    real_mask = np.hstack([np.zeros(int(np.floor(len(input_weight[0]) / block_size)), dtype=int), real_mask])
    np.save('{}.npy'.format(mask_path), real_mask)
    attacked_weight = []
    attacked_weight.append(input_weight[0])
    attacked_weight.append(total_weight)
    return attacked_weight
