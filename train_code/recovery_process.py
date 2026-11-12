import time
import itertools
import numpy as np
from IEEE_754 import *


def recovery_operate(total_weight, total_ori_weight, group_num, block_size, cha_per_bits, hash_per_bits, diff_val,
                     tamper_rate, npy_load_path, mask_path, sorted_score):
    # ********************************<<参数说明>>*********************** #
    # total_weight:已经排序好的所选层的所有参数
    # total_ori_weight: 原始未排序的所选层所有参数
    # group_num:初始给定的分组数
    # block_size:分块大小
    # cha_per_bits:每个参数的特征比特个数
    # hash_per_bits:每个块的hash比特数
    # diff_val:等差数组的公差
    # npy_load_path:原始特征比特的保存路径
    # mask_path:真实篡改mask的保存路径
    # ********************************<<参数说明>>*********************** #
    block_num = int(np.floor(len(total_weight) / block_size))
    print('The total number of block:', block_num)
    rest_weight = total_weight[block_size * block_num:]
    real_weight = total_weight[0:block_size * block_num]

    rest_ori_weight = total_ori_weight[block_size * block_num:]
    real_ori_weight = total_ori_weight[0:block_size * block_num]
    # ************<<将排序参数转化为32位二进制比特>>************** #
    total_weight_bits = []
    for tem_weight in real_weight:
        tem_value = tem_weight
        total_weight_bits.append(floatToBinary32(float(tem_value)))

    real_bits = np.array(list(itertools.chain(*total_weight_bits)))
    real_bits = real_bits.astype(int)
    real_bits_32 = real_bits.reshape(len(real_weight), 32)

    total_cha_bits = real_bits_32[:, 0:cha_per_bits]  # 排序后的所有特征比特

    # ************<<将原始参数转化为32位二进制比特>>************** #
    total_ori_weight_bits = []
    for tem_weight1 in real_ori_weight:
        tem_value1 = tem_weight1
        total_ori_weight_bits.append(floatToBinary32(float(tem_value1)))

    real_ori_bits = np.array(list(itertools.chain(*total_ori_weight_bits)))
    real_ori_bits = real_ori_bits.astype(int)
    real_ori_bits_32 = real_ori_bits.reshape(len(real_ori_weight), 32)

    # 计算剩余的可嵌入空间
    total_select_weight_num = len(real_weight)
    rest_capacity = total_select_weight_num * (32 - cha_per_bits) - (hash_per_bits * block_num)

    # 取特征位和水印位比特
    ext_cha_bits = real_ori_bits_32[:, 0:cha_per_bits]
    ext_marked_bits = real_ori_bits_32[:, cha_per_bits:32]

    # ***********************************************<<进行篡改定位>>********************************************** #
    # ***********************************************<<进行篡改定位>>********************************************** #
    print('# =============================<<The process of tamper localization>>============================= #')
    cha_per_block_num = cha_per_bits * block_size
    ref_per_block_num = int((32 - cha_per_bits) * block_size - hash_per_bits)
    Ext_ref_bits_total = np.zeros(ref_per_block_num * block_num, dtype=int)
    block_label = np.zeros(block_num, dtype=int)
    Total_cha_bits_label = np.zeros((len(real_weight), cha_per_bits), dtype=int)
    Tamper_block_num = 0
    time_start = time.perf_counter()  # 记录开始时间
    for block_idx in range(block_num):
        cha_per_block = ext_cha_bits[block_idx * block_size:(block_idx + 1) * block_size, :].flatten()
        Ext_ref_bits = ext_marked_bits[block_idx * block_size:(block_idx + 1) * block_size,
                       0:32 - cha_per_bits - 2].flatten()
        Ext_hash_bits = ext_marked_bits[block_idx * block_size:(block_idx + 1) * block_size, -2:].flatten()
        np.random.seed(block_idx)  # 随机种子 密钥
        A_hash = np.random.randint(0, 2, (hash_per_bits, cha_per_block_num + ref_per_block_num))
        Cal_hash_bit = np.mod(np.matmul(A_hash, np.hstack([cha_per_block, Ext_ref_bits])), 2)  # 可能存在问题

        # ***************<<对比哈希值>>**************** #
        if (Ext_hash_bits == Cal_hash_bit).all():
            Ext_ref_bits_total[block_idx * ref_per_block_num:(block_idx + 1) * ref_per_block_num] = Ext_ref_bits
        else:
            Total_cha_bits_label[block_idx * block_size:(block_idx + 1) * block_size, :] = 1
            Ext_ref_bits_total[block_idx * ref_per_block_num:(block_idx + 1) * ref_per_block_num] = np.ones(
                ref_per_block_num, dtype=int) * 2
            block_label[block_idx] = 1
            Tamper_block_num = Tamper_block_num + 1

    time_end = time.perf_counter()  # 记录结束时间
    time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
    print('The time consumed by the entire tamper location process (s):', time_sum)
    print('The number of tampered block:', Tamper_block_num)
    print('The number of calculated tampered weights:', Tamper_block_num * block_size)

    # 加载真实篡改mask

    # real_mask = np.load('{}.npy'.format(mask_path))
    # tamper_acc = np.sum((real_mask - block_label) == 0)
    # print('the equal block number:', tamper_acc)
    # print('The accuracy of tamper detection:{} %'.format((tamper_acc / len(real_mask)) * 100))
    # print('\n')
    # ================================================<<篡改内容恢复过程>>================================================ #
    # ================================================<<篡改内容恢复过程>>================================================ #

    # 注意现在对应的特征比特的label是对应原始未排序的特征比特的，但是在进行参考比特生成时是根据排序后的特征比特生成的，因此需要将label进行等级排序
    sorted_total_cha_bits_label = np.zeros((Total_cha_bits_label.shape[0], Total_cha_bits_label.shape[1]), dtype=int)
    for label_idx1 in range(len(real_weight)):
        sorted_total_cha_bits_label[label_idx1, :] = Total_cha_bits_label[sorted_score[label_idx1], :]

    # ======================<<构建5个等级>>======================= #
    sort_cha_list = [None] * 5
    cha_lable_list = [None] * 5
    for i in range(5):
        if i == 0:
            level1_num = int(np.ceil(len(sorted_score) * 0.2))
            sort_cha_list[i] = total_cha_bits[0:level1_num, :]
            cha_lable_list[i] = sorted_total_cha_bits_label[0:level1_num, :]

        elif i == 1:
            level2_num = int(np.ceil(len(sorted_score) * 0.2))
            sort_cha_list[i] = total_cha_bits[level2_num:level2_num * 2, :]
            cha_lable_list[i] = sorted_total_cha_bits_label[level2_num:level2_num * 2, :]

        elif i == 2:
            level3_num = int(np.ceil(len(sorted_score) * 0.2))
            sort_cha_list[i] = total_cha_bits[level3_num * 2:level3_num * 3, :]
            cha_lable_list[i] = sorted_total_cha_bits_label[level3_num * 2:level3_num * 3, :]

        elif i == 3:
            level4_num = int(np.ceil(len(sorted_score) * 0.2))
            sort_cha_list[i] = total_cha_bits[level4_num * 3:level4_num * 4, :]
            cha_lable_list[i] = sorted_total_cha_bits_label[level4_num * 3:level4_num * 4, :]

        elif i == 4:
            level5_num = int(len(sorted_score) - np.ceil(len(sorted_score) * 0.2) * 4)
            sort_cha_list[i] = total_cha_bits[-1 * level5_num:, :]
            cha_lable_list[i] = sorted_total_cha_bits_label[-1 * level5_num:, :]

    # **********************<<按照等差数列计算扩展率>>************************
    x = symbols('x')
    str1 = str(sort_cha_list[0].size) + ' * x + ' + str(sort_cha_list[1].size) + ' * (x + ' + str(
        diff_val) + ') + ' + str(
        sort_cha_list[2].size) + ' * (x +' + str(diff_val) + '* 2) + ' + str(sort_cha_list[3].size) + ' * (x + ' + str(
        diff_val) + ' * 3) + ' + str(sort_cha_list[4].size) + ' * (x + ' + str(diff_val) + ' * 4) - ' + str(
        rest_capacity)
    print('求解由等差数组计算得来的扩展比的方程:', str1)
    Ex_o_min = solve(str1, x)
    print(Ex_o_min)
    Ex_o_min = np.array(Ex_o_min)

    # **********************<<提取并恢复参考位的顺序>>************************
    Ex_o_rate = np.zeros(len(sort_cha_list))
    total_ref_num = 0
    ref_per_layer = np.zeros(len(sort_cha_list), dtype=int)
    group_num_list = np.zeros(len(sort_cha_list), dtype=int)
    for i in range(len(sort_cha_list)):
        Ex_o_rate[i] = Ex_o_min + i * diff_val
        print('The index of the layer:{0}, the corresponding expansion is:{1}'.format(i, Ex_o_min + i * diff_val))
        real_group_num = int(np.floor(len(sort_cha_list[i].flatten()) / group_num) - 1)
        print('第i层:{0}, 特征比特对应的分组数目:{1}'.format(i, real_group_num + 1))
        group_num_list[i] = real_group_num + 1

        for j in range(real_group_num + 1):
            if j < real_group_num:
                tem_ref_num = int(np.floor(group_num * Ex_o_rate[i]))  # 对应每组生成的参考比特个数
            else:
                tem_ref_num = int(np.floor((sort_cha_list[i].size - group_num * real_group_num) * Ex_o_rate[i]))
            total_ref_num = total_ref_num + tem_ref_num
        ref_per_layer[i] = total_ref_num

    print('\n')
    Total_ref_bits = Ext_ref_bits_total[0:total_ref_num]  # 提取真实生成ref
    # *****************<<恢复原始生成的参考比特顺序>>*****************
    Total_ref_bits_inv = np.zeros(total_ref_num, dtype=int)
    np.random.seed(319)
    RP_ref = np.random.permutation(total_ref_num)
    for i in range(total_ref_num):
        Total_ref_bits_inv[RP_ref[i]] = Total_ref_bits[i]  # 恢复原始生成的参考比特顺序

    # ==================================<<分组进行内容恢复>>===================================
    print(
        '# ==========================================<<The process of recovering>>================================= #')
    ref_index = 0
    recovery_cha = [None] * 5
    time_start = time.perf_counter()
    for i in range(len(sort_cha_list)):
        Ex_tem = Ex_o_rate[i]
        time_start_level = time.perf_counter()
        cha_bits_per_layer = sort_cha_list[i].flatten()
        cha_bits_label_per_layer = cha_lable_list[i].flatten()
        load_path = npy_load_path + str(i)
        ori_cha_bits_per_layer = np.load('{}.npy'.format(load_path))
        ori_cha_bits_per_layer = ori_cha_bits_per_layer.flatten()
        differ_tampered_cha_bits = cha_bits_per_layer - ori_cha_bits_per_layer
        print('The index of level:{0}; the number of tampered cha_bits:{1}'.format(i, np.sum(
            differ_tampered_cha_bits != 0)))

        group_num_tem = group_num_list[i]
        cha_per_group_list = [group_num, int(len(sort_cha_list[i].flatten()) - (group_num_tem - 1) * group_num)]
        ref_per_group_list = [int(np.floor(group_num * Ex_tem)),
                              int(np.floor(
                                  (len(sort_cha_list[i].flatten()) - (group_num_tem - 1) * group_num) * Ex_tem))]
        ref_bits_per_layer = Total_ref_bits_inv[ref_index:ref_per_layer[i]]
        ref_index = ref_per_layer[i]

        # ********************************<<对特征比特位进行置乱>>*******************************
        cha_bits_ran = np.zeros(len(cha_bits_per_layer), dtype=int)
        cha_bits_ran_label = np.zeros(len(cha_bits_per_layer), dtype=int)
        tem_seed = i
        np.random.seed(tem_seed)
        RP_cha = np.random.permutation(len(cha_bits_per_layer))
        for j in range(len(cha_bits_per_layer)):
            cha_bits_ran[j] = cha_bits_per_layer[RP_cha[j]]
            cha_bits_ran_label[j] = cha_bits_label_per_layer[RP_cha[j]]

        # ===================<<进行分组恢复>>=================== #
        NS = 0
        for group_index in range(group_num_tem):
            if group_index % 10000 == 0:
                print('The idx of level:{0}, The idx of group:{1}'.format(i, group_index))
            if group_index < group_num_tem - 1:
                cha_per_group_num = cha_per_group_list[0]
                ref_per_group_num = ref_per_group_list[0]
                cha_per_group = cha_bits_ran[group_index * cha_per_group_num:(group_index + 1) * cha_per_group_num]
                cha_per_group_label = cha_bits_ran_label[
                                      group_index * cha_per_group_num:(group_index + 1) * cha_per_group_num]
                ref_per_group = ref_bits_per_layer[
                                group_index * ref_per_group_num:(group_index + 1) * ref_per_group_num]
            else:
                cha_per_group_num = cha_per_group_list[1]
                ref_per_group_num = ref_per_group_list[1]
                cha_per_group = cha_bits_ran[-1 * cha_per_group_num:]
                cha_per_group_label = cha_bits_ran_label[-1 * cha_per_group_num:]
                ref_per_group = ref_bits_per_layer[-1 * ref_per_group_num:]

            np.random.seed(group_index)  # 随机种子 密钥
            A_ref = np.random.randint(0, 2, (cha_per_group_num, ref_per_group_num))
            A_ref = np.transpose(A_ref)
            label, lth, y = findbits(cha_per_group, cha_per_group_label, ref_per_group, A_ref)

            if label == 0:
                NS = NS + 1
            else:
                if lth > 0:
                    tempfind = np.array(np.nonzero(cha_per_group_label)).flatten()
                    for kkk in range(lth):
                        cha_per_group[tempfind[kkk]] = y[kkk]
                    if group_index < group_num_tem - 1:
                        cha_bits_ran[
                        group_index * cha_per_group_num:(group_index + 1) * cha_per_group_num] = cha_per_group
                    else:
                        cha_bits_ran[-1 * cha_per_group_num:] = cha_per_group

        cha_bits_inv = np.zeros(len(cha_bits_per_layer), dtype=int)
        tem_seed = i
        np.random.seed(tem_seed)
        RP_cha_inv = np.random.permutation(len(cha_bits_per_layer))
        for jj in range(len(cha_bits_per_layer)):
            cha_bits_inv[RP_cha_inv[jj]] = cha_bits_ran[jj]

        # ====================================<<评估每一层恢复情况>>=====================================
        diff_cha_after_rec_inv = ori_cha_bits_per_layer - cha_bits_inv
        print(
            '# =====================<<The bits of unrecoverable>>===================== #')
        print('The index of level:{0}; the unrecoverable number of the cha_bits:{1}'.format(i, np.sum(
            diff_cha_after_rec_inv != 0)))
        # print('the pos of unrecoverable bits:', np.argwhere(diff_cha_after_rec_inv != 0).flatten())
        time_end_level = time.perf_counter()  # 记录结束时间
        time_sum_level = time_end_level - time_start_level  # 计算的时间差为程序的执行时间，单位为秒/s
        print('The time of each level cha_bits to recover (s):', time_sum_level)
        print('# ===================================================================== #')
        print('\n')

        recovery_cha_bits_tem = cha_bits_inv.reshape(sort_cha_list[i].shape[0], sort_cha_list[i].shape[1])
        np.random.seed(1226)
        LSB_bits = np.random.randint(0, 2, (sort_cha_list[i].shape[0], 32 - cha_per_bits))
        # LSB_bits = np.zeros((sort_cha_list[i].shape[0], 32 - cha_per_bits), dtype=int)
        recovered_bits_tem = np.hstack([recovery_cha_bits_tem, LSB_bits])
        recovery_cha[i] = recovered_bits_tem

    time_end = time.perf_counter()  # 记录结束时间
    time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
    print('The time consumed by the entire recovery process (s):', time_sum)
    recovered_weight = []
    for r_idx in range(len(recovery_cha)):
        tem1 = recovery_cha[r_idx]
        for f1 in tem1:
            tem_str1 = "".join([str(x) for x in f1])  # join 拼接字符 “ ” 里面是相隔的字符 任意设置 rest_weight_float
            tem_binary1 = Binary32Tofloat(tem_str1)
            recovered_weight.append(tem_binary1)

    if len(rest_weight) == 0:
        last_recovered_weight = recovered_weight
    else:
        last_recovered_weight = recovered_weight + rest_weight

    # 将重要性排序逆向回原来顺序
    recovered_weights_float = np.zeros(len(last_recovered_weight), dtype=np.float32)
    for sort_idx in range(len(recovered_weights_float)):
        recovered_weights_float[sorted_score[sort_idx]] = last_recovered_weight[sort_idx]

    return recovered_weights_float
