import time
import itertools
from IEEE_754 import *


def Self_embedding(total_weight, total_ori_weight, group_num, block_size, cha_per_bits, hash_per_bits, diff_val,
                   npy_path, sorted_score):
    # ********************************<<参数说明>>*********************** #
    # total_weight:已经排序好的所选层的所有参数
    # total_ori_weight:原始未排序的参数
    # group_num:初始给定的分组数
    # block_size:分块大小
    # cha_per_bits:每个参数的特征比特个数
    # hash_per_bits:每个块的hash比特数
    # diff_val:等差数组的公差
    # npy_path:原始特征比特，原始ref比特等的保存路径
    # sorted_score:参数的排序
    # ********************************<<参数说明>>*********************** #
    block_num = int(np.floor(len(total_weight) / block_size))
    print('The total number of block:', block_num)
    real_weight = total_weight[0:block_size * block_num]  # 用于生成参考比特的排序后的参数
    real_unsorted_weight = total_ori_weight[0:block_size * block_num]
    rest_unsorted_weight = total_ori_weight[block_size * block_num:]  # 不参参与水印嵌的参数
    # ****************<<将参数转化为32位二进制比特>>**************** #
    total_weight_bits = []
    for tem_weight in real_weight:
        tem_value = tem_weight
        total_weight_bits.append(floatToBinary32(float(tem_value)))

    real_bits = np.array(list(itertools.chain(*total_weight_bits)))
    real_bits = real_bits.astype(int)
    real_bits_32 = real_bits.reshape(len(real_weight), 32)  # [23232+327200, 32]

    total_cha_bits = real_bits_32[:, 0:cha_per_bits]  # 取所有参数的特征比特

    total_unsorted_weight_bits = []
    for tem_weight2 in real_unsorted_weight:
        tem_value2 = tem_weight2
        total_unsorted_weight_bits.append(floatToBinary32(float(tem_value2)))

    real_unsorted_bits = np.array(list(itertools.chain(*total_unsorted_weight_bits)))
    real_unsorted_bits = real_unsorted_bits.astype(int)
    real_unsorted_bits_32 = real_unsorted_bits.reshape(len(real_unsorted_weight), 32)  # [23232+327200, 32]
    total_unsorted_cha_bits = real_unsorted_bits_32[:, 0:cha_per_bits]  # 取所有参数的特征比特

    # ************************************<<进行等级排序>>************************* #
    # ************************************<<进行等级排序>>************************* #
    # 构建5个等级
    sort_cha_list = [None] * 5
    for i in range(5):
        if i == 0:
            level1_num = int(np.ceil(len(sorted_score) * 0.2))
            sort_cha_list[i] = total_cha_bits[0:level1_num, :]
            npy_path_tem = npy_path + '_ori_cha_bits_' + str(i)
            np.save('{}.npy'.format(npy_path_tem), sort_cha_list[i])

        elif i == 1:
            level2_num = int(np.ceil(len(sorted_score) * 0.2))
            sort_cha_list[i] = total_cha_bits[level2_num:level2_num * 2, :]
            npy_path_tem = npy_path + '_ori_cha_bits_' + str(i)
            np.save('{}.npy'.format(npy_path_tem), sort_cha_list[i])

        elif i == 2:
            level3_num = int(np.ceil(len(sorted_score) * 0.2))
            sort_cha_list[i] = total_cha_bits[level3_num * 2:level3_num * 3, :]
            npy_path_tem = npy_path + '_ori_cha_bits_' + str(i)
            np.save('{}.npy'.format(npy_path_tem), sort_cha_list[i])

        elif i == 3:
            level4_num = int(np.ceil(len(sorted_score) * 0.2))
            sort_cha_list[i] = total_cha_bits[level4_num * 3:level4_num * 4, :]
            npy_path_tem = npy_path + '_ori_cha_bits_' + str(i)
            np.save('{}.npy'.format(npy_path_tem), sort_cha_list[i])

        elif i == 4:
            level5_num = int(len(sorted_score) - np.ceil(len(sorted_score) * 0.2) * 4)
            sort_cha_list[i] = total_cha_bits[-1 * level5_num:, :]
            npy_path_tem = npy_path + '_ori_cha_bits_' + str(i)
            np.save('{}.npy'.format(npy_path_tem), sort_cha_list[i])

    # 计算剩余的可嵌入空间
    total_select_weight_num = len(real_weight)
    rest_capacity = total_select_weight_num * (32 - cha_per_bits) - (hash_per_bits * block_num)

    # ***********************************<<按照等差数列计算扩展率>>****************************************
    x = symbols('x')
    str1 = str(sort_cha_list[0].size) + ' * x + ' + str(sort_cha_list[1].size) + ' * (x + ' + str(
        diff_val) + ') + ' + str(
        sort_cha_list[2].size) + ' * (x +' + str(diff_val) + '* 2) + ' + str(sort_cha_list[3].size) + ' * (x + ' + str(
        diff_val) + ' * 3) + ' + str(sort_cha_list[4].size) + ' * (x + ' + str(diff_val) + ' * 4) - ' + str(
        rest_capacity)
    print('The Eq of dff:', str1)
    Ex_o_min = solve(str1, x)
    print(Ex_o_min)
    Ex_o_min = np.array(Ex_o_min)

    # =======================================<<为每层特征比特生成参考比特>>============================================= #
    print('# ==================== <<The process of ref generation>> ==================== #')
    time_start = time.perf_counter()  # 记录开始时间
    Total_ref_bits = []
    for i in range(len(sort_cha_list)):
        Ex_o = Ex_o_min + i * diff_val
        print('第i层:{0}，对应的扩展系数:{1}'.format(i, Ex_o))
        cha_bits_tem = sort_cha_list[i].flatten()  # 对应等级的所有特征比特
        # 求真实分组数,不是均匀分组，最后一组的数目会更多，因此正确的组数是real_group_num + 1
        real_group_num = int(np.floor(len(cha_bits_tem) / group_num) - 1)
        print('第i层:{0}, 特征比特对应的分组数目:{1}'.format(i, real_group_num + 1))
        cha_per_group_list = [group_num, int(len(cha_bits_tem) - real_group_num * group_num)]  # 每组的特征比特个数
        ref_per_group_list = [int(np.floor(group_num * Ex_o)),
                              int(np.floor((len(cha_bits_tem) - real_group_num * group_num) * Ex_o))]
        print('The layer index is: {0}, The group number of cha:{1}, The group number of ref:{2}'.format(i,
                                                                                                         cha_per_group_list,
                                                                                                         ref_per_group_list))

        # ==================== <<对每一层的特征比特进行置乱>> ==================== #
        cha_bits_ran = np.zeros(len(cha_bits_tem), dtype=int)
        tem_seed = i
        np.random.seed(tem_seed)
        RP_cha = np.random.permutation(len(cha_bits_tem))
        for j in range(len(cha_bits_tem)):
            cha_bits_ran[j] = cha_bits_tem[RP_cha[j]]

        # ==================== 《生成参考比特》 ==================== #
        ref_bits_tem = np.zeros(ref_per_group_list[0] * real_group_num + ref_per_group_list[1], dtype=int)  # 对应层参考位的总数
        print('\n')
        for group_index in range(real_group_num + 1):
            if group_index < real_group_num:
                cha_per_group = cha_per_group_list[0]
                ref_per_group = ref_per_group_list[0]
                tem_cha_bits_group = cha_bits_ran[group_index * cha_per_group: (group_index + 1) * cha_per_group]
                np.random.seed(group_index)  # 随机种子 密钥
                A_ref = np.random.randint(0, 2, (cha_per_group, ref_per_group))
                ref_bits_tem[group_index * ref_per_group: (group_index + 1) * ref_per_group] = np.mod(
                    np.matmul(tem_cha_bits_group, A_ref), 2)
            else:
                cha_per_group = cha_per_group_list[1]
                ref_per_group = ref_per_group_list[1]
                tem_cha_bits_group = cha_bits_ran[-1 * cha_per_group:]  # 用置乱后的cha生成ref
                np.random.seed(group_index)  # 随机种子 密钥
                A_ref = np.random.randint(0, 2, (cha_per_group, ref_per_group))
                ref_bits_tem[-1 * ref_per_group:] = np.mod(np.matmul(tem_cha_bits_group, A_ref), 2)

        Total_ref_bits.append(ref_bits_tem)

    Total_ref_bits = np.array(list(itertools.chain(*Total_ref_bits)))
    npy_path_tem = npy_path + 'total_ref_bits'
    np.save('{}.npy'.format(npy_path_tem), Total_ref_bits)
    time_end = time.perf_counter()   # 记录结束时间
    time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
    print('The time of ref generation (s):', time_sum)

    # # ***********************************<<对所有参考比特进行置乱>>************************************ # #
    # # ***********************************<<对所有参考比特进行置乱>>************************************ # #
    Total_ref_bits_ran = np.zeros(len(Total_ref_bits), dtype=int)
    np.random.seed(319)
    RP_ref = np.random.permutation(len(Total_ref_bits))
    for i_ref in range(len(Total_ref_bits)):
        Total_ref_bits_ran[i_ref] = Total_ref_bits[RP_ref[i_ref]]

    Total_ref_bits_ran = Total_ref_bits_ran.tolist()
    add_num = rest_capacity - len(Total_ref_bits_ran)
    print('The added number in block dividing:', add_num)
    if add_num == 0:
        print('Continue')
    else:
        for i in range(add_num):
            Total_ref_bits_ran.append(0)  # 当参考比特不足以分块时，通过0填充来进行补齐

    Total_ref_bits_ran = np.array(Total_ref_bits_ran)

    # # *****************************************《生成哈希比特》***************************************** # #
    print('# ================================<<The process of hash generation>>=========================== #')
    time_start = time.perf_counter()   # 记录开始时间
    cha_per_block_num = cha_per_bits * block_size  # 每个块的特征比特个数
    ref_per_block_num = int((32 - cha_per_bits) * block_size - hash_per_bits)
    weight_list_float = []
    Waterarked_bits = []
    for block_idx in range(block_num):
        # 此处注意:在生成hash和嵌入水印比特时是在《原始的参数位置》上进行的，并不是排序后的参数
        cha_per_block = total_unsorted_cha_bits[block_idx * block_size:(block_idx + 1) * block_size, :]
        ref_per_block = Total_ref_bits_ran[block_idx * ref_per_block_num:(block_idx + 1) * ref_per_block_num]
        np.random.seed(block_idx)  # 随机种子 密钥
        A_hash = np.random.randint(0, 2, (hash_per_bits, cha_per_block_num + ref_per_block_num))
        hash_bit = np.mod(np.matmul(A_hash, np.hstack([cha_per_block.flatten(), ref_per_block])), 2)
        hash_bit = hash_bit.reshape(block_size, 2)
        ref_per_block = ref_per_block.reshape(block_size, 32 - cha_per_bits - 2)
        marked_bits = np.hstack([ref_per_block, hash_bit])
        Waterarked_bits.append(marked_bits.flatten())
        block_bits = np.hstack([cha_per_block, marked_bits])
        for j in block_bits:
            tem_str = "".join([str(x).split(".")[0] for x in j])  # join 拼接字符 “ ” 里面是相隔的字符 任意设置 rest_weight_float
            tem_binary = Binary32Tofloat(tem_str)
            weight_list_float.append(tem_binary)

    npy_path_tem = npy_path + 'marked_bits'
    np.save('{}.npy'.format(npy_path_tem), Waterarked_bits)
    time_end = time.perf_counter()   # 记录结束时间
    time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
    print('The time of  hash generation and watermark embedding (s):', time_sum)

    # 将未使用的参数添加回原位置
    if len(rest_unsorted_weight) == 0:
        weight_tem = weight_list_float
    else:
        weight_tem = weight_list_float + rest_unsorted_weight

    last_marked_weight = weight_tem
    return last_marked_weight
