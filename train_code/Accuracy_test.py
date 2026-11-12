import os
import torch
import torch.nn as nn
from IEEE_754 import *
from random import *
import matplotlib.pyplot as plt  # plt 用于显示图片
import numpy as np
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import FashionMNIST
import torch.utils.data as Data
from model_structure import *
from model_structure import vit_base_patch16_224_in21k as vit_model

# 设备设置
# torch.cuda.set_device(0)  # 这句用来设置pytorch在哪块GPU上运行，pytorch-cpu版本不需要运行这句
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def acc_test(test_loader, device, model):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            model = model.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            acc = 100 * correct / total
        print('Test Accuracy of the model on the test images: {} %'.format(acc))
    return acc


# ***************************************<<数据预处理>>************************************* #
# 数据增广方法
batch_size = 64
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
test_dataset = torchvision.datasets.CIFAR10(root=r'D:\HL\project\dataset\cifar-10\data', train=False,
                                            transform=transform)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# ****************************************************<<模型选择>>********************************************* #
num_classes = 10
cha_per_bits = 14
diff_val = 0.3
tamper_rate = 0
model_idx = 2  # 1表示alexnet，2表示vgg16，3表示resnet18，4表示mobilenet
acc_test_idx = 3  # 0表示原始模型，1表示水印模型，2表示篡改模型，3表示恢复模型
group_num = 300
print('The para seting <<model_idx:{}, group_num:{}, cha_bits:{}, diff_val:{}, tamper_rate:{}{}'.format(model_idx,
                                                                                                        group_num,
                                                                                                        cha_per_bits,
                                                                                                        diff_val,
                                                                                                        tamper_rate,
                                                                                                        '>>'))
attack_times = 10  # 多次实验次数, 注意原始模型和水印模型没有多次实验，因为嵌入方式固定只要参数不变模型性能不变，同样的对于原始模型其参数固定性能也不会发生变化
acc_tem = 0

for t_idx in range(attack_times):
    if model_idx == 1:
        model = AlexNet()
        num_fc = model.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
        model.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
        if acc_test_idx == 0:
            model_path = './test_model/alexnet_model_9'
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))
        elif acc_test_idx == 1:
            marked_model_path = './Marked_model/alexnet_model' + '_' + str(cha_per_bits) + '_' + str(diff_val) + '_' + str(group_num)
            model.load_state_dict(torch.load('{}.pth'.format(marked_model_path)))
        elif acc_test_idx == 2:
            # tampered_model_path = './Tampered_model/alexnet/model' + '_' + str(cha_per_bits) + '_' + str(
            #     diff_val) + '_' + str(group_num) + '_' + str(
            #     tamper_rate) + '_' + str(t_idx)
            tampered_model_path = './tamper_mask/alexnet/mask' + '_' + str(group_num) + '_' + str(
                        tamper_rate) + '' + 'flip' + '_' + str(t_idx)
            model.load_state_dict(torch.load('{}.pth'.format(tampered_model_path)))
        elif acc_test_idx == 3:
            recovered_model_path = './Recovered_model/alexnet/recovered' + '_' + str(cha_per_bits) + '_' + str(
                diff_val) + '_' + str(group_num) + '_' + str(tamper_rate) + '_' + str(t_idx)
            # recovered_model_path = './Recovered_model/alexnet/recovered' + '_' + 'test' + '_' + str(t_idx)
            model.load_state_dict(torch.load('{}.pth'.format(recovered_model_path)))
        else:
            print('The idx is error...')
        print('The test model is alexnet model')

    elif model_idx == 2:
        model = Vgg16()
        num_fc = model.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
        model.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
        if acc_test_idx == 0:
            model_path = './test_model/vgg16_model_9'
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))
        elif acc_test_idx == 1:
            marked_model_path = './Marked_model/vgg16_model' + '_' + str(cha_per_bits) + '_' + str(diff_val) + '_' + str(group_num)
            model.load_state_dict(torch.load('{}.pth'.format(marked_model_path)))
        elif acc_test_idx == 2:
            # tampered_model_path = './Tampered_model/vgg16/model' + '_' + str(cha_per_bits) + '_' + str(
            #     diff_val) + '_' + str(group_num) + '_' + str(
            #     tamper_rate) + '_' + str(t_idx)
            tampered_model_path = './tamper_mask/vgg16/mask' + '_' + str(group_num) + '_' + str(
                tamper_rate) + '' + 'flip' + '_' + str(t_idx)
            model.load_state_dict(torch.load('{}.pth'.format(tampered_model_path)))
        elif acc_test_idx == 3:
            recovereded_model_path = './Recovered_model/vgg16/recovered' + '_' + str(cha_per_bits) + '_' + str(
                diff_val) + '_' + str(tamper_rate) + '_' + str(t_idx)
            model.load_state_dict(torch.load('{}.pth'.format(recovereded_model_path)))
        else:
            print('The idx is error...')
        print('The test model is vgg16 model')

    elif model_idx == 3:
        model = resnet18()
        num_fc = model.fc.in_features  # 获取最后一个全连接层的输入是多少
        model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
        if acc_test_idx == 0:
            model_path = './test_model/resnet18_model_9'
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))
        elif acc_test_idx == 1:
            marked_model_path = './Marked_model/resnet18' + '_' + str(cha_per_bits) + '_' + str(diff_val) + '_' + str(group_num)
            model.load_state_dict(torch.load('{}.pth'.format(marked_model_path)))
        elif acc_test_idx == 2:
            # tampered_model_path = './Tampered_model/resnet18/model' + '_' + str(cha_per_bits) + '_' + str(
            #     diff_val) + '_' + str(group_num) + '_' + str(
            #     tamper_rate) + '_' + str(t_idx)
            tampered_model_path = './tamper_mask/resnet18/mask' + '_' + str(group_num) + '_' + str(
                tamper_rate) + '' + 'flip' + '_' + str(t_idx)
            model.load_state_dict(torch.load('{}.pth'.format(tampered_model_path)))
        elif acc_test_idx == 3:
            recovereded_model_path = './Recovered_model/resnet18/recovered' + '_' + str(cha_per_bits) + '_' + str(
                diff_val) + '_' + str(group_num) + '_' + str(tamper_rate) + '_' + str(t_idx)
            model.load_state_dict(torch.load('{}.pth'.format(recovereded_model_path)))
        else:
            print('The idx is error...')
        print('The test model is resnet18 model')

    elif model_idx == 4:
        model = mobilenet_v2()
        num_fc = model.classifier[1].in_features  # 获取最后一个全连接层的输入是多少
        model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
        if acc_test_idx == 0:
            model_path = './test_model/mobilenet_model_9'
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))
        elif acc_test_idx == 1:
            marked_model_path = './Marked_model/mobilenet' + '_' + str(cha_per_bits) + '_' + str(diff_val) + '_' + str(group_num)
            model.load_state_dict(torch.load('{}.pth'.format(marked_model_path)))
        elif acc_test_idx == 2:
            # tampered_model_path = './Tampered_model/mobilenet/model' + '_' + str(cha_per_bits) + '_' + str(
            #     diff_val) + '_' + str(group_num) + '_' + str(
            #     tamper_rate) + '_' + str(t_idx)
            tampered_model_path = './tamper_mask/mobilenet/mask' + '_' + str(group_num) + '_' + str(
                tamper_rate) + '' + 'flip' + '_' + str(t_idx)
            model.load_state_dict(torch.load('{}.pth'.format(tampered_model_path)))
        elif acc_test_idx == 3:
            recovereded_model_path = './Recovered_model/mobilenet/recovered' + '_' + str(cha_per_bits) + '_' + str(
                diff_val) + '_' + str(group_num) + '_' + str(tamper_rate) + '_' + str(t_idx)
            model.load_state_dict(torch.load('{}.pth'.format(recovereded_model_path)))
        else:
            print('The idx is error...')
        print('The test model is mobilenet model')

    elif model_idx == 5:
        model = vit_model(num_classes=num_classes, has_logits=False).to(device)
        if acc_test_idx == 0:
            model_path = './test_model/vit_B16_2'
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))
        elif acc_test_idx == 1:
            marked_model_path = './Marked_model/vit' + '_' + str(cha_per_bits) + '_' + str(diff_val) + '_' + str(group_num)
            model.load_state_dict(torch.load('{}.pth'.format(marked_model_path)))
        elif acc_test_idx == 2:
            # tampered_model_path = './Tampered_model/vit/model' + '_' + str(cha_per_bits) + '_' + str(
            #     diff_val) + '_' + str(group_num) + '_' + str(
            #     tamper_rate) + '_' + str(t_idx)
            tampered_model_path = './tamper_mask/vit/mask' + '_' + str(group_num) + '_' + str(
                tamper_rate) + '' + 'flip' + '_' + str(t_idx)
            model.load_state_dict(torch.load('{}.pth'.format(tampered_model_path)))
        elif acc_test_idx == 3:
            recovereded_model_path = './Recovered_model/vit/recovered' + '_' + str(cha_per_bits) + '_' + str(
                diff_val) + '_' + str(group_num) + '_' + str(tamper_rate) + '_' + str(t_idx)
            model.load_state_dict(torch.load('{}.pth'.format(recovereded_model_path)))
        else:
            print('The idx is error...')
        print('The test model is vit model')

    else:
        print('The model_idx is error...')

    acc = acc_test(test_loader, device, model)
    acc_tem = acc_tem + acc

print('The average accuracy:', acc_tem/attack_times)

