from experiment_config import CIFAR10, FASHION_MNIST, TINY_IMAGENET, evaluation_dataset
import os
import math
import time
import timm
import torch
import itertools
import warnings
import torchvision
import numpy as np
import os.path as osp
import torch.nn as nn
from IEEE_754 import *
from self_embedding_new import *
import torchvision.models as models
from torchvision import transforms, datasets
from Tinyimagenet_dataset import TinyImageNet

# ======================================<<环境设备设置>>====================================== #
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
warnings.filterwarnings("ignore")

# ================================ <<模型选择>> ================================ #
class Log:
    def __init__(self, log_path):
        self.log_path = log_path

    def __call__(self, msg):
        print(msg, end='\n')
        with open(self.log_path, 'a') as f:
            f.write(msg)


def model_load(dataset_idx, model_idx, batch_size):
    tem_path = './Marked_model/'
    if not os.path.exists(tem_path):
        os.makedirs(tem_path)

    if dataset_idx == 1:
        print('Dataset is FashionMNIST')
        num_classes = 10
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        train_dataset = torchvision.datasets.FashionMNIST(root=FASHION_MNIST,
                                                          train=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        test_dataset = torchvision.datasets.FashionMNIST(root=FASHION_MNIST,
                                                         train=False, transform=transform)
        test_loader = torch.utils.data.DataLoader(dataset=evaluation_dataset(test_dataset), batch_size=batch_size, shuffle=False)
        # ============================================<<加载模型>>============================================ #
        if model_idx == 1:
            model = models.alexnet(pretrained=False)
            model.features._modules['0'] = nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2)  # 对应alexnet
            num_fc = model.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = './test_model/alexnet_fash'
            model.load_state_dict(torch.load('{}.pth'.format(model_path), map_location="cpu", weights_only=True))
            save_path = './Marked_model/alexnet_fash'
            sorted_path = './importance_score/alexnet_fash'  # 参数重要性排序文件路径
            layer_name = ['features.0.weight', 'features.3.weight']

        elif model_idx == 2:
            model = models.vgg16(pretrained=False)
            model.features._modules['0'] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)  # 对应Vgg16
            num_fc = model.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = './test_model/vgg16_fash'
            model.load_state_dict(torch.load('{}.pth'.format(model_path), map_location="cpu", weights_only=True))
            save_path = './Marked_model/vgg16_fash'
            sorted_path = './importance_score/vgg16_fash'  # 参数重要性排序文件路径
            layer_name = ['features.0.weight', 'features.5.weight']

        elif model_idx == 3:
            model = models.resnet18(pretrained=False)
            model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            num_fc = model.fc.in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = './test_model/resnet18_fash'
            model.load_state_dict(torch.load('{}.pth'.format(model_path), map_location="cpu", weights_only=True))
            save_path = './Marked_model/resnet18_fash'
            sorted_path = './importance_score/resnet18_fash'  # 参数重要性排序文件路径
            layer_name = ['conv1.weight', 'layer1.0.conv1.weight']

        elif model_idx == 4:
            model = models.mobilenet_v2(pretrained=False)
            model.features._modules['0'][0] = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=3, bias=False)
            num_fc = model.classifier[1].in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = './test_model/mobilenet_fash'
            model.load_state_dict(torch.load('{}.pth'.format(model_path), map_location="cpu", weights_only=True))
            save_path = './Marked_model/mobilenet_fash'
            sorted_path = './importance_score/mobilenet_fash'  # 参数重要性排序文件路径
            layer_name = ['features.0.0.weight', 'features.1.conv.0.0.weight']

        elif model_idx == 5:
            model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes)
            model.patch_embed.proj = nn.Conv2d(1, 768, kernel_size=16, stride=16)
            model_path = './test_model/vit_fash'  # 原始模型参数路径
            model.load_state_dict(torch.load('{}.pth'.format(model_path), map_location="cpu", weights_only=True))
            save_path = './Marked_model/vit_fash'
            sorted_path = './importance_score/vit_fash'  # 参数重要性排序文件路径
            layer_name = ['cls_token', 'pos_embed']

    elif dataset_idx == 2:
        print('Dataset is Cifar-10')
        num_classes = 10
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        train_dataset = torchvision.datasets.CIFAR10(root=CIFAR10, train=True,
                                                     transform=transform)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        test_dataset = torchvision.datasets.CIFAR10(root=CIFAR10, train=False,
                                                    transform=transform)
        test_loader = torch.utils.data.DataLoader(dataset=evaluation_dataset(test_dataset), batch_size=batch_size, shuffle=False)
        # ============================================<<加载模型>>============================================ #
        if model_idx == 1:
            model = models.alexnet(pretrained=False)  # 加载官网模型结构
            num_fc = model.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = './test_model/alexnet_cifar'  # 原始模型参数路径
            model.load_state_dict(torch.load('{}.pth'.format(model_path), map_location="cpu", weights_only=True))
            save_path = './Marked_model/alexnet_cifar'
            sorted_path = './importance_score/alexnet_cifar'  # 参数重要性排序文件路径
            layer_name = ['features.0.weight', 'features.3.weight']

        elif model_idx == 2:
            model = models.vgg16(pretrained=False)
            num_fc = model.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = './test_model/vgg16_cifar'  # 原始模型参数路径
            model.load_state_dict(torch.load('{}.pth'.format(model_path), map_location="cpu", weights_only=True))
            save_path = './Marked_model/vgg16_cifar'
            sorted_path = './importance_score/vgg16_cifar'  # 参数重要性排序文件路径
            layer_name = ['features.0.weight', 'features.2.weight']

        elif model_idx == 3:
            model = models.resnet18(pretrained=False)
            num_fc = model.fc.in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = './test_model/resnet18_cifar'  # 原始模型参数路径
            model.load_state_dict(torch.load('{}.pth'.format(model_path), map_location="cpu", weights_only=True))
            save_path = './Marked_model/resnet18_cifar'
            sorted_path = './importance_score/resnet18_cifar'  # 参数重要性排序文件路径
            layer_name = ['conv1.weight', 'layer1.0.conv1.weight']

        elif model_idx == 4:
            model = models.mobilenet_v2(pretrained=False)
            num_fc = model.classifier[1].in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = './test_model/mobilenet_cifar'  # 原始模型参数路径
            model.load_state_dict(torch.load('{}.pth'.format(model_path), map_location="cpu", weights_only=True))
            save_path = './Marked_model/mobilenet_cifar'
            sorted_path = './importance_score/mobilenet_cifar'  # 参数重要性排序文件路径
            layer_name = ['features.0.0.weight', 'features.1.conv.0.0.weight']

        elif model_idx == 5:
            model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes)
            model_path = './test_model/vit_cifar'  # 原始模型参数路径
            model.load_state_dict(torch.load('{}.pth'.format(model_path), map_location="cpu", weights_only=True))
            save_path = './Marked_model/vit_cifar'
            sorted_path = './importance_score/vit_cifar'  # 参数重要性排序文件路径
            layer_name = ['cls_token', 'pos_embed']

    else:
        print('Dataset is Tiny ImageNet')
        num_classes = 200
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        data_dir = TINY_IMAGENET
        train_dataset = TinyImageNet(data_dir, train=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        test_dataset = TinyImageNet(data_dir, train=False, transform=transform)
        test_loader = torch.utils.data.DataLoader(dataset=evaluation_dataset(test_dataset), batch_size=batch_size, shuffle=False)
        # ============================================<<加载模型>>============================================ #
        if model_idx == 1:
            model = models.alexnet(pretrained=False)
            num_fc = model.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = './test_model/alexnet_tiny'  # 原始模型参数路径
            model.load_state_dict(torch.load('{}.pth'.format(model_path), map_location="cpu", weights_only=True))
            save_path = './Marked_model/alexnet_tiny'
            sorted_path = './importance_score/alexnet_tiny'  # 参数重要性排序文件路径
            layer_name = ['features.0.weight', 'features.3.weight']

        elif model_idx == 2:
            model = models.vgg16(pretrained=False)
            num_fc = model.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = './test_model/vgg16_tiny'  # 原始模型参数路径
            model.load_state_dict(torch.load('{}.pth'.format(model_path), map_location="cpu", weights_only=True))
            save_path = './Marked_model/vgg16_tiny'
            sorted_path = './importance_score/vgg16_tiny'  # 参数重要性排序文件路径
            layer_name = ['features.0.weight', 'features.5.weight']

        elif model_idx == 3:
            model = models.resnet18(pretrained=False)
            num_fc = model.fc.in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = './test_model/resnet18_tiny'  # 原始模型参数路径
            model.load_state_dict(torch.load('{}.pth'.format(model_path), map_location="cpu", weights_only=True))
            save_path = './Marked_model/resnet18_tiny'
            sorted_path = './importance_score/resnet18_tiny'  # 参数重要性排序文件路径
            layer_name = ['conv1.weight', 'layer1.0.conv1.weight']

        elif model_idx == 4:
            model = models.mobilenet_v2(pretrained=False)
            num_fc = model.classifier[1].in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = './test_model/mobilenet_tiny'  # 原始模型参数路径
            model.load_state_dict(torch.load('{}.pth'.format(model_path), map_location="cpu", weights_only=True))
            save_path = './Marked_model/mobilenet_tiny'
            sorted_path = './importance_score/mobilenet_tiny'  # 参数重要性排序文件路径
            layer_name = ['features.0.0.weight', 'features.1.conv.0.0.weight']

        elif model_idx == 5:
            model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes)
            model_path = './test_model/vit_tiny'  # 原始模型参数路径
            model.load_state_dict(torch.load('{}.pth'.format(model_path), map_location="cpu", weights_only=True))
            save_path = './Marked_model/vit_tiny'
            sorted_path = './importance_score/vit_tiny'  # 参数重要性排序文件路径
            layer_name = ['cls_token', 'pos_embed']

    return model, train_loader, test_loader, layer_name, save_path, sorted_path


def para_load(model, layer_name):
    # ================================ <<读取模型参数>> ================================ #
    ori_weight = []
    weight_shape_list = []
    weight_num_list = []
    sorted_idx = 0
    for layer_idx in range(len(layer_name)):
        current_layer_name = layer_name[layer_idx]
        for name, parameters in model.named_parameters():
            # print(name, ':', parameters.size())
            if current_layer_name in name:  # 如何区分features.21.weight和features.2.weight
                # print('The layer name:', name)
                parameters = parameters.data.cpu()
                weight_parameters = parameters.detach().numpy()
                weight_shape = weight_parameters.shape
                weight_shape_list.append(weight_shape)
                weight_parameters = weight_parameters.flatten()
                sorted_idx = sorted_idx + 1
                weight_num_list.append(len(weight_parameters))
                ori_weight.append(weight_parameters)
                break

    return ori_weight, weight_num_list, weight_shape_list


def acc_test(test_loader, model):
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
    # print('Test Accuracy of the model on the test images: {} %'.format(acc))
    return acc

# =======================<<相关参数设置>>======================= #
batch_size = 64
dataset_idx_list = [1]  # 1:FashionMNIST; 2:cifar-10; 3:Tiny_ImageNet
model_idx_list = [3]  # 1:alexnet; 2:vgg16; 3:resnet18; 4:Mobilenet_V2
group_num = 300  # 分组数初始值
block_size = 64  # 分块的数量
cha_per_bits = 12  # MSB比特个数
# ===========================================<<cha_per_bits>>============================================ #
# ========<<Alexnet-Fash:   12(92.17); Alexnet-Cifar10:   12(92.29); Alexnet-Tiny:   13(57.60)>> ======== #
# ========<<Vgg16-Fash:     13(92.71); Vgg16-Cifar10:     13(93.02); Vgg16-Tiny:     13(61.77)>> ======== #
# ========<<Resnet18-Fash:  12(93.41); Resnet18-Cifar10:  13(94.52); Resnet18-Tiny:  14(66.34)>> ======== #
# ========<<Mobilenet-Fash: 14(92.16); Mobilenet-Cifar10: 16(93.70); Mobilenet-Tiny: 14(58.01)>> ======== #
# ========<<VIT-Fash:       14(96.63); VIT-Cifar10:       12(96.97); VIT-Tiny:       12(80.40)>> ======== #
# ===========================================<<cha_per_bits>>============================================ #
hash_per_bits = 128  # 每个块生成的hash比特个数
diff_val = 0.3  # 扩展比的公差
log = Log(osp.join('./Marked_model/', 'M_acc.txt'))

for dataset_idx in dataset_idx_list:
    for model_idx in model_idx_list:
        print('The para seting <<model_idx:{}, group_num:{}, cha_bits:{}, diff_val:{}{}'.format(model_idx, group_num,
                                                                                                cha_per_bits, diff_val,
                                                                                                '>>'))

        model, train_loader, test_loader, layer_name, save_path, sorted_path = model_load(dataset_idx, model_idx,
                                                                                          batch_size)

        # acc = acc_test(test_loader, model)
        # print('The original model accuracy:==========================>>', acc)

        model.eval()
        time_start = time.perf_counter()  # 记录开始时间
        ori_weight, weight_num_list, weight_shape_list = para_load(model, layer_name)
        total_ori_weight = np.array(list(itertools.chain(*ori_weight)))
        sorted_score = np.load('{}.npy'.format(sorted_path))
        soeted_weight = np.zeros(len(total_ori_weight), dtype=np.float32)

        for sort_idx in range(len(sorted_score)):
            soeted_weight[sort_idx] = total_ori_weight[sorted_score[sort_idx]]

        # total_sorted_bits = list(itertools.chain(*sorted_weight_binary_list))
        # # 测试观察排序前后的呼是否出错
        # weight_list_float_tem = sorted_weight[0]
        # Isort_weight_list2_float = np.zeros(len(sorted_weight[0]), dtype=np.float32)
        # for sort_idx2 in range(len(weight_list_float_tem)):
        #     Isort_weight_list2_float[sorted_score1[sort_idx2]] = weight_list_float_tem[sort_idx2]

        # print('The diff of two kinds weight:', np.sum((Isort_weight_list2_float - ori_weight[0])))

        # ================================ <<自嵌入>> ================================ #
        npy_path = save_path + '_' + str(group_num)
        Marked_weight = Self_embedding(soeted_weight, total_ori_weight, group_num, block_size, cha_per_bits,
                                       hash_per_bits, diff_val, npy_path, sorted_score)

        M1 = np.array(Marked_weight[0:weight_num_list[0]])
        diff1 = M1 - total_ori_weight[0:weight_num_list[0]]
        M2 = np.array(Marked_weight[-1 * weight_num_list[1]:])
        diff2 = M2 - total_ori_weight[-1 * weight_num_list[1]:]
        print('第1层参数水印前后差值的均值：{}， 第2层参数水印前后差值的均值：{}'.format(np.mean(diff1), np.mean(diff2)))
        print('第1层参数水印前后差值的方差：{}， 第2层参数水印前后差值的方差：{}'.format(np.var(diff1), np.var(diff2)))

        # ================================== <<重新加载水印参数>> ================================== #
        for layer_idx in range(len(layer_name)):
            if layer_idx == 0:
                Marked_weight_tem = M1
            else:
                Marked_weight_tem = M2

            layer_name_tem = layer_name[layer_idx]
            weight_shape_tem = weight_shape_list[layer_idx]
            if len(weight_shape_tem) == 2:
                Marked_weight_1 = Marked_weight_tem.reshape(weight_shape_tem[0], weight_shape_tem[1])

            elif len(weight_shape_tem) == 3:
                Marked_weight_1 = Marked_weight_tem.reshape(weight_shape_tem[0], weight_shape_tem[1],
                                                            weight_shape_tem[2])

            elif len(weight_shape_tem) == 4:
                Marked_weight_1 = Marked_weight_tem.reshape(weight_shape_tem[0], weight_shape_tem[1],
                                                            weight_shape_tem[2],
                                                            weight_shape_tem[3])

            Marked_weight_tensor = torch.from_numpy(Marked_weight_1)
            model_dict = model.state_dict()
            model_dict[layer_name_tem] = Marked_weight_tensor
            model.load_state_dict(model_dict)

        torch.save(model.state_dict(), '{}.pth'.format(save_path))
        time_end = time.perf_counter()  # 记录结束时间
        time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
        print('The time of code running (s):', time_sum)
        acc = acc_test(test_loader, model)
        msg = f'dataset index: {dataset_idx}, | model index: {model_idx}, | Marked acc: {acc}\n'
        log(msg)
