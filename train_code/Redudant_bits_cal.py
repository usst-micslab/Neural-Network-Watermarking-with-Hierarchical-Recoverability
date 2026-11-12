import os
import copy
import timm
import torch
import warnings
import itertools
import numpy as np
import torchvision
from random import *
import torch.nn as nn
import os.path as osp
from IEEE_754 import *
import torchvision.models as models
import torchvision.transforms as transforms
from Tinyimagenet_dataset import TinyImageNet

# ======================================<<环境设备设置>>====================================== #
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
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
        train_dataset = torchvision.datasets.FashionMNIST(root=r'D:\HL\project\dataset\FashionMNIST\data\FashionMNIST',
                                                          train=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        test_dataset = torchvision.datasets.FashionMNIST(root=r'D:\HL\project\dataset\FashionMNIST\data\FashionMNIST',
                                                         train=False, transform=transform)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
        # ============================================<<加载模型>>============================================ #
        if model_idx == 1:
            model = models.alexnet(pretrained=False)
            model.features._modules['0'] = nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2)  # 对应alexnet
            num_fc = model.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = './test_model/alexnet_fash'
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))
            layer_name = ['features.0.weight', 'features.3.weight']

        elif model_idx == 2:
            model = models.vgg16(pretrained=False)
            model.features._modules['0'] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)  # 对应Vgg16
            num_fc = model.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = './test_model/vgg16_fash'
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))
            layer_name = ['features.0.weight', 'features.5.weight']

        elif model_idx == 3:
            model = models.resnet18(pretrained=False)
            model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            num_fc = model.fc.in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = './test_model/resnet18_fash'
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))
            layer_name = ['conv1.weight', 'layer1.0.conv1.weight']

        elif model_idx == 4:
            model = models.mobilenet_v2(pretrained=False)
            model.features._modules['0'][0] = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=3, bias=False)
            num_fc = model.classifier[1].in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = './test_model/mobilenet_fash'
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))
            layer_name = ['features.0.0.weight', 'features.1.conv.0.0.weight']

        elif model_idx == 5:
            model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes)
            model.patch_embed.proj = nn.Conv2d(1, 768, kernel_size=16, stride=16)
            model_path = './test_model/vit_fash'  # 原始模型参数路径
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))
            layer_name = ['cls_token', 'pos_embed']

    elif dataset_idx == 2:
        print('Dataset is Cifar-10')
        num_classes = 10
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        train_dataset = torchvision.datasets.CIFAR10(root=r'D:\HL\project\dataset\cifar-10\data/', train=True,
                                                     transform=transform)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        test_dataset = torchvision.datasets.CIFAR10(root=r'D:\HL\project\dataset\cifar-10\data/', train=False,
                                                    transform=transform)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
        # ============================================<<加载模型>>============================================ #
        if model_idx == 1:
            model = models.alexnet(pretrained=False)  # 加载官网模型结构
            num_fc = model.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = './test_model/alexnet_cifar'  # 原始模型参数路径
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))
            layer_name = ['features.0.weight', 'features.3.weight']

        elif model_idx == 2:
            model = models.vgg16(pretrained=False)
            num_fc = model.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = './test_model/vgg16_cifar'  # 原始模型参数路径
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))
            layer_name = ['features.0.weight', 'features.2.weight']

        elif model_idx == 3:
            model = models.resnet18(pretrained=False)
            num_fc = model.fc.in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = './test_model/resnet18_cifar'  # 原始模型参数路径
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))
            layer_name = ['conv1.weight', 'layer1.0.conv1.weight']

        elif model_idx == 4:
            model = models.mobilenet_v2(pretrained=False)
            num_fc = model.classifier[1].in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = './test_model/mobilenet_cifar'  # 原始模型参数路径
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))
            layer_name = ['features.0.0.weight', 'features.1.conv.0.0.weight']

        elif model_idx == 5:
            model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes)
            model_path = './test_model/vit_cifar'  # 原始模型参数路径
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))
            layer_name = ['cls_token', 'pos_embed']

    else:
        print('Dataset is Tiny ImageNet')
        num_classes = 200
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        data_dir = r'D:\HL\project\dataset\tiny-imagenet-200'
        train_dataset = TinyImageNet(data_dir, train=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        test_dataset = TinyImageNet(data_dir, train=False, transform=transform)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
        # ============================================<<加载模型>>============================================ #
        if model_idx == 1:
            model = models.alexnet(pretrained=False)
            num_fc = model.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = './test_model/alexnet_tiny'  # 原始模型参数路径
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))
            layer_name = ['features.0.weight', 'features.3.weight']

        elif model_idx == 2:
            model = models.vgg16(pretrained=False)
            num_fc = model.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = './test_model/vgg16_tiny'  # 原始模型参数路径
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))
            layer_name = ['features.0.weight', 'features.5.weight']

        elif model_idx == 3:
            model = models.resnet18(pretrained=False)
            num_fc = model.fc.in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = './test_model/resnet18_tiny'  # 原始模型参数路径
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))
            layer_name = ['conv1.weight', 'layer1.0.conv1.weight']

        elif model_idx == 4:
            model = models.mobilenet_v2(pretrained=False)
            num_fc = model.classifier[1].in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = './test_model/mobilenet_tiny'  # 原始模型参数路径
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))
            layer_name = ['features.0.0.weight', 'features.1.conv.0.0.weight']

        elif model_idx == 5:
            model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes)
            model_path = './test_model/vit_tiny'  # 原始模型参数路径
            model.load_state_dict(torch.load('{}.pth'.format(model_path)))
            layer_name = ['cls_token', 'pos_embed']

    return model, train_loader, test_loader, layer_name, model_path


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


def bit_modify(weight_parameters, bit_idx):
    k = 0
    for para_idx in weight_parameters:
        tam_para = para_idx
        tam_bin = floatToBinary32(float(tam_para))
        tem_str = ''
        if bit_idx == 0:
            tem_binary_1 = tam_bin[0:31] + str(np.mod(randrange(999), 2))
        else:
            for i in range(bit_idx):
                tem_str = tem_str + str(np.mod(randrange(999), 2))  # 生成随机01比特
            tem_binary_1 = tam_bin[0:32 - bit_idx] + tem_str
        weight_parameters[k] = Binary32Tofloat(tem_binary_1)
        k = k + 1
    return weight_parameters


# =======================<<相关参数设置>>======================= #
batch_size = 64
dataset_idx_list = [1]  # 1:FashionMNIST; 2:cifar-10; 3:Tiny_ImageNet
model_idx_list = [3]  # 1:alexnet; 2:vgg16; 3:resnet18; 4:Mobilenet_V2
times = 5
log = Log(osp.join('./Red_bits/', 'Aacc.txt'))
for dataset_idx in dataset_idx_list:
    for model_idx in model_idx_list:
        acc_table = np.zeros([32, 1])
        table_row = 0
        for bit_idx in range(0, 32):
            acc_sum = 0
            for i in range(times):
                model, train_loader, test_loader, layer_name, model_path = model_load(dataset_idx, model_idx, batch_size)
                model.load_state_dict(torch.load('{}.pth'.format(model_path)))
                model.eval()
                ori_weight, weight_num_list, weight_shape_list = para_load(model, layer_name)
                acc_ori = acc_test(test_loader, model)
                total_ori_weight = np.array(list(itertools.chain(*ori_weight)))
                modified_weight = bit_modify(total_ori_weight, bit_idx)
                for layer_idx in range(len(layer_name)):
                    if layer_idx == 0:
                        Marked_weight_tem = modified_weight[0:weight_num_list[0]]
                    else:
                        Marked_weight_tem = modified_weight[weight_num_list[0]:]

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

                acc = acc_test(test_loader, model)
                print('The modified model acc:', acc)
                acc_sum = acc_sum + acc

            ave_acc = acc_sum / times
            acc_table[table_row, 0] = ave_acc
            table_row = table_row + 1
            msg = f'dataset index: {dataset_idx}, | model index: {model_idx}, | Ori_acc: {acc_ori}, | Modified acc: {ave_acc}\n'
            log(msg)
        # np.save('{}.npy'.format('mobilenet_fash'), acc_table)



