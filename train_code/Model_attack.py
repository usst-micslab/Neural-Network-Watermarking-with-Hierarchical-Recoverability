from experiment_config import CIFAR10, FASHION_MNIST, TINY_IMAGENET, evaluation_dataset, positive_env
import os
import math
import timm
import torch
import itertools
import warnings
import torchvision
import numpy as np
import os.path as osp
import torch.nn as nn
from acttack_method import *
import torchvision.models as models
from torchvision import transforms, datasets
from Tinyimagenet_dataset import TinyImageNet

# ======================================<<环境设备设置>>====================================== #
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
warnings.filterwarnings("ignore")


# =======================<<相关操作函数>>======================= #
class Log:
    def __init__(self, log_path):
        self.log_path = log_path

    def __call__(self, msg):
        print(msg, end='\n')
        with open(self.log_path, 'a') as f:
            f.write(msg)


def model_load(dataset_idx, model_idx, batch_size):
    tem_path = './Tampered_model/'
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
            model_path = './Marked_model/alexnet_fash'
            save_path = './Tampered_model/alexnet_fash'
            layer_name = ['features.0.weight', 'features.3.weight']

        elif model_idx == 2:
            model = models.vgg16(pretrained=False)
            model.features._modules['0'] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)  # 对应Vgg16
            num_fc = model.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = './Marked_model/vgg16_fash'
            save_path = './Tampered_model/vgg16_fash'
            layer_name = ['features.0.weight', 'features.5.weight']

        elif model_idx == 3:
            model = models.resnet18(pretrained=False)
            model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            num_fc = model.fc.in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = './Marked_model/resnet18_fash'
            save_path = './Tampered_model/resnet18_fash'
            layer_name = ['conv1.weight', 'layer1.0.conv1.weight']

        elif model_idx == 4:
            model = models.mobilenet_v2(pretrained=False)
            model.features._modules['0'][0] = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=3, bias=False)
            num_fc = model.classifier[1].in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = './Marked_model/mobilenet_fash'
            save_path = './Tampered_model/mobilenet_fash'
            layer_name = ['features.0.0.weight', 'features.1.conv.0.0.weight']

        elif model_idx == 5:
            model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes)
            model.patch_embed.proj = nn.Conv2d(1, 768, kernel_size=16, stride=16)
            model_path = './Marked_model/vit_fash'
            save_path = './Tampered_model/vit_fash'
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
            model_path = './Marked_model/alexnet_cifar'
            save_path = './Tampered_model/alexnet_cifar'
            layer_name = ['features.0.weight', 'features.3.weight']

        elif model_idx == 2:
            model = models.vgg16(pretrained=False)
            num_fc = model.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = './Marked_model/vgg16_cifar'
            save_path = './Tampered_model/vgg16_cifar'
            layer_name = ['features.0.weight', 'features.2.weight']

        elif model_idx == 3:
            model = models.resnet18(pretrained=False)
            num_fc = model.fc.in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = './Marked_model/resnet18_cifar'
            save_path = './Tampered_model/resnet18_cifar'
            layer_name = ['conv1.weight', 'layer1.0.conv1.weight']

        elif model_idx == 4:
            model = models.mobilenet_v2(pretrained=False)
            num_fc = model.classifier[1].in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = './Marked_model/mobilenet_cifar'
            save_path = './Tampered_model/mobilenet_cifar'
            layer_name = ['features.0.0.weight', 'features.1.conv.0.0.weight']

        elif model_idx == 5:
            model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes)
            model_path = './Marked_model/vit_cifar'
            save_path = './Tampered_model/vit_cifar'
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
            model_path = './Marked_model/alexnet_tiny'
            save_path = './Tampered_model/alexnet_tiny'
            layer_name = ['features.0.weight', 'features.3.weight']

        elif model_idx == 2:
            model = models.vgg16(pretrained=False)
            num_fc = model.classifier[6].in_features  # 获取最后一个全连接层的输入是多少
            model.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = './Marked_model/vgg16_tiny'
            save_path = './Tampered_model/vgg16_tiny'
            layer_name = ['features.0.weight', 'features.5.weight']

        elif model_idx == 3:
            model = models.resnet18(pretrained=False)
            num_fc = model.fc.in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = './Marked_model/resnet18_tiny'
            save_path = './Tampered_model/resnet18_tiny'
            layer_name = ['conv1.weight', 'layer1.0.conv1.weight']

        elif model_idx == 4:
            model = models.mobilenet_v2(pretrained=False)
            num_fc = model.classifier[1].in_features  # 获取最后一个全连接层的输入是多少
            model.fc = torch.nn.Linear(in_features=num_fc, out_features=num_classes)  # 重新赋值最后一个全连接层的输入输出
            model_path = './Marked_model/mobilenet_tiny'
            save_path = './Tampered_model/mobilenet_tiny'
            layer_name = ['features.0.0.weight', 'features.1.conv.0.0.weight']

        elif model_idx == 5:
            model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes)
            model_path = './Marked_model/vit_tiny'
            save_path = './Tampered_model/vit_tiny'
            layer_name = ['cls_token', 'pos_embed']

    return model, train_loader, test_loader, layer_name, model_path, save_path


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
# 0：表示翻转指数位最高位的后门攻击(无差别破坏的后门攻击)；
# 1：表示按照块篡改，按照篡改率选择若干个块，然后篡改块内的若干个数（置为0）
# 2：表示随机篡改篡改，按照篡改率随机选择参数进行篡改（置为0）
# 3：表示使用正态分布的随机噪声来模拟攻击
# 4：表示无篡改
# 5：单独篡改FC(方式与1一致)
# 6：单独篡改Conv1(方式与1一致)
attack_method_idx = 1  # 攻击方法选择
flip_bits = 6
dataset_idx_list = [1]
model_idx_list = [3]  # 1:alexnet; 2:vgg16; 3:resnet18; 4:Mobilenet_V2(fash数据集0.1篡改率时设置修改10个参数)
# tamper_rate_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
tamper_rate_list = [0.1]
batch_size = 64
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
attack_times = positive_env("NNWHR_ATTACK_TRIALS", 5)  # 多次实验次数
log = Log(osp.join('./Tampered_model/', 'T_acc.txt'))
for dataset_idx in dataset_idx_list:
    for model_idx in model_idx_list:
        model, train_loader, test_loader, layer_name, model_path, save_path = model_load(dataset_idx, model_idx,
                                                                                                    batch_size)
        for tamper_rate in tamper_rate_list:
            print('The tampering rate:============>>', tamper_rate)
            acc_sum = 0
            for t_idx in range(attack_times):
                hy_seed = 26 * (t_idx + 1)
                model.load_state_dict(torch.load('{}.pth'.format(model_path), map_location="cpu", weights_only=True))
                # acc = acc_test(test_loader, device, model)
                # print('The watermarked model acc:==========>>', acc)
                model.eval()
                ori_weight, weight_num_list, weight_shape_list = para_load(model, layer_name)

                # ======================== <<攻击方式选择>> ======================== #
                # 0：表示翻转指数位最高位的后门攻击(无差别破坏的后门攻击)；
                # 1：表示按照块篡改，按照篡改率选择若干个块，然后篡改块内的若干个数（置为0）
                # 2：表示随机篡改篡改，按照篡改率随机选择参数进行篡改（置为0）
                # 3：表示使用正态分布的随机噪声来模拟攻击
                # 4：表示无篡改
                # 5：单独篡改FC(方式与1一致)
                # 6：单独篡改Conv1(方式与1一致)
                # ======================== <<攻击方式选择>> ======================== #
                if attack_method_idx == 0:
                    attacked_weight = bit_flip(ori_weight, block_size, flip_bits, hy_seed)

                elif attack_method_idx == 1:
                    mask_path = save_path + '_mask_' + str(tamper_rate) + '_' + str(t_idx)
                    attacked_weight = ran_block(ori_weight, tamper_rate, block_size, mask_path, hy_seed)

                elif attack_method_idx == 2:
                    attacked_weight = ran_weight(ori_weight, tamper_rate, block_size)

                elif attack_method_idx == 3:
                    attacked_weight = ran_gauss_dis(ori_weight, tamper_rate)

                elif attack_method_idx == 4:
                    attacked_weight = ran_no(ori_weight, tamper_rate)

                for layer_idx in range(len(layer_name)):
                    layer_name_tem = layer_name[layer_idx]
                    Marked_weight_tem = np.array(attacked_weight[layer_idx])
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

                # tampered_save_path = tampered_save_path + '_' + str(tamper_rate) + '_' + str(t_idx)
                tampered_model_path = save_path + '_' + str(tamper_rate) + '_' + str(t_idx)
                torch.save(model.state_dict(), '{}.pth'.format(tampered_model_path))
                acc = acc_test(test_loader, model)
                print('The modified model acc:==========>>', acc)
                acc_sum = acc_sum + acc
            acc_ave = acc_sum / attack_times
            msg = f'dataset index: {dataset_idx}, | model index: {model_idx}, | Tamper rate: {tamper_rate}, | Tamper acc_ave: {acc_ave}\n'
            log(msg)



