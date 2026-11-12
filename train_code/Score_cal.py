from experiment_config import CIFAR10, FASHION_MNIST, TINY_IMAGENET, evaluation_dataset, limit_dataset, positive_env
import os
import timm
import torch
import warnings
import random
import numpy as np
import torchvision
import torch.nn as nn
import os.path as osp
import torchvision.models as models
import torchvision.transforms as transforms
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
        train_loader = torch.utils.data.DataLoader(dataset=limit_dataset(train_dataset, "NNWHR_SCORE_SAMPLES"), batch_size=batch_size, shuffle=True)
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
        train_loader = torch.utils.data.DataLoader(dataset=limit_dataset(train_dataset, "NNWHR_SCORE_SAMPLES"), batch_size=batch_size, shuffle=True)
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
        train_loader = torch.utils.data.DataLoader(dataset=limit_dataset(train_dataset, "NNWHR_SCORE_SAMPLES"), batch_size=batch_size, shuffle=True)
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

    return model, train_loader, test_loader, layer_name, sorted_path


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


def model_training(model, train_loader, learning_rate, num_epochs):
    loss_fun = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            model = model.to(device)
            output = model(images)
            loss = loss_fun(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model


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
num_epochs = positive_env("NNWHR_SCORE_EPOCHS", 3)
dataset_idx_list = [1]  # 1:FashionMNIST; 2:cifar-10; 3:Tiny_ImageNet
model_idx_list = [3]  # 1:alexnet; 2:vgg16; 3:resnet18; 4:Mobilenet_V2
learning_rate = 0.0001
log = Log(osp.join('./Marked_model/', 'M_acc.txt'))
for dataset_idx in dataset_idx_list:
    for model_idx in model_idx_list:
        model, train_loader, test_loader, layer_name, sorted_path = model_load(dataset_idx, model_idx, batch_size)
        model = model_training(model, train_loader, learning_rate, num_epochs)
        count = 0
        score_abs = []
        score_gradient = []
        score_weight = []
        k = 0
        for current_layer in layer_name:
            for name, parms in model.named_parameters():
                # print(name, ':', parms.size())
                if current_layer in name:
                    # print(name, ':', parms.size())
                    # # 计算一阶泰勒展开的绝对值
                    improtance_score1 = torch.abs(torch.mul(parms, parms.grad))
                    improtance_score1 = improtance_score1.data.cpu()
                    improtance_score1 = improtance_score1.detach().numpy()
                    improtance_score1 = improtance_score1.flatten()
                    # 计算一阶泰勒展开的平方
                    improtance_score2 = torch.abs(parms.grad)
                    improtance_score2 = improtance_score2.data.cpu()
                    improtance_score2 = improtance_score2.detach().numpy()
                    improtance_score2 = improtance_score2.flatten()
                    # 计算一阶泰勒展开的绝对值
                    improtance_score3 = torch.abs(parms)
                    improtance_score3 = improtance_score3.data.cpu()
                    improtance_score3 = improtance_score3.detach().numpy()
                    improtance_score3 = improtance_score3.flatten()

                    if k == 0:
                        weight_num = len(improtance_score1)
                        k = k + 1

                    elif k == 1:
                        weight_num = weight_num + len(improtance_score1)
                        tem_score1 = np.zeros(weight_num)
                        tem_score2 = np.zeros(weight_num)
                        tem_score3 = np.zeros(weight_num)

                    count = count + 1
                    if count > 0:
                        break

            score_abs.append(improtance_score1)
            score_gradient.append(improtance_score2)
            score_weight.append(improtance_score3)

        score_abs_tem = np.hstack([np.array(score_abs[0]), np.array(score_abs[1])])
        score_gradient_tem = np.hstack([np.array(score_gradient[0]), np.array(score_gradient[1])])
        score_weight_tem = np.hstack([np.array(score_weight[0]), np.array(score_weight[1])])

        tem_score1 = tem_score1 + score_abs_tem  # 对模型参数重要性得分进行累加求和
        tem_score2 = tem_score2 + score_gradient_tem
        tem_score3 = tem_score3 + score_weight_tem
        total_step = len(train_loader)
        average_score1 = tem_score1 / (total_step * num_epochs)  # 求平均得分
        average_score2 = tem_score2 / (total_step * num_epochs)  # 求平均得分
        average_score3 = tem_score3 / (total_step * num_epochs)  # 求平均得分
        abs_sort = np.argsort(average_score1)
        gradient_sort = np.argsort(average_score2)
        weight_sort = np.argsort(average_score3)
        os.makedirs(os.path.dirname(sorted_path), exist_ok=True)
        score_path1 = sorted_path
        score_path2 = sorted_path + '_gra'
        score_path3 = sorted_path + '_wei'
        np.save('{}.npy'.format(score_path1), abs_sort)
        np.save('{}.npy'.format(score_path2), gradient_sort)
        np.save('{}.npy'.format(score_path3), weight_sort)
