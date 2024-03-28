import numpy as np
import pandas as pd
import sys, os
from random import shuffle
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
import pickle

# 从文件中导入自定义模块
from config import DGSDTAModelConfig  # 导入模型配置
from model import DGSDTAModel  # 导入自定义模型类
# from dataset import DGSDTADataset  # 导入自定义数据集类
from my_dataset import DGSDTADataset  # 导入自定义数据集类
from utils import *  # 导入自定义工具函数
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"   # 果然是numpy版本的问题，升级了numpy到最新的版本没有问题了
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# 每个周期的训练函数
def train(model, device, train_loader, optimizer, epoch):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    for batch_idx, data in enumerate(train_loader):
        graph, seq_embed, seq_mask = data
        graph = graph.to(device)
        seq_embed = seq_embed.to(device)
        seq_mask = seq_mask.to(device)

        optimizer.zero_grad()
        output = model(graph, seq_embed)
        # print(output, graph.y.view(-1, 1).float().to(device))
        loss = loss_fn(output, graph.y.view(-1, 1).float().to(device))
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:  # 打印训练日志
            # print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
            #                                                                batch_idx * len(graph.x),
            #                                                                len(train_loader.dataset),
            #                                                                100. * batch_idx / len(train_loader),
            #                                                                loss.item()))
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx*64,
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))


# 预测函数
def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for graph, seq_embed, seq_mask in loader:
            graph = graph.to(device)
            seq_embed = seq_embed.to(device)
            seq_mask = seq_mask.to(device)

            output = model(graph, seq_embed)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, graph.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()


# 初始化训练
def train_init(dataset='skempi', pretrain=None):  # kiba  davis
    # 从pickle文件中加载序列到路径的映射，pickle文件中的内容是预先训练的embedding结果,这文件应该还包含掩码信息。
    with open('data/seq2path_prot_albert.pickle', 'rb') as handle:
        # 还原出来一个字典，key为seq、value为该氨基酸生成embedding的保存pt文件路径
        seq2path = pickle.load(handle)
    # 从pickle文件中加载SMILE到图的映射
    # with open('data/smile_graph.pickle', 'rb') as f:
    with open('data/pdb_stru_graph.pickle', 'rb') as f:
        smile2graph = pickle.load(f)
    cuda_name = CUDA
    # 加载训练和测试数据集，这两个数据集其实都是三列数据（smiles、target_sequence、affinity），所以后面改的时候可以改的时候只需要生成一个类似的csv文件就行
    train_data = DGSDTADataset('data/{}_train.csv'.format(dataset), smile2graph, seq2path)
    test_data = DGSDTADataset('data/{}_test.csv'.format(dataset), smile2graph, seq2path)
    # 准备数据以便于PyTorch小批量处理
    train_loader = DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False)
    # 设置设备为GPU或CPU
    device = torch.device(cuda_name if torch.cuda.is_available() else 'cpu')

    config = DGSDTAModelConfig()  # 获取模型配置
    graphNet = config['graphNet']  # 获取图网络类型
    model = DGSDTAModel(config).to(device)  # 实例化模型并移动到设备上
    if pretrain:  # 如果有预训练模型，则加载预训练参数
        print("used pretrain model {}".format(pretrain))
        state_dict = torch.load(pretrain)
        model.load_state_dict(state_dict)
    # 损失函数和优化器
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    return model, device, train_loader, test_loader, loss_fn, optimizer, graphNet, dataset


# 绘制学习曲线
import matplotlib.pyplot as plt


def learning_curve(statistics_mse, statistics_rmse, statistics_pcc, statistics_ci, num_epoch):
    # train_sizes, train_scores, valid_scores = learning_curve(model, x, y1, train_sizes=np.linspace(0.1, 1.0, 10), cv=5,
    #                                                          random_state=0)
    # train_scores_mean = np.mean(train_scores, axis=1)
    # valid_scores_mean = np.mean(valid_scores, axis=1)
    x = np.linspace(1, num_epoch, num_epoch)
    plt.plot(x, statistics_mse, label='MSE', marker = "o", mfc = "white", ms = 5, c='r')  # 绘制测试均方误差曲线
    plt.plot(x, statistics_rmse, label='RMSE', marker = "x", mfc = "white", ms = 5, c='b')  # 绘制测试RMSE曲线
    plt.plot(x, statistics_pcc, label='PCC', marker = ".", mfc = "white", ms = 5, c='g')  # 绘制测试PCC曲线
    plt.plot(x, statistics_ci, label='CI', marker = "^", mfc = "white", ms = 5, c='y')  # 绘制测试CI曲线
    # plt.plot(x, y2, label='Train MSE')  # 绘制训练均方误差曲线

    plt.title("MSE")  # 设置图标题
    plt.xlabel("epoch")  # 设置x轴标签
    plt.ylabel("MSE")  # 设置y轴标签
    plt.legend()  # 显示图例
    plt.grid(True)  # 显示网格线
    # plt.show()  # 显示图形
    plt.savefig("output/mse.png")  # 保存图像
    plt.close()
def mse_curve(statistics_mse, statistics_rmse,  num_epoch):

    x = np.linspace(1, num_epoch, num_epoch)
    plt.plot(x, statistics_mse, label='MSE', marker = "o", mfc = "white", ms = 5, c='r')  # 绘制测试均方误差曲线
    plt.plot(x, statistics_rmse, label='RMSE', marker = "x", mfc = "white", ms = 5, c='b')  # 绘制测试RMSE曲线
    plt.title("MSE AND RMSE")  # 设置图标题
    plt.xlabel("Epoch")  # 设置x轴标签
    plt.ylabel("MSE AND RMSE")  # 设置y轴标签
    plt.legend()  # 显示图例
    plt.grid(True)  # 显示网格线
    # plt.show()  # 显示图形
    plt.savefig("output/mse.png")  # 保存图像
    plt.close()

def pcc_curve(statistics_pcc, statistics_ci, num_epoch):

    x = np.linspace(1, num_epoch, num_epoch)
    plt.plot(x, statistics_pcc, label='PCC', marker = ".", mfc = "white", ms = 5, c='g')  # 绘制测试PCC曲线
    plt.plot(x, statistics_ci, label='CI', marker = "^", mfc = "white", ms = 5, c='y')  # 绘制测试CI曲线

    plt.title("PCC AND CI")  # 设置图标题
    plt.xlabel("Epoch")  # 设置x轴标签
    plt.ylabel("PCC AND CI")  # 设置y轴标签
    plt.legend()  # 显示图例
    plt.grid(True)  # 显示网格线
    # plt.show()  # 显示图形
    plt.savefig("output/pcc.png")  # 保存图像
    plt.close()

# 画带线性回归最佳拟合线的散点图
def scatter_pic(G, P, pcc_value, rmse_value):
    fig, ax = plt.subplots(figsize=(7, 3), dpi=200)

    # --- Remove spines and add gridlines

    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(ls="--", lw=0.25, color="#4E616C")

    # 计算最佳拟合线的斜率和截距
    slope, intercept = np.polyfit(P, G, 1)

    # 绘制散点图
    plt.scatter(P, G)

    # 绘制最佳拟合线
    plt.plot(P, slope*P + intercept,color='red')

    # 设置图形标题和坐标轴标签
    plt.title('Performance on protein mutations')
    plt.xlabel('Predicted ΔΔG')
    plt.ylabel('Experimental ΔΔG')

    ax.xaxis.set_tick_params(length=2, color="#4E616C", labelcolor="#4E616C", labelsize=6)
    ax.yaxis.set_tick_params(length=2, color="#4E616C", labelcolor="#4E616C", labelsize=6)

    ax.spines["bottom"].set_edgecolor("#4E616C")
    # 添加文字
    x_min, x_max = plt.xlim()
    y_min, y_max = plt.ylim()
    plt.text(x_max * 0.9, y_min * 0.65, 'PCC: {}'.format(str(pcc_value)[:5]), ha='right', va='bottom')
    plt.text(x_max * 0.9, y_min * 0.85, 'RMSE: {}'.format(str(rmse_value)[:5]), ha='right', va='bottom')
    plt.legend()  # 显示图例
    plt.grid(True)  # 显示网格线
    # plt.show()  # 显示图形
    plt.savefig("output/scatter.png")  # 保存图像
    # 显示图形
    # plt.show()
    plt.close()



# 设置训练参数
TRAIN_BATCH_SIZE = 64  # 当使用SKEMPI_all_pdbs的时候，要把批处理大小降低，因为GPU内存不如cpu那么大
TEST_BATCH_SIZE = 64
LR = 0.0002
LOG_INTERVAL = 10  # 日志间隔，原来是100现在改成10，因为一个batch根本没有100条数据
NUM_EPOCHS = 500
CUDA = 'cuda:0'  # GPU设备
# CUDA = 'cpu'  # GPU设备

# 预训练模型文件名
pretrain = False
# 进行单点突变的单独实验，要使用skempiv2数据集训练好的模型进行预训练。
# pretrain = 'D:\wyx\dgsddg\data\model\model_GAT_skempi.model'
model, device, train_loader, test_loader, loss_fn, optimizer, graphNet, dataset = train_init(pretrain=pretrain)
# 这个是skempi数据集实验结果的保存路径
model_file_name = 'model_' + graphNet + '_' + dataset + '_40' + '.model'
# 这个是SKP1102s的模型保存路径
# model_file_name = 'model_' + graphNet + '_' + 'skp1102s' + '.model'
# 这个是AB-BindS645的模型保存路径
# model_file_name = 'model_' + graphNet + '_' + 's645' + '.model'
result_file_name = 'result_' + graphNet + '_' + dataset + '.csv'

print('Learning rate: ', LR)
print('Epochs: ', NUM_EPOCHS)
print('The best model will be saved at: {}'.format(model_file_name))

best_mse = 1000
best_ci = 0
best_pcc = 0
best_epoch = -1
the_mse = []
the_pcc = 0
the_rmse = 0
statistics_mse = []
statistics_rmse = []
statistics_pcc = []
statistics_ci = []

for epoch in range(NUM_EPOCHS):
    train(model, device, train_loader, optimizer, epoch + 1)  # 训练的过程
    G, P = predicting(model, device, test_loader)  # G、P分别是labels和predict的flatten的结果
    ret = [rmse(G, P), mse(G, P), pearson(G, P), spearman(G, P), ci(G, P)]  # 计算参数
    statistics_mse.append(ret[1])
    statistics_rmse.append(ret[0])
    statistics_pcc.append(max(ret[2], ret[3]))
    statistics_ci.append(ret[-1])
    # the_pcc = max(ret[2], ret[3])
    # the_rmse = ret[0]

    if ret[1] < best_mse:  # 是根据mse来决定是不是最好的一次训练模型
    # if (ret[2] > best_pcc) or (ret[3] > best_pcc):  # 是根据pcc来决定是不是最好的一次训练模型
        torch.save(model.state_dict(), model_file_name)  # 保存最佳模型
        with open(result_file_name, 'w') as f:
            f.write(','.join(map(str, ret)))  # 写入结果文件
        best_epoch = epoch + 1
        best_mse = ret[1]
        best_ci = ret[-1]
        best_pcc = max(ret[2], ret[3])
        the_pcc = max(ret[2], ret[3])
        the_rmse = ret[0]
        scatter_pic(G, P, the_pcc, the_rmse)
        print('rmse improved at epoch ', best_epoch, '; best_mse,best_ci:', best_mse, best_ci, graphNet, dataset)
    else:
        print(ret[1], 'No improvement since epoch ', best_epoch, '; best_mse,best_ci:', best_mse, best_ci, graphNet,
              dataset)
    # scatter_pic(G, P, the_pcc, best_mse)
mse_curve(statistics_mse, statistics_rmse, NUM_EPOCHS)
pcc_curve(statistics_pcc, statistics_ci, NUM_EPOCHS)