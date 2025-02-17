import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Sequential, Linear, ReLU
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp


# 定义 GAT_GCN 模型类
class GAT_GCN(torch.nn.Module):
    # def __init__(self, graph_output_dim=128, graph_features_dim=78, dropout=0.2):
    def __init__(self, graph_output_dim=128, graph_features_dim=24, dropout=0.2):
        super(GAT_GCN, self).__init__()
        # 定义图卷积层
        self.conv1 = GATConv(graph_features_dim, graph_features_dim, heads=10)
        self.conv2 = GCNConv(graph_features_dim * 10, graph_features_dim * 10)
        # 定义全连接层
        self.fc_g1 = torch.nn.Linear(graph_features_dim * 10 * 2, 1500)
        self.fc_g2 = torch.nn.Linear(1500, graph_output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, graph):
        x, edge_index, batch = graph.x, graph.edge_index, graph.batch
        # 第一个图卷积层和激活函数
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        # 第二个图卷积层和激活函数
        x = self.conv2(x, edge_index)
        x = self.relu(x)
        # 进行全局最大池化和全局平均池化
        x = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        # 进行全连接层和激活函数
        x = self.relu(self.fc_g1(x))
        x = self.dropout(x)
        x = self.fc_g2(x)
        return x


# 定义 GATNet 模型类
class GATNet(torch.nn.Module):
    # def __init__(self, graph_output_dim=128, graph_features_dim=78, dropout=0.2):
    def __init__(self, graph_output_dim=128, graph_features_dim=36, dropout=0.2):
        super(GATNet, self).__init__()

        # 定义图卷积层
        self.gcn1 = GATConv(graph_features_dim, graph_features_dim, heads=10, dropout=dropout)
        self.gcn2 = GATConv(graph_features_dim * 10, graph_output_dim, dropout=dropout)
        # 定义全连接层
        self.fc_g1 = nn.Linear(graph_output_dim, graph_output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, graph):
        x, edge_index, batch = graph.x, graph.edge_index, graph.batch

        # 应用 dropout
        x = F.dropout(x, p=0.2, training=self.training)
        # 第一个图卷积层和激活函数
        x = F.elu(self.gcn1(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        # 第二个图卷积层和激活函数
        x = self.gcn2(x, edge_index)
        x = self.relu(x)
        # 进行全局最大池化
        x = gmp(x, batch)
        # 进行全连接层和激活函数
        x = self.fc_g1(x)
        x = self.relu(x)
        return x

    # 定义 SeqNet 模型类


class SeqNet(torch.nn.Module):
    def __init__(self, seq_embed_dim=1024, n_filters=256, seq_output_dim=1024, dropout=0.2):
        super(SeqNet, self).__init__()

        # 定义卷积层和池化层
        self.conv_xt_1 = nn.Conv1d(in_channels=seq_embed_dim, out_channels=n_filters, kernel_size=5)
        self.pool_xt_1 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.conv_xt_2 = nn.Conv1d(in_channels=n_filters, out_channels=seq_embed_dim, kernel_size=5)
        self.pool_xt_2 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.conv_xt_3 = nn.Conv1d(in_channels=seq_embed_dim, out_channels=n_filters, kernel_size=5)
        self.pool_xt_3 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.conv_xt_4 = nn.Conv1d(in_channels=n_filters, out_channels=int(n_filters / 2), kernel_size=3)
        self.pool_xt_4 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        # 定义全连接层
        self.fc1_xt = nn.Linear(128 * 61, seq_output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, seq_embed, seq_mask=None):
        # 应用卷积层和激活函数
        xt = self.conv_xt_1(seq_embed.transpose(1, 2))
        xt = self.relu(xt)
        xt = self.pool_xt_1(xt)
        xt = self.conv_xt_2(xt)
        xt = self.relu(xt)
        xt = self.pool_xt_2(xt)
        xt = self.conv_xt_3(xt)
        xt = self.relu(xt)
        xt = self.pool_xt_3(xt)
        xt = self.conv_xt_4(xt)
        xt = self.relu(xt)
        xt = self.pool_xt_4(xt)

        # 展平
        xt = xt.view(-1, 128 * 61)
        # 进行全连接层和激活函数
        xt = self.fc1_xt(xt)
        xt = self.relu(xt)
        xt = self.dropout(xt)
        return xt


# 定义 DGSDTAModel 模型类
class DGSDTAModel(torch.nn.Module):
    def __init__(self, config):
        super(DGSDTAModel, self).__init__()
        dropout = config['dropout']
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # SMILES 图分支
        if config['graphNet'] == 'GAT_GCN':
            self.graph = GAT_GCN(config['graph_output_dim'], config['graph_features_dim'], dropout)
        elif config['graphNet'] == 'GAT':
            self.graph = GATNet(config['graph_output_dim'], config['graph_features_dim'], dropout)
        else:
            print("Unknown model name")

        # 序列分支
        self.seqnet = SeqNet(config['seq_embed_dim'], config['n_filters'], config['seq_output_dim'], dropout)

        # 组合层
        self.fc1 = nn.Linear(config['graph_output_dim'] + config['seq_output_dim'], 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.out = nn.Linear(256, 1)

    def forward(self, graph, seq_embed, seq_mask=None):
        graph_output = self.graph(graph)
        seq_output = self.seqnet(seq_embed, seq_mask)

        # 连接两个分支的输出
        xc = torch.cat((graph_output, seq_output), 1)
        # 添加一些全连接层
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        out = self.out(xc)
        return out


if __name__ == "__main__":
    from torch_geometric.loader import DataLoader
    import pickle
    from dataset import DGSDTADataset
    from config import DGSDTAModelConfig

    with open('data/seq2path_prot_albert.pickle', 'rb') as handle:
        seq2path = pickle.load(handle)
    with open('data/smile_graph.pickle', 'rb') as f:
        smile2graph = pickle.load(f)

    train_dataset = DGSDTADataset('data/davis_train.csv', smile2graph, seq2path)
    loader = DataLoader(dataset=train_dataset, batch_size=8, shuffle=True)

    device = torch.device("cpu")
    config = DGSDTAModelConfig()
    print(config['graphNet'])
    model = DGSDTAModel(config).to(device)

    for i, data in enumerate(loader):
        graph, seq_embed, seq_mask = data
        graph = graph.to(device)
        seq_embed = seq_embed.to(device)
        seq_mask = seq_mask.to(device)
        out = model(graph, seq_embed)
        print(out.shape)
        break
