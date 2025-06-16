# 基于跨视图一致性与自适应加权机制的多视图聚类系统

## 1. 项目背景
近年来，多视图聚类（MVC）技术因其在解决多来源无监督聚类任务中的重要作用而备受关注。然而，现有的MVC方法仍存在一些关键局限性，制约了其性能的进一步提升。一方面，大多数方法在学习多视图一致性表示时，忽视了视图特定信息的保留，未能充分挖掘多视图数据的独特特征及其潜在聚类结构。另一方面，现有模型主要聚焦于不同视图中同一样本的一致性学习，却忽略了跨视图场景中相似但非同类样本之间的潜在关联。

**技术动机**：  
- 跨视图一致性：解决不同视图特征空间不一致问题  
- 自适应加权：自动学习各视图的贡献权重，避免人工调参  

----



## 2. 现有方法对比
| 方法类别   | 代表算法                       | 优缺点           |
| ---------- | ------------------------------ | ---------------- |
| 子空间学习 | Multi-view Spectral Clustering | 视图权重固定     |
| 图融合     | Multi-view Graph Clustering    | 忽略视图质量差异 |
| 深度聚类   | Autoencoder-based Methods      | 需要大量标注数据 |

**本方法定位**：  
结合子空间学习与自适应加权的半监督聚类方法

----



## 3. 核心方法

### 3.1 整体流程
模型图整体框架：![](https://raw.githubusercontent.com/Xpccccc/PicGo/main/data202506122140423.png)

AFW模块细节：

![](https://raw.githubusercontent.com/Xpccccc/PicGo/main/data202506122140116.png)

----



### 3.2 关键算法

**目标函数**：
$$
\mathcal{L} = \mathcal{L}_R + \mathcal{L}_C + \mathcal{L}_A + \mathcal{L}_Q
$$
其中： ${\mathcal{L}}_{R}$ 重构损失， ${\mathcal{L}}_{C}$是跨视图对比损失，${\mathcal{L}}_{A}$是使用AFW模块后的$KL$散度损失，${\mathcal{L}}_{Q}$ 是标签损失.

**关键代码**

`network.py`文件：

```python
import torch.nn as nn
# import torch.nn.functional as F
from torch.nn.functional import normalize
from torch.nn import Parameter
from sklearn.cluster import KMeans
import torch
from typing import Optional
from loss import Contrastive_loss


class Encoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Encoder, self).__init__()  ##super用来调用父类
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 2000),
            nn.ReLU(),
            nn.Linear(2000, feature_dim),
        )

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, 2000),
            nn.ReLU(),
            nn.Linear(2000, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, input_dim)
        )

    def forward(self, x):
        return self.decoder(x)


class Clustering(nn.Module):
    def __init__(self, args):
        super(Clustering, self).__init__()
        self.kmeans = KMeans(n_clusters=args.num_cluster, n_init=20)
        self.clustering_layer = DECModule(cluster_number=args.num_cluster,
                                          embedding_dimension=args.cluster_hidden_dim)

    def forward(self, h):
        # self.kmeans.fit(h.cpu().detach().numpy())
        clustering_layer = self.clustering_layer
        # cluster_centers = torch.tensor(
        #     self.kmeans.cluster_centers_, dtype=torch.float, requires_grad=True)
        # # cluster_centers = cluster_centers.to(device)
        # with torch.no_grad():
        #     clustering_layer.cluster_centers.copy_(cluster_centers)
        q = clustering_layer(h)
        return q



class VCF(nn.Module):
    def __init__(self,in_feature_dim,class_num):
        super(VCF,self).__init__()
        TransformerEncoderLayer = nn.TransformerEncoderLayer(d_model=in_feature_dim, nhead=1,dim_feedforward=256)
        self.TransformerEncoder = nn.TransformerEncoder(TransformerEncoderLayer, num_layers=1)
        self.cluster = nn.Sequential(
            nn.Linear(in_feature_dim,class_num),
            nn.Softmax(dim=1)
        )
    def forward(self,C):
        # print(" in {}".format(C.size()))
        temp = self.TransformerEncoder(C)
        t = self.cluster(temp)
        return t, temp

class Network(nn.Module):
    def __init__(self, view, input_size, args, class_num, device):
        super(Network, self).__init__()
        feature_dim = args.feature_dim
        high_feature_dim = args.high_feature_dim
        self.encoders = []
        self.decoders = []
        self.log_y = []
        for v in range(view):
            self.encoders.append(Encoder(input_size[v], feature_dim).to(device))
            self.decoders.append(Decoder(input_size[v], feature_dim).to(device))
        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)
        
        self.feature_contrastive_module = nn.Sequential(
            nn.Linear(feature_dim, high_feature_dim),
        )
        
        self.clustering = Clustering(args)
        self.view = view
        self.VCF = VCF(feature_dim * self.view,class_num)

    def forward(self, xs):
        xrs = []
        zs = []
        hs = []
        qs = []
        for v in range(self.view):
            x = xs[v]
            z = self.encoders[v](x)
            xr = self.decoders[v](z)
            h_v = normalize(self.feature_contrastive_module(z), dim=1)
            q = self.clustering(h_v)
            
            zs.append(z)
            hs.append(h_v)
            xrs.append(xr)
            qs.append(q)
        
        return hs, qs, xrs, zs

   
    def forward_cluster(self, xs):
        qs = []
        preds = []
        for v in range(self.view):
            x = xs[v]
            z = self.encoders[v](x)
            h_v = normalize(self.feature_contrastive_module(z), dim=1)
            q = self.clustering(h_v)
            qs.append(q)
            pred = torch.argmax(q, dim=1)
            qs.append(q)
            preds.append(pred)
        return qs, preds



class DECModule(nn.Module):
    def __init__(
            self,
            cluster_number: int,
            embedding_dimension: int,
            alpha: float = 1.0,
            cluster_centers: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Module to handle the soft assignment, for a description see in 3.1.1. in Xie/Girshick/Farhadi,
        where the Student's t-distribution is used measure similarity between feature vector and each
        cluster centroid.

        param cluster_number: number of clusters
        param embedding_dimension: embedding dimension of feature vectors
        param alpha: parameter representing the degrees of freedom in the t-distribution, default 1.0
        param cluster_centers: clusters centers to initialise, if None then use Xavier uniform
        """
        super(DECModule, self).__init__()
        self.embedding_dimension = embedding_dimension
        self.cluster_number = cluster_number
        self.alpha = alpha
        if cluster_centers is None:
            initial_cluster_centers = torch.zeros(
                self.cluster_number, self.embedding_dimension, dtype=torch.float
            )
            nn.init.xavier_uniform_(initial_cluster_centers)
        else:
            initial_cluster_centers = cluster_centers
        self.cluster_centers = Parameter(initial_cluster_centers)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Compute the soft assignment for a batch of feature vectors, returning a batch of assignments
        for each cluster.
        param batch: FloatTensor of [batch size, embedding dimension]
        return: FloatTensor [batch size, number of clusters]
        """
        norm_squared = torch.sum((batch.unsqueeze(1) - self.cluster_centers) ** 2, 2)
        numerator = 1.0 / (1.0 + (norm_squared / self.alpha))
        power = float(self.alpha + 1) / 2
        numerator = numerator ** power
        return numerator / torch.sum(numerator, dim=1, keepdim=True)  # qij


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, num_heads, attn_bias_dim):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads

        self.att_size = att_size = hidden_size // num_heads
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size//2, num_heads * att_size)
        self.linear_k = nn.Linear(hidden_size//2, num_heads * att_size)
        self.linear_v = nn.Linear(hidden_size//2, num_heads * att_size)
        self.linear_bias = nn.Linear(attn_bias_dim, num_heads)
        self.att_dropout = nn.Dropout(attention_dropout_rate)

        self.output_layer = nn.Linear(num_heads * att_size, 1)

    def forward(self, q, k, v):

        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)

        q = self.linear_q(q).view(batch_size, -1, self.num_heads, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.num_heads, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.num_heads, d_v)

        q = q.transpose(1, 2)                  # [b, h, q_len, d_k]
        v = v.transpose(1, 2)                  # [b, h, v_len, d_v]
        k = k.transpose(1, 2).transpose(2, 3)  # [b, h, d_k, k_len]


        q = q * self.scale
        x = torch.matmul(q, k)  # [b, h, q_len, k_len]

        x = torch.softmax(x, dim=3)
        x = self.att_dropout(x)
        x = x.matmul(v)  # [b, h, q_len, attn]

        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
        x = x.view(batch_size, self.num_heads * d_v)

        x = self.output_layer(x)

        return x



class FeedForwardNetwork(nn.Module):
    def __init__(self, view, ffn_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(view, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, view)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        x = x.unsqueeze(1)
        return x
```

`train.py`文件：

```python
import numpy
import torch
from network import Network,MultiHeadAttention, FeedForwardNetwork
from metric import valid, eva
from torch.utils.data import Dataset
import numpy as np
import argparse
import random
from loss import Contrastive_loss
from dataloader import load_data

from sklearn.preprocessing import MinMaxScaler  # 从 sklearn 库中导入 MinMaxScaler，用于特征缩放
from sklearn.cluster import KMeans  # 从 sklearn 库中导入 KMeans 类，用于 K均值聚类
from scipy.optimize import linear_sum_assignment  # 从 scipy 库中导入 linear_sum_assignment 函数，用于线性分配
# from sklearn import manifold
# import matplotlib.pyplot as plt
import scipy.io as sio

import torch.nn as nn  # 导入 PyTorch 的神经网络模块，并将其重命名为 nn
import torch.nn.functional as F  # 导入 PyTorch 的功能性神经网络模块，并将其重命名为 F
import os

import setproctitle
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*reduction: 'mean'.*")

setproctitle.setproctitle('Xp')

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# MNIST-USPS 1
# BDGP  1
# CCV
# Fashion 
# Caltech101
# CIFAR  1
# Hdigit
# Prokaryotic
# Scene15
# BBCSport
# Reuters
# YouTubeFace
# NUS_WIDE
# Caltech-5V
# Cora
# Wiki
Dataname = 'MNIST-USPS'
parser = argparse.ArgumentParser(description='train')

parser.add_argument("--threshold", type=float, default=0.8)  # 添加阈值参数

parser.add_argument('--dataset', default=Dataname)
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument("--temperature_c", default=0.01)

parser.add_argument("--lmd", default=1.0)

parser.add_argument("--workers", default=8)

parser.add_argument('--num_heads', type=int, default=8)
parser.add_argument('--hidden_dim', type=int, default=256)
parser.add_argument('--ffn_size', type=int, default=32)
parser.add_argument('--attn_bias_dim', type=int, default=6)
parser.add_argument('--attention_dropout_rate', type=float, default=0.5)

parser.add_argument("--feature_dim", default=512)
parser.add_argument("--high_feature_dim", default=512)
parser.add_argument("--cluster_hidden_dim", default=512)
parser.add_argument("--num_cluster", default=10)
parser.add_argument("--views", default=3)
parser.add_argument("--device",
                    default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

parser.add_argument("--weight_decay", type=float, default=0.)  # 添加权重衰减参数
parser.add_argument("--learning_rate", default=0.0003)# 0.0003 ,0.0002
parser.add_argument("--seed", type=int, default=25)
parser.add_argument("--mse_epochs", default=200)# 2 
parser.add_argument("--con_epochs", default=300)# 2


args = parser.parse_args()
args.learning_rate = float(args.learning_rate)
args.temperature_c = float(args.temperature_c)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# The seed was fixed for the performance reproduction, which was higher than the values shown in the paper.
if args.dataset == "MNIST-USPS":
    args.learning_rate = 0.00059
    args.con_epochs = 250
    args.mse_epochs = 50
    args.seed = 2 # 25
    args.temperature_c = 0.01

    
if args.dataset == "BDGP":
    args.learning_rate = 0.00002
    args.mse_epochs = 200
    args.con_epochs = 300 # 10
    args.seed = 25 # 25

    args.temperature_c = 0.6
    
    
# if args.dataset == "CCV":
#     args.learning_rate = 0.0003
#     args.con_epochs = 100
#     seed = 30
# if args.dataset == "CIFAR":
#     args.learning_rate = 0.0003
#     args.con_epochs = 50
#     seed = 20
    
if args.dataset == "Caltech-5V":
    # args.learning_rate = 0.00034 # 0.00045
    args.con_epochs = 151
    args.temperature_c = 0.02
    args.lmd = 1.0

    
if args.dataset == "Caltech-4V":
    args.learning_rate = 0.00023
    args.con_epochs = 300
    args.seed = 25 # 3
    args.temperature_c = 0.12
    
    
if args.dataset == "Caltech-3V":
    args.learning_rate = 0.00023
    args.con_epochs = 300
    args.seed = 3 # 3
    args.temperature_c = 0.05
    
    
if args.dataset == "Caltech-2V":
    args.learning_rate = 0.0002
    args.mse_epochs = 50
    args.con_epochs = 100
    args.seed = 25 # 3
    args.temperature_c = 0.03 # 0.03
    

if args.dataset == "Cora":
    args.learning_rate = 0.00032
    args.mse_epochs = 50
    args.con_epochs = 15  # 0.0003 20,
    args.seed = 25 # 3
    args.temperature_c = 0.55 # 0.03
    
    
if args.dataset == "Prokaryotic":
    args.learning_rate = 0.00001
    args.mse_epochs = 50
    args.con_epochs = 28
    args.seed = 25 # 3
    
# if args.dataset == "YouTubeFace":
#     args.learning_rate = 0.0003
#     args.con_epochs = 30
#     seed = 10
if args.dataset == "Scene15":
    args.learning_rate = 0.00004
    args.mse_epochs = 50
    args.con_epochs = 300
    args.seed = 25 # 3
    args.temperature_c = 0.02 # 0.03
    
# if args.dataset == "BBCSport":
#     args.learning_rate = 0.0002
#     args.con_epochs = 200
#     seed = 25
    
    
if args.dataset == "Wiki":
    args.learning_rate = 0.000055
    args.mse_epochs = 50
    args.con_epochs = 300
    args.seed = 25 # 3
    args.temperature_c = 0.06 # 0.03

    
# if args.dataset == "NUS_WIDE":
#     args.learning_rate = 0.0003
#     args.con_epochs = 100  # 0.0004
#     seed = 7
# if args.dataset == "Hdigit":
#     args.learning_rate = 0.0003
#     args.con_epochs = 100  # 0.0004
#     seed = 30 # 18 19

if args.dataset == "Reuters":
    args.learning_rate = 0.00005
    args.con_epochs = 144  # 0.0003
    args.seed = 25
    args.mse_epochs = 50
    args.temperature_c = 0.03

    
# if args.dataset == "Caltech101":
#     args.learning_rate = 0.0003
#     args.con_epochs = 150  # 0.0003
#     seed = 62 # 50 52

    
# if args.dataset == "NUS_WIDE":
#     args.learning_rate = 0.0002
#     args.con_epochs = 150  # 0.0003
#     seed = 3 # 50 52

    
# if args.dataset == "MSRC_v1":
#     # args.batch_size = 128
#     args.learning_rate = 0.0003
#     args.con_epochs = 150  # 0.0003




def setup_seed(seed):
    torch.manual_seed(seed)  # 设置CPU生成随机数的种子，方便下次复现实验结果
    torch.cuda.manual_seed_all(seed)  # 在GPU中设置生成随机数的种子
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# setup_seed(seed)
setup_seed(args.seed)


dataset, dims, view, data_size, class_num = load_data(args.dataset)

print("len :{}".format(len(dataset)))
args.num_cluster = class_num

# 按照batch size封装成Tensor
data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=args.batch_size,
    shuffle=True,
    drop_last=True,
)


def pretrain(epoch):
    tot_loss = 0.
    criterion = torch.nn.MSELoss()  # 均方损失
    for batch_idx, (xs, _, _) in enumerate(data_loader):  # enumerate：将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标
        for v in range(view):
            # print("v {}".format(v))
            xs[v] = xs[v].to(args.device)
        optimizer.zero_grad()
        hs, _, xrs, zs= model(xs)
        # hs, _, xrs, zs = model(xs)
        loss_list = []
        # Zs = []
        # Hs = []
        for v in range(view):
            loss_list.append(criterion(xs[v], xrs[v]))
        loss = sum(loss_list)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
    if(epoch == 50):
        print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss / len(data_loader)))

    return tot_loss / len(data_loader)


def contrastive_train(epoch,lmd,p_sample,adaptive_weight):
    tot_loss = 0.
    mse = torch.nn.MSELoss()
    Z_batch = []
    H_batch = []
    for batch_idx, (xs, _, _) in enumerate(data_loader):
        for v in range(view):
            xs[v] = xs[v].to(args.device)
        optimizer.zero_grad()
        hs, qs, xrs, zs  = model(xs)
        loss_list = []
        local_pse = []
        
        # global contrastive calibration
        hs_tensor = torch.tensor([]).cuda()

        # obtain global view feature
        for v in range(view):
            hs_tensor = torch.cat((hs[v], hs_tensor), 0)
            # print("hs {}".format(hs[v].size()))
        hs_tensor = torch.tensor([]).cuda()

        for v in range(view):
            hs_tensor = torch.cat((hs_tensor, torch.mean(hs[v], 1).unsqueeze(1)), 1) # d * v

        # print("hs_tensor {}".format(hs_tensor.size()))

        # transpose
        hs_tensor = hs_tensor.t()

        # process by the attention
        hs_atten = attention_net(hs_tensor, hs_tensor, hs_tensor) # v * 1

        # learn the view sampling distribution
        p_learn = p_net(p_sample) # v * 1

        # regulatory factor
        r = hs_atten * p_learn
        s_p = nn.Softmax(dim=0)
        r = s_p(r)

        # adjust adaptive weight
        adaptive_weight = r * adaptive_weight

        # obtain fusion feature
        fusion_feature = torch.zeros([hs[0].shape[0], hs[0].shape[1]]).cuda()
        # print("hs {}".format(hs[0].size()))
        
        # print("f {}".format(fusion_feature.size()))
        
        fusion_features_list = []  # 创建一个列表用于存储加权后的特征

        for v in range(view):
            # 计算加权后的特征
            fusion_feature = adaptive_weight[v].item() * hs[v]
            fusion_features_list.append(fusion_feature)  # 将加权后的特征添加到列表中

        # 使用 torch.cat 拼接所有视图的特征
        Z = torch.cat(fusion_features_list, dim=1)  # 按列拼接
        # print("Z {}".format(Z.size()))
        
        targ,_ = model.VCF(Z)
        weight = targ ** 2 / targ.sum(0)
        P = (weight.t() / weight.sum(1)).t()
        
        for v in range(view):
            for w in range(v + 1, view):
                loss_list.append(criterion1.forward_class(qs[v], qs[w]))
                sim = torch.exp(torch.mm(hs[v], hs[w].t()))
                sim_probs = sim / sim.sum(1, keepdim=True)

                # pseudo matrix
                Q = torch.mm(qs[v], qs[w].t())
                Q.fill_diagonal_(1)
                pos_mask = (Q >= args.threshold).float()
                Q = Q * pos_mask
                Q = Q / Q.sum(1, keepdims=True)

                local_pse.append(Q)
                loss_contrast_local = - (torch.log(sim_probs + 1e-7) * Q).sum(1)
                loss_contrast_local = loss_contrast_local.mean()

                # loss_list.append(loss_contrast_local)
            # loss_list.append(lmd * F.kl_div(torch.log(P),qs[v]))
            loss_list.append(mse(xs[v], xrs[v]))  # 重建损失
        
        loss = sum(loss_list)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()

    return tot_loss / len(data_loader)



accs = []
nmis = []
aris = []
loss = []

T = 1
for i in range(T):
    print("ROUND:{}".format(i + 1))

    model = Network(view, dims, args, class_num, args.device)
    print(model)
    model = model.to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion1 = Contrastive_loss(args).to(args.device)
    attention_net = MultiHeadAttention(args.cluster_hidden_dim, args.attention_dropout_rate, args.num_heads, args.attn_bias_dim)
    p_net = FeedForwardNetwork(view, args.ffn_size, args.attention_dropout_rate)
    attention_net = attention_net.to(device)
    p_net = p_net.to(device)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    optimizer_atten_net = torch.optim.Adam(attention_net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    optimizer_p_net = torch.optim.Adam(p_net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # init p distribution
    p_sample = np.ones(view)
    weight_history = []
    p_sample = p_sample / sum(p_sample)
    p_sample = torch.FloatTensor(p_sample).cuda()


    # init adaptive weight
    adaptive_weight = np.ones(view)
    adaptive_weight = adaptive_weight / sum(adaptive_weight)
    adaptive_weight = torch.FloatTensor(adaptive_weight).cuda()
    adaptive_weight = adaptive_weight.unsqueeze(1)



    epoch = 1
    print("pretrain:")
    while epoch <= args.mse_epochs:
        loss_epoch = pretrain(epoch)
        
        acc, nmi, pur, h, _ = eva(model, args.device, dataset, view, data_size, class_num,Dataname+ "_training_results.log", eval_h=False)
        # acc, nmi, pur, h, _ = eva(model, args.device, dataset, view, data_size, class_num, eval_h=False)
        epoch += 1
    print("contrastive_train:")
    while epoch <= args.mse_epochs + args.con_epochs:
        loss_epoch = contrastive_train(epoch,args.lmd,p_sample,adaptive_weight)

        acc, nmi, pur, h, q = eva(model, args.device, dataset, view, data_size, class_num,Dataname+ "_training_results.log",eval_h=False)
        # acc, nmi, pur, h, q = eva(model, args.device, dataset, view, data_size, class_num,eval_h=False)
        epoch += 1

torch.save(model.state_dict(), 'CVC-AFWM.pth')
```

---



## 4. 系统实现

### 4.1 FastAPI 接口

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import torch
import numpy as np
import random
import argparse
import os
import uvicorn
from torch.utils.data import Dataset, DataLoader
from network import Network, MultiHeadAttention, FeedForwardNetwork
from metric import valid, eva
from loss import Contrastive_loss
from dataloader import load_data
import warnings

import numpy
import torch
from network import Network,MultiHeadAttention, FeedForwardNetwork
from metric import valid, eva
from torch.utils.data import Dataset
import numpy as np
import argparse
import random
from loss import Contrastive_loss
from dataloader import load_data
from fastapi.responses import RedirectResponse

from sklearn.preprocessing import MinMaxScaler  # 从 sklearn 库中导入 MinMaxScaler，用于特征缩放
from sklearn.cluster import KMeans  # 从 sklearn 库中导入 KMeans 类，用于 K均值聚类
from scipy.optimize import linear_sum_assignment  # 从 scipy 库中导入 linear_sum_assignment 函数，用于线性分配
# from sklearn import manifold
# import matplotlib.pyplot as plt
import scipy.io as sio

import torch.nn as nn  # 导入 PyTorch 的神经网络模块，并将其重命名为 nn
import torch.nn.functional as F  # 导入 PyTorch 的功能性神经网络模块，并将其重命名为 F
import os

import setproctitle
import warnings

# 禁用警告
warnings.filterwarnings("ignore", category=UserWarning, message=".*reduction: 'mean'.*")

# 初始化FastAPI应用
app = FastAPI(
    title="Multi-view Clustering API",
    description="API for contrastive multi-view clustering with FastAPI",
    version="1.0.0"
)

# --- 全局变量 ---
model = None
args = None
dataset = None
view = None
data_size = None
class_num = None
data_loader = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Pydantic请求模型 ---
class TrainRequest(BaseModel):
    dataset_name: str = "MNIST-USPS"
    batch_size: int = 256
    learning_rate: float = 0.0003
    num_cluster: int = 10
    views: int = 3
    mse_epochs: int = 200
    con_epochs: int = 300
    seed: int = 25
    threshold: float = 0.8

class PredictRequest(BaseModel):
    data: List[List[float]]
    view_index: int = 0

class ClusterRequest(BaseModel):
    data_list: List[List[List[float]]]
    return_labels: bool = True

# --- 核心函数 ---
def setup_seed(seed):
    """固定随机种子保证可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def setup_parameters(dataset_name):
    """初始化训练参数"""
    global args
    args = argparse.Namespace()
    
    # 基础参数配置
    args.dataset = dataset_name
    args.batch_size = 256
    args.temperature_c = 0.01
    args.lmd = 1.0
    args.workers = 8
    args.num_heads = 8
    args.hidden_dim = 256
    args.ffn_size = 32
    args.attn_bias_dim = 6
    args.attention_dropout_rate = 0.5
    args.feature_dim = 512
    args.high_feature_dim = 512
    args.cluster_hidden_dim = 512
    args.num_cluster = 10
    args.views = 3
    args.device = device
    args.weight_decay = 0.0
    args.learning_rate = 0.0003
    args.seed = 25
    args.mse_epochs = 200
    args.con_epochs = 300
    args.threshold = 0.8
    
    # 数据集特定参数
    if args.dataset == "MNIST-USPS":
        args.learning_rate = 0.00059
        args.con_epochs = 250
        args.mse_epochs = 50
        args.seed = 2
        args.temperature_c = 0.01
    elif args.dataset == "BDGP":
        args.learning_rate = 0.00002
        args.mse_epochs = 200
        args.con_epochs = 300
        args.seed = 25
        args.temperature_c = 0.6
    # 其他数据集配置...
    
    return args

def initialize_model():
    """初始化模型和数据加载器"""
    global model, dataset, view, data_size, class_num, data_loader, args
    
    # 加载数据集
    dataset, dims, view, data_size, class_num = load_data(args.dataset)
    args.num_cluster = class_num
    
    # 创建数据加载器
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers
    )
    
    # 初始化模型组件
    model = Network(view, dims, args, class_num, device).to(device)
    attention_net = MultiHeadAttention(
        args.cluster_hidden_dim,
        args.attention_dropout_rate,
        args.num_heads,
        args.attn_bias_dim
    ).to(device)
    p_net = FeedForwardNetwork(view, args.ffn_size, args.attention_dropout_rate).to(device)
    
    return model, attention_net, p_net

def pretrain(epoch):
    """预训练阶段"""
    model.train()
    total_loss = 0.0
    criterion = torch.nn.MSELoss()
    
    for batch_idx, (xs, _, _) in enumerate(data_loader):
        # 数据转移到设备
        xs = [x.to(device) for x in xs]
        
        # 前向传播
        optimizer.zero_grad()
        hs, _, xrs, zs = model(xs)
        
        # 计算重建损失
        loss = sum([criterion(xs[v], xrs[v]) for v in range(view)])
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(data_loader)

def contrastive_train(epoch, lmd, p_sample, adaptive_weight):
    """对比学习训练阶段"""
    model.train()
    total_loss = 0.0
    mse_loss = torch.nn.MSELoss()
    
    for batch_idx, (xs, _, _) in enumerate(data_loader):
        # 数据转移到设备
        xs = [x.to(device) for x in xs]
        
        # 前向传播
        optimizer.zero_grad()
        hs, qs, xrs, zs = model(xs)
        
        # 多视图特征融合
        hs_tensor = torch.cat([torch.mean(h, 1).unsqueeze(1) for h in hs], dim=1).transpose(1, 0)
        hs_atten = attention_net(hs_tensor, hs_tensor, hs_tensor)
        p_learn = p_net(p_sample)
        
        # 自适应权重计算
        r = nn.Softmax(dim=0)(hs_atten * p_learn)
        adaptive_weight = r * adaptive_weight
        
        # 对比学习损失
        loss = 0.0
        for v in range(view):
            # 重建损失
            loss += mse_loss(xs[v], xrs[v])
            
            # 视图间对比损失
            for w in range(v + 1, view):
                loss += criterion1.forward_class(qs[v], qs[w])
                
                # 局部对比损失
                sim = torch.exp(torch.mm(hs[v], hs[w].t()))
                sim_probs = sim / sim.sum(1, keepdim=True)
                
                Q = torch.mm(qs[v], qs[w].t())
                Q.fill_diagonal_(1)
                pos_mask = (Q >= args.threshold).float()
                Q = Q * pos_mask / Q.sum(1, keepdims=True)
                
                loss += -(torch.log(sim_probs + 1e-7) * Q).sum(1).mean()
        
        # 反向传播
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(data_loader)

# --- API路由 ---

# 根路径重定向到 /docs
@app.get("/", include_in_schema=False)
async def redirect_to_docs():
    return RedirectResponse(url="/docs")

@app.post("/train", summary="训练多视图聚类模型")
async def train_endpoint(request: TrainRequest):
    """训练接口"""
    try:
        global model, args, optimizer, criterion1, attention_net, p_net
        
        # 1. 参数初始化
        args = setup_parameters(request.dataset_name)
        args.batch_size = request.batch_size
        args.learning_rate = request.learning_rate
        args.num_cluster = request.num_cluster
        args.views = request.views
        args.mse_epochs = request.mse_epochs
        args.con_epochs = request.con_epochs
        args.seed = request.seed
        args.threshold = request.threshold
        
        setup_seed(args.seed)
        
        # 2. 模型初始化
        model, attention_net, p_net = initialize_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        criterion1 = Contrastive_loss(args).to(device)
        
        # 3. 训练循环
        print("开始预训练...")
        for epoch in range(1, 1):
            pretrain(epoch)
        
        print("开始对比学习训练...")
        p_sample = torch.ones(view).to(device) / view
        adaptive_weight = torch.ones(view).to(device) / view
        adaptive_weight = adaptive_weight.unsqueeze(1)
        
        for epoch in range(args.mse_epochs + 1, args.mse_epochs + args.con_epochs + 1):
            contrastive_train(epoch, args.lmd, p_sample, adaptive_weight)
        
        # 4. 保存模型
        torch.save(model.state_dict(), 'CVC-AFWM.pth')
        
        # 5. 评估模型
        acc, nmi, pur, _, _ = eva(model, device, dataset, view, data_size, class_num, 
                                 f"{args.dataset}_results.log", False)
        
        return {
            "status": "success",
            "metrics": {
                "accuracy": float(acc),
                "nmi": float(nmi),
                "purity": float(pur)
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class PredictRequest(BaseModel):
    data_index: int
    return_cluster_assignment: bool = True
    return_features: bool = False
    return_reconstruction: bool = False
    compare_with_ground_truth: bool = True

class PredictResponse(BaseModel):
    data_index: int
    status: str
    cluster_probabilities: List[float]
    cluster_assignment: Optional[int] = None
    hidden_features: Optional[List[float]] = None
    reconstructed_features: Optional[List[float]] = None
    ground_truth_label: Optional[int] = None
    is_correct: Optional[bool] = None


@app.post("/predict", summary="预测单个数据点", response_model=PredictResponse)
async def predict_endpoint(request: PredictRequest):
    try:
        global dataset, model, view, data_size

        if model is None:
            raise HTTPException(status_code=400, detail="请先训练模型")

        print(type(dataset))

        if not hasattr(dataset, '__getitem__'):
            raise HTTPException(status_code=400, detail="数据集未正确加载")

        if request.data_index < 0 or request.data_index >= data_size:
            raise HTTPException(status_code=400, detail=f"数据索引必须在0到{data_size-1}之间")

        # ✅ 这里读取对应样本
        features_list, true_label, _ = dataset[request.data_index]

        inputs = []
        for feature in features_list:
            if not isinstance(feature, torch.Tensor):
                feature = torch.tensor(feature, dtype=torch.float32, device=device)
            feature = feature.unsqueeze(0) if feature.dim() == 1 else feature.unsqueeze(0)
            inputs.append(feature.to(device))

        with torch.no_grad():
            model.eval()
            hs, qs, xrs, _ = model(inputs)

        response_data = {
            "data_index": request.data_index,
            "status": "success",
            "cluster_probabilities": qs[0][0].cpu().numpy().tolist()
        }

        if request.return_cluster_assignment or request.compare_with_ground_truth:
            predicted_label = torch.argmax(qs[0][0]).item()
            if request.return_cluster_assignment:
                response_data["cluster_assignment"] = predicted_label

            if request.compare_with_ground_truth:
                response_data["ground_truth_label"] = int(true_label)
                response_data["is_correct"] = predicted_label == int(true_label)

        if request.return_features:
            response_data["hidden_features"] = hs[0][0].cpu().numpy().tolist()

        if request.return_reconstruction:
            response_data["reconstructed_features"] = xrs[0][0].cpu().numpy().tolist()

        return response_data

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"预测过程中发生错误: {str(e)}")

    
# --- 主程序 ---
if __name__ == "__main__":
    uvicorn.run(
        "api:app",  # 改为模块导入字符串形式
        host="0.0.0.0",
        port=8919,
        log_level="info",
        reload=True  # 保持reload功能
    )
```

---



### 4.2 Web界面功能

![](https://raw.githubusercontent.com/Xpccccc/PicGo/main/data202506161107280.png)

其中`/train`路径的功能是训练数据，`/predict`路径的功能是预测数据点，并且与真实标签对比。

---



### 4.3 部署到docker

创建Dockerfile文件：

```dockerfile
# 使用官方 Python 镜像（带完整 glibc 支持）
FROM python:3.9

# 设置工作目录（绝对路径）
WORKDIR ./

# 安装系统依赖（包括 PyTorch 所需的库）
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libopenblas-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装 Python 依赖（使用 PyTorch 官方源）
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

# 暴露端口
EXPOSE 8919

# 启动命令
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8919"]
```

创建`docker-compose.yml`文件：

```yml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8919:8919"
    volumes:
      - ./app:/app
    environment:
      - PYTHONUNBUFFERED=1
    restart: unless-stopped
```

创建`requirements.txt`文件：

```txt
fastapi==0.112.2
uvicorn==0.20.0
torch==1.12.1
numpy==1.21.2
scikit-learn==0.24.2
scipy==1.7.1
pydantic==1.8.2
h5py==3.7.0
setproctitle==1.2.2
sympy==1.13.3
```

在终端执行命令，创建容器并发布到docker服务器上：

```bash
docker build -t cvc-afwm . --no-cache
```

以后想要执行这个API，只需要使用docker命令：

```bash
docker run -d --name test -p 8919:8919 cvc-afwm
```

----



## 5. 运行效果

**API调用示例**

前提是网页已启动。

`/train`：

```bash
curl -X 'POST' \
  'http://localhost:8919/train' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "dataset_name": "MNIST-USPS",
  "batch_size": 256,
  "learning_rate": 0.00059,
  "num_cluster": 10,
  "views": 3,
  "mse_epochs": 1,
  "con_epochs": 1,
  "seed": 2,
  "threshold": 0.01
}'
```

`/pridict`：

```bash
curl -X 'POST' \
  'http://localhost:8919/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "data_index": 6,
  "return_cluster_assignment": true,
  "return_features": false,
  "return_reconstruction": false,
  "compare_with_ground_truth": true
}'
```

**典型响应**：

`/train`：

```bash
{"status":"success","metrics":{"accuracy":0.3482,"nmi":0.5376550011129111,"purity":0.3482}}
```

`/pridict`：

```bash
{"data_index":6,"status":"success","cluster_probabilities":[0.071192167699337,0.10085364431142807,0.08006078004837036,0.15838401019573212,0.08812542259693146,0.14283229410648346,0.07477735728025436,0.10585857927799225,0.10346271842718124,0.07445304095745087],"cluster_assignment":3,"hidden_features":null,"reconstructed_features":null,"ground_truth_label":3,"is_correct":true
```

**界面操作流程**

1. `/train`路径下，点击`try it out`，便可以调整需要训练的数据集及其参数。然后点击`execute`运行等待返回结果。
1. `/predict`路径下，点击`try it out`，便可以调整需要预测的单个数据点。然后点击`execute`运行等待返回结果。

训练结果：

![](https://raw.githubusercontent.com/Xpccccc/PicGo/main/data202506161351399.png)

预测结果：

![](https://raw.githubusercontent.com/Xpccccc/PicGo/main/data202506161215839.png)

----



## 6. 实验数据

### 公开数据集

| Dataset     | Samples | Views | Clusters | Dimensionality of features     |
| ----------- | ------- | ----- | -------- | ------------------------------ |
| MNIST-USPS  | 5000    | 2     | 10       | [784, 784]                     |
| Hdigit      | 10000   | 2     | 10       | [784, 256]                     |
| Synthetic3d | 600     | 3     | 3        | [3, 3, 3]                      |
| Fashion     | 10000   | 3     | 10       | [784, 784, 784]                |
| Prokaryotic | 551     | 3     | 4        | [438, 3, 393]                  |
| Reuters     | 1200    | 5     | 6        | [2000, 2000, 2000, 2000, 2000] |
| NUS-WIDE    | 2000    | 5     | 31       | [65, 226, 145, 74, 129]        |
| CCV         | 6773    | 3     | 20       | [5000, 5000, 4000]             |
| Caltech-5V  | 1400    | 5     | 7        | [40, 254, 928, 512, 1984]      |
