

import torch
import pandas as pd
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import os
from itertools import chain
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

def load_and_preprocess_data(file_path, fixed_timesteps=1024):
    """
    加载并预处理数据，处理不同特征维度的情况
    :param file_path: 数据文件夹路径
    :param fixed_timesteps: 统一的时间步长
    :param max_features: 最大特征维度（用于填充）
    :return: 统一形状的numpy数组
    """
    all_data = []  # 初始化一个空列表用于存储所有数据
    # 获取文件夹中的所有Excel文件
    file_list = sorted([f for f in os.listdir(file_path) if f.endswith('.xlsx') or f.endswith('.xls')])
    # 遍历文件夹中的每个文件
    for file in file_list:
        file_path_full = os.path.join(file_path, file)
        df = pd.read_excel(file_path_full, header=None)
        # 删除第一行（如果存在）
        if df.shape[0] > 1:
            df = df.drop(0, axis=0)

        data = df.values.astype(np.float32)

        # 统一时间步长
        if data.shape[0] > fixed_timesteps:
            data = data[:fixed_timesteps, :]
        elif data.shape[0] < fixed_timesteps:
            pad_len = fixed_timesteps - data.shape[0]
            data = np.pad(data, ((0, pad_len), (0, 0)), mode='constant')

        # 归一化
        min_vals = np.min(data, axis=0, keepdims=True)
        max_vals = np.max(data, axis=0, keepdims=True)
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1.0  # 避免除以零
        normalized_data = (data - min_vals) / range_vals
        all_data.append(normalized_data)
    all_data = np.stack(all_data, axis=0)

    return all_data

##生成标签
def create_labels(num_samples, num_classes):
    """
    根据样本总数和类别数生成标签。
    :param num_samples: 样本总数
    :param num_classes: 类别总数
    :return: 标签数组 (NumPy)
    """
    # 计算每个类别的样本数
    samples_per_class = num_samples // num_classes

    # 创建标签数组
    labels = np.repeat(np.arange(num_classes), samples_per_class)

    return labels


def process_data(train_file_path, support_file_path, query_file_path, num_classes=10):
    # 加载并预处理train数据
    train_input_data = load_and_preprocess_data(train_file_path)
    train_input_tensor = torch.tensor(train_input_data).float()
    num_train_samples = train_input_data.shape[0]
    train_labels = create_labels(num_train_samples, num_classes=num_classes)
    train_labels_tensor = torch.tensor(train_labels).squeeze()

    # 处理support数据
    support_input_data = load_and_preprocess_data(support_file_path)
    support_input_tensor = torch.tensor(support_input_data).float()
    num_support_samples = support_input_data.shape[0]
    support_labels = create_labels(num_support_samples, num_classes=num_classes)
    support_labels_tensor = torch.tensor(support_labels).squeeze()

    # 处理query数据
    query_input_data = load_and_preprocess_data(query_file_path)
    query_input_tensor = torch.tensor(query_input_data).float()
    num_query_samples = query_input_data.shape[0]
    query_labels = create_labels(num_query_samples, num_classes=num_classes)
    query_labels_tensor = torch.tensor(query_labels).squeeze()

    # 创建DataLoader
    train_input_loader = DataLoader(train_input_tensor, batch_size=20, drop_last=False)
    train_label_loader = DataLoader(train_labels_tensor, batch_size=20, drop_last=False)
    support_input_loader = DataLoader(support_input_tensor, batch_size=20, drop_last=False)
    support_label_loader = DataLoader(support_labels_tensor, batch_size=20, drop_last=False)
    query_input_loader = DataLoader(query_input_tensor, batch_size=20, drop_last=False)
    query_label_loader = DataLoader(query_labels_tensor, batch_size=20, drop_last=False)

    return ( train_input_loader, support_input_loader, query_input_loader,
             train_label_loader, support_label_loader, query_label_loader )


# 对第1组任务 数据进行处理
train_file_1 = r"C:\Users\PC\Desktop\第一篇数据\轴承T_3\model0\random20_50%\train"
support_file_1 = r"C:\Users\PC\Desktop\第一篇数据\轴承T_3\model1\random20_50%\support"
query_file_1 = r"C:\Users\PC\Desktop\第一篇数据\轴承T_3\model1\random20_50%\query"
(train_input_loader_1, support_input_loader_1, query_input_loader_1,
 train_label_loader_1, support_label_loader_1, query_label_loader_1) = process_data(train_file_1, support_file_1, query_file_1)

# 对第2组任务 数据进行处理
train_file_2 = r"C:\Users\PC\Desktop\第一篇数据\轴承T_3\model1\random20_50%\train"
support_file_2 = r"C:\Users\PC\Desktop\第一篇数据\轴承T_3\model2\random20_50%\support"
query_file_2 = r"C:\Users\PC\Desktop\第一篇数据\轴承T_3\model2\random20_50%\query"
(train_input_loader_2,support_input_loader_2, query_input_loader_2,
 train_label_loader_2,support_label_loader_2, query_label_loader_2) = process_data( train_file_2, support_file_2, query_file_2)

# 对第3组任务 数据进行处理
train_file_3 = r"C:\Users\PC\Desktop\第一篇数据\轴承T_3\model2\random20_50%\train"
support_file_3 = r"C:\Users\PC\Desktop\第一篇数据\轴承T_3\model0\random20_50%\support"
query_file_3 = r"C:\Users\PC\Desktop\第一篇数据\轴承T_3\model0\random20_50%\query"
(train_input_loader_3,support_input_loader_3, query_input_loader_3,
 train_label_loader_3,support_label_loader_3, query_label_loader_3) = process_data( train_file_3, support_file_3, query_file_3)

# 将三个DataLoader组组合成列表，每个元素包含对应组的DataLoader
dataloader_groups = [
    (train_input_loader_1, train_label_loader_1, support_input_loader_1, support_label_loader_1, query_input_loader_1, query_label_loader_1),
    (train_input_loader_2, train_label_loader_2, support_input_loader_2, support_label_loader_2, query_input_loader_2, query_label_loader_2),
    (train_input_loader_3, train_label_loader_3, support_input_loader_3, support_label_loader_3, query_input_loader_3, query_label_loader_3)
]

##定义一维卷积
class CNN1D(nn.Module):
    def __init__(self):
        super(CNN1D, self).__init__()
        self.model1 = nn.Sequential(
            # 输入: (batch_size, 3, 1024)
            nn.Conv1d(3, 12, kernel_size=3, stride=2),  # 输出: (batch_size, 12, 511)
            nn.ReLU(),
            nn.MaxPool1d(2),                             # 输出: (batch_size, 12, 255)
            nn.Conv1d(12, 24, kernel_size=3, stride=1),  # 输出: (batch_size, 24, 253)
            nn.ReLU(),
            nn.MaxPool1d(2),                             # 输出: (batch_size, 24, 126)
            nn.Conv1d(24, 48, kernel_size=3, stride=1),  # 输出: (batch_size, 48, 124)
            nn.ReLU(),
            nn.MaxPool1d(5),                             # 输出: (batch_size, 48, 24)
            nn.Flatten(),                                 # 输出: (batch_size, 48*24) = (batch_size, 1152)
            nn.Linear(48 * 24, 30)                        # 输出: (batch_size, 20)
        )
    def forward(self, x):
        # 输入 x 形状: (batch_size, sequence_length, features) = (10, 1024, 3)
        x = x.permute(0, 2, 1)  # 转换为 (10, 3, 1024)
        x = self.model1(x)
        return x
cnn1d = CNN1D()

##定义原型网络
class PrototypeNetwork(nn.Module):
    def __init__(self, feature_dim=30, num_classes=10):
        super(PrototypeNetwork, self).__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes

    def forward(self, train_feature, train_label):
        """
        Args:
            train_feature: (total_train_samples, feature_dim)
            train_label: (total_train_samples,)
        Returns:
            prototypes: (num_classes, feature_dim)
        """
        class_feature = {}
        for n in range(self.num_classes):
            #为每个类别创建一个布尔掩码，过滤属于当前类别的特征，并将其存储在字典中，键为类别索引
            # 确保 train_label 是一维张量
            train_label = train_label.squeeze()
            # 生成布尔掩码
            class_mask = (train_label == n)
            if class_mask.dim() > 1:
                class_mask = class_mask.any(dim=1)
                # 确保 class_mask 是一维的
            class_mask = class_mask.squeeze()
            # 打印形状以确认
            #class_mask shape: torch.Size([100])
            #print(f"class_mask shape: {class_mask.shape}")

            #features=(num_samples_per_class, 10),每个类别有10个样本--〉(10,10)
            features = train_feature[class_mask]
            #第n个类别的特征矩阵，#(num_samples_per_class, 10)
            class_feature[n] = features
        #行数为类别数，列数是特征维度10
        prototypes = torch.zeros(self.num_classes, self.feature_dim, device=train_feature.device)

        for n in range(self.num_classes):

            #features的形状为(num_samples_per_class, feature_dim),
            #features[1]访问该类别的第2个样本的特征向量,形状为(feature_dim,)
            features = class_feature[n]
            num_samples = features.size(0)
            #print(features[1].shape)

            if num_samples == 0:
                prototypes[n] = torch.zeros(self.feature_dim, device=train_feature.device)
                continue

            #计算相似度矩阵A,(num_samples_per_class, num_samples_per_class),第n个类别的相似度矩阵,是对称阵
            #每个类别中每个样本与其余样本的相似度,每个A[i,j]是两两样本的余弦相似度，然后取平均（除以样本数减一）。
            A = torch.zeros(num_samples, num_samples, device=features.device)
            for i in range(num_samples):
                for j in range(num_samples):
                    if i != j:
                        A[i, j] = F.cosine_similarity(features[i].unsqueeze(0), features[j].unsqueeze(0), dim=1)
            A = A / (num_samples - 1) if num_samples > 1 else A
            #print(A.shape)

            # 计算类原型P,第n个类别的原型向量，特征维度为(10,)
            #取A的第k行的均值（即该样本与其他样本的平均相似度），然后乘以特征向量features[k]，最后对所有样本求和并除以样本数
            P = torch.zeros(self.feature_dim, device=features.device)
            for k in range(num_samples):
                P += A[k].mean() * features[k]
            P /= num_samples
            #print(P.shape) #10个类别，有10个原型，原型维度=10=特征维度
            #所有类别的原型矩阵 (10, 10)
            prototypes[n] = P

        return prototypes
prototypeNetwork = PrototypeNetwork()

##定义两步注意力机制，生成重构原型
class TwoStageAttentionModel(nn.Module):
    def __init__(self, d_model):
        super(TwoStageAttentionModel, self).__init__()
        # 第一阶段线性变换
        self.wk1 = nn.Linear(d_model, d_model)  # K 的变换
        self.wv1 = nn.Linear(d_model, d_model)  # V 的变换
        self.wq1 = nn.Linear(d_model, d_model)  # Q 的变换

        # 第二阶段线性变换
        self.wk2 = nn.Linear(d_model, d_model)  # K 的变换
        self.wv2 = nn.Linear(d_model, d_model)  # V 的变换
        self.wq2 = nn.Linear(d_model, d_model)  # Q 的变换

    def forward(self, class_prototypes, support_features, query_features):
        # 第一步：第一阶段注意力机制
        K1 = self.wk1(support_features)  # 形状 (100, 10)
        V1 = self.wv1(support_features)  # 形状 (100, 10)
        Q1 = self.wq1(class_prototypes)  # 形状 (10, 10)

        # 缩放点积注意力
        scores1 = torch.matmul(Q1, K1.T) / torch.sqrt(
            torch.tensor(K1.size(-1), dtype=torch.float32))  # 形状 (10, 100)
        attention_weights1 = F.softmax(scores1, dim=-1)  # 注意力分数，形状 (10, 100)

        # 加权求和，生成混合原型 P，形状为 (10, 10)
        P = torch.matmul(attention_weights1, V1)  # 形状 (10, 10)

        # 第二步：第二阶段注意力机制
        K2 = self.wk2(query_features)  # 形状 (100, 10)
        V2 = self.wv2(query_features)  # 形状 (100, 10)
        Q2 = self.wq2(P)  # 形状 (10, 10)

        # 缩放点积注意力
        scores2 = torch.matmul(Q2, K2.T) / torch.sqrt(
            torch.tensor(K2.size(-1), dtype=torch.float32))  # 形状 (10, 100)
        attention_weights2 = F.softmax(scores2, dim=-1)  # 注意力分数，形状 (10, 100)

        # 加权求和，生成重构原型 P'，形状为 (10, 10)
        P_prime = torch.matmul(attention_weights2, V2)  # 形状 (10, 10)

        return P, P_prime

d_model = 30
 # 实例化模型
model = TwoStageAttentionModel(d_model)

# 定义余弦相似度损失函数
def cosine_similarity_loss(x, y):
    cos_sim = F.cosine_similarity(x, y, dim=-1)  # 计算每个样本的余弦相似度
    loss = 1 - cos_sim.mean()  # 最大化余弦相似度等价于最小化 (1 - 余弦相似度)
    return loss

# 参数设置
epochs = 900
loss_fn = nn.CrossEntropyLoss()
#num_groups = 3  # DataLoader组的数量
# 初始化损失记录
epoch_losses = []  # 记录每轮训练的总损失

# 定义超参数，用于平衡相似性损失和分类损失
lambda1 = 0.3  # 相似性损失权重
lambda2 = 0.7  # 分类损失权重

# 用于存储每一轮中每一组的分类准确性
#group_accuracies_over_epochs = [[] for _ in range(num_groups)]
# 训练循环
for epoch_1 in range(epochs):
    print(f"--------训练阶段第{epoch_1 + 1}轮训练开始---------")

    # 初始化优化器，包含 CNN1D 和 model 的参数
    optimizer = torch.optim.Adam(
        list(cnn1d.parameters()) + list(model.parameters()), lr=1e-3)
    cnn1d.train()

    # 定义交叉熵损失函数
    loss_fn = nn.CrossEntropyLoss()
    # 初始化当前轮的总分类损失
    total_classification_loss = torch.tensor(0.0, requires_grad=True)

    # 初始化一个列表，用于存储当前轮次中每一组的分类准确性
    #group_accuracies = []

    # 遍历每个DataLoader组（共3组）
    for group_idx, (train_input_loader, train_label_loader, support_input_loader, support_label_loader,
                    query_input_loader, query_label_loader)   in enumerate(dataloader_groups):
        print(f"正在处理第 {group_idx + 1} 组DataLoader...")

        # 累积当前组的train特征和标签
        all_train_features = []
        all_train_labels = []
        for train_input, train_label in zip(train_input_loader, train_label_loader):
            # 打印原始输入形状
            #print(f"Input shape before processing: {train_input.shape}")
            train_input = torch.reshape(train_input, (20, 1024, 3)).to(torch.float)
            #print(f"Input shape after reshape: {train_input.shape}")
            train_label = train_label.to(torch.long).squeeze()
            train_feature = cnn1d(train_input)  # 使用共享元学习器提取train特征
            all_train_features.append(train_feature)
            all_train_labels.append(train_label)
        # 合并所有train特征和标签
        all_train_features = torch.cat(all_train_features, dim=0)
        all_train_labels = torch.cat(all_train_labels, dim=0)
        #print(f" {all_train_features}") #torch.Size([100, 64])
        # 计算类原型
        prototypes = prototypeNetwork(all_train_features, all_train_labels)  # (num_classes, 64)
        #print(prototypes) #torch.Size([10, 64])

        # 处理当前组的support数据
        all_support_features = []
        all_support_labels = []
        for support_input, support_label in zip(support_input_loader, support_label_loader):
            #print(f"Input shape before reshape: {support_input.shape}")
            support_input = torch.reshape(support_input, (20, 1024, 3)).to(torch.float)
            #print(f"Input shape after reshape: {support_input.shape}")
            support_feature = cnn1d(support_input)  # 使用共享元学习器提取特征
            support_label = support_label.to(torch.long).squeeze()
            all_support_features.append(support_feature)
            all_support_labels.append(support_label)
        # 合并所有support特征和标签
        all_support_features = torch.cat(all_support_features, dim=0)  # (num_s, 20)
        all_support_labels = torch.cat(all_support_labels, dim=0)  # (num_s,)
        #print(all_support_features.shape,all_support_labels.shape,all_support_features.type)
        # 按类别分组support特征
        support_class_feature = {}
        for n in range(prototypeNetwork.num_classes):
            class_mask = (all_support_labels == n)
            support_class_feature[n] = all_support_features[class_mask]
        #print(support_class_feature[1].shape)

        # 处理当前组的query数据
        all_query_features = []
        all_query_labels = []
        for query_input, query_label in zip(query_input_loader, query_label_loader):
            #print(f"Input shape before reshape: {query_input.shape}")
            query_input = torch.reshape(query_input, (20, 1024, 3)).to(torch.float)
            #print(f"Input shape after reshape: {query_input.shape}")
            query_feature = cnn1d(query_input)  # 使用共享元学习器提取特征
            query_label = query_label.to(torch.long).squeeze()
            all_query_features.append(query_feature)
            all_query_labels.append(query_label)
        # 合并所有Query特征和标签
        all_query_features = torch.cat(all_query_features, dim=0)  # (num_query, 10)
        all_query_labels = torch.cat(all_query_labels, dim=0)  # (num_query,)
        #print(all_query_features.shape,all_query_labels.shape)
        #print(all_query_labels)
        # 按类别分组Query特征
        query_class_feature = {}
        for n in range(prototypeNetwork.num_classes):
            class_mask = (all_query_labels == n)
            query_class_feature[n] = all_query_features[class_mask]
        #print(query_class_feature[1].shape)

        # 准备输入数据
        class_prototypes = prototypes.float()  # 类原型，形状 (10, 10)
        support_features = all_support_features.float()  # 支持特征，形状 (100, 10)
        query_features = all_query_features.float()  # 查询特征，形状 (100, 10)

        all_P = []
        all_P_prime = []

        for class_idx in range(prototypeNetwork.num_classes):  # 遍历每个类别
            # 获取当前类别的支持特征和查询特征
            class_support_features = support_class_feature[class_idx]  # 形状 (num_support_per_class, d_model)
            class_query_features = query_class_feature[class_idx]  # 形状 (num_query_per_class, d_model)
            class_prototype = prototypes[class_idx].float()  # 当前类别的原型，形状 (d_model,)
            # print("Class prototypes shape:", class_prototypes)
            # print("Support features shape:", class_support_features)
            # print("Query features shape:", class_query_features)
            # 前向传播
            P, P_prime = model(class_prototype, class_support_features, class_query_features)

            # 收集所有类别的混合原型和重构原型
            all_P.append(P)
            all_P_prime.append(P_prime)

        # 合并所有类别的混合原型和重构原型
        all_P = torch.stack(all_P, dim=0)  # 形状 (num_classes, d_model)
        all_P_prime = torch.stack(all_P_prime, dim=0)  # 形状 (num_classes, d_model)

        # 计算损失
        loss = cosine_similarity_loss(all_P, all_P_prime)

        # 使用重构原型对查询特征进行分类
        distances = torch.cdist(all_query_features, all_P_prime)
        #print(distances)# (num_query, num_classes)
        temperature = 1  # 温度参数
        logits = -distances / temperature
        #print(logits)

        # 获取预测标签
        predicted_labels = torch.argmax(logits, dim=1)  # (100,)
        # 计算分类损失
        classification_loss = loss_fn(logits, all_query_labels)
        # 将预测标签和真实标签转换为CPU（如果需要）
        predicted_labels = predicted_labels.cpu()
        all_query_labels = all_query_labels.cpu()

        # 计算准确性
        accuracy = accuracy_score(all_query_labels, predicted_labels)
        # 定义总损失：结合相似性损失和分类损失
        total_loss = lambda1 * loss + lambda2 * classification_loss
        # 将当前组的分类准确性添加到对应组的列表中
        #group_accuracies_over_epochs[group_idx].append(accuracy * 100)

        # 反向传播
        optimizer.zero_grad()  # 清空梯度
        total_loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

        # 累积当前组的分类损失
        total_classification_loss = total_classification_loss + classification_loss.item()
        print(f"第 {group_idx + 1} 组的分类损失: {classification_loss.item():.4f}, 相似性损失: {loss.item():.4f}, "
              f"分类准确性: {accuracy * 100:.2f}%")

    # 计算当前轮的平均分类损失
    average_classification_loss = total_classification_loss / len(dataloader_groups)
    # 将当前 epoch 的损失值记录到 epoch_losses 列表中
    epoch_losses.append(average_classification_loss.item())

    print(f"第 {epoch_1 + 1} 轮训练的平均分类损失: {average_classification_loss:.4f}")

# 绘制损失曲线
plt.figure(figsize=(20, 10))  # 设置图像大小
plt.plot(range(1, epochs + 1), epoch_losses, label="training Loss", color="blue", linewidth=2)
# 设置坐标轴标签和标题
plt.xlabel('Epoch', fontdict={'fontsize': 24, 'fontname': 'Times New Roman'})
plt.ylabel('Loss', fontdict={'fontsize': 24, 'fontname': 'Times New Roman'})
plt.title('Loss Curve_3', fontdict={'fontsize': 24, 'fontname': 'Times New Roman'})
# 设置刻度字体大小和字体类型
plt.xticks(fontsize=20, fontname='Times New Roman')
y_min = min(epoch_losses)
y_max = max(epoch_losses)
yticks_values = np.linspace(y_min, y_max, num=6)  # 生成5个均匀分布的刻度值
plt.yticks(ticks=yticks_values, labels=[f'{val:.1f}' for val in yticks_values], fontsize=20, fontname='Times New Roman')

plt.grid(True)
# 显示图例
plt.legend(fontsize=14)
plt.show()















