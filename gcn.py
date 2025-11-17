"""
完整的GCN实现Demo
这个代码会:
1. 创建一个简单的图数据集
2. 定义GCN模型
3. 训练模型
4. 可视化结果

哼!这可是我精心写的代码!你要好好学习哦!
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


# ==================== 第一步: 定义GCN层 ====================
class GCNLayer(nn.Module):
    """
    单层GCN
    这是GCN的核心!你可要看清楚了!
    """

    def __init__(self, in_features, out_features):
        """
        参数:
            in_features: 输入特征维度
            out_features: 输出特征维度
        """
        super(GCNLayer, self).__init__()

        # 这就是那个可学习的权重矩阵W!
        self.linear = nn.Linear(in_features, out_features, bias=False)

        # 可选的bias项
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, X, A_norm):
        """
        前向传播

        参数:
            X: 节点特征矩阵 (N × in_features)
            A_norm: 归一化的邻接矩阵 (N × N)

        返回:
            输出特征矩阵 (N × out_features)
        """
        # 步骤1: 特征变换 X × W
        # 就是我之前讲的那个步骤!你还记得吧?
        XW = self.linear(X)

        # 步骤2: 邻居聚合 A_norm × (X × W)
        # 这里融合了图结构信息!
        out = torch.mm(A_norm, XW)

        # 步骤3: 加上bias
        out = out + self.bias

        return out


# ==================== 第二步: 定义完整的GCN模型 ====================
class GCN(nn.Module):
    """
    两层GCN模型
    这就是我们要训练的网络!
    """

    def __init__(self, in_features, hidden_features, out_features, dropout=0.5):
        """
        参数:
            in_features: 输入特征维度
            hidden_features: 隐藏层维度(你可以调整这个!)
            out_features: 输出维度(类别数)
            dropout: Dropout概率,防止过拟合
        """
        super(GCN, self).__init__()

        # 第一层GCN: in_features → hidden_features
        self.gc1 = GCNLayer(in_features, hidden_features)

        # 第二层GCN: hidden_features → out_features
        self.gc2 = GCNLayer(hidden_features, out_features)

        # Dropout层,防止过拟合
        self.dropout = nn.Dropout(dropout)

    def forward(self, X, A_norm):
        """
        前向传播

        参数:
            X: 节点特征 (N × in_features)
            A_norm: 归一化邻接矩阵 (N × N)

        返回:
            输出logits (N × out_features)
        """
        # 第一层: GCN + ReLU + Dropout
        H = self.gc1(X, A_norm)
        H = F.relu(H)  # 激活函数!你应该懂了吧?
        H = self.dropout(H)

        # 第二层: GCN
        out = self.gc2(H, A_norm)

        # 这里不加激活函数,因为后面会用CrossEntropyLoss
        # CrossEntropyLoss内部会做softmax

        return out


# ==================== 第三步: 准备数据 ====================
def create_karate_club_graph():
    """
    创建一个经典的空手道俱乐部图
    这是GCN领域最著名的小数据集!

    背景故事:
    一个空手道俱乐部因为意见分歧分裂成了两派
    我们要用GCN预测每个成员属于哪一派!
    """
    print("=== 创建空手道俱乐部图 ===")

    # 使用networkx创建图
    G = nx.karate_club_graph()

    # 提取邻接矩阵
    A = nx.to_numpy_array(G)

    # 节点数量
    n_nodes = A.shape[0]
    print(f"节点数量: {n_nodes}")
    print(f"边数量: {G.number_of_edges()}")

    # 为每个节点创建简单的特征
    # 这里我们用one-hot编码(实际应该用更有意义的特征)
    X = np.eye(n_nodes)

    # 标签:每个成员属于哪一派
    # 0派 or 1派
    labels = np.array([G.nodes[i]['club'] == 'Mr. Hi' for i in G.nodes()]).astype(int)

    print(f"第0派人数: {(labels == 0).sum()}")
    print(f"第1派人数: {(labels == 1).sum()}")

    return A, X, labels, G


def normalize_adjacency(A):
    """
    归一化邻接矩阵
    就是我之前讲的那个公式: D^(-1/2) × A × D^(-1/2)
    """
    # 添加自环
    A_hat = A + np.eye(A.shape[0])

    # 计算度矩阵
    D = np.diag(A_hat.sum(axis=1))

    # D^(-1/2)
    D_inv_sqrt = np.linalg.inv(np.sqrt(D))

    # 归一化
    A_norm = D_inv_sqrt @ A_hat @ D_inv_sqrt

    return A_norm


# ==================== 第四步: 训练模型 ====================
def train_gcn():
    """
    训练GCN模型
    这是完整的训练流程!
    """
    print("\n=== 开始训练GCN ===")

    # 1. 准备数据
    A, X, labels, G = create_karate_club_graph()
    A_norm = normalize_adjacency(A)

    # 转换为PyTorch张量
    X = torch.FloatTensor(X)
    A_norm = torch.FloatTensor(A_norm)
    labels = torch.LongTensor(labels)

    # 2. 创建模型
    n_features = X.shape[1]  # 34
    n_hidden = 16  # 隐藏层维度,你可以改这个!
    n_classes = 2  # 2个派别

    model = GCN(in_features=n_features,
                hidden_features=n_hidden,
                out_features=n_classes,
                dropout=0.5)

    print(f"\n模型结构:")
    print(f"输入维度: {n_features}")
    print(f"隐藏层维度: {n_hidden}")
    print(f"输出维度: {n_classes}")

    # 3. 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    # 4. 训练!
    n_epochs = 200
    losses = []
    accuracies = []

    print(f"\n开始训练 {n_epochs} 个epoch...")

    for epoch in range(n_epochs):
        # 前向传播
        model.train()
        optimizer.zero_grad()

        # GCN前向传播
        logits = model(X, A_norm)

        # 计算损失
        loss = criterion(logits, labels)

        # 反向传播
        loss.backward()
        optimizer.step()

        # 计算准确率
        with torch.no_grad():
            pred = logits.argmax(dim=1)
            acc = (pred == labels).float().mean().item()

        losses.append(loss.item())
        accuracies.append(acc)

        # 每20个epoch打印一次
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch + 1:3d} | Loss: {loss.item():.4f} | Acc: {acc:.4f}")

    print("\n训练完成!")
    print(f"最终准确率: {accuracies[-1]:.4f}")

    # 5. 可视化训练过程
    visualize_training(losses, accuracies)

    # 6. 可视化图和预测结果
    model.eval()
    with torch.no_grad():
        logits = model(X, A_norm)
        pred = logits.argmax(dim=1).numpy()

    visualize_graph(G, labels.numpy(), pred)

    return model, losses, accuracies


# ==================== 第五步: 可视化 ====================
def visualize_training(losses, accuracies):
    """
    可视化训练过程
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # 损失曲线
    ax1.plot(losses, color='blue', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training Loss', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # 准确率曲线
    ax2.plot(accuracies, color='green', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Training Accuracy', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gcn_training.png'), dpi=150, bbox_inches='tight')
    print("\n训练曲线已保存!")


def visualize_graph(G, true_labels, pred_labels):
    """
    可视化图结构和预测结果
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # 布局
    pos = nx.spring_layout(G, seed=42)

    # 左图: 真实标签
    colors_true = ['#FF6B6B' if label == 0 else '#4ECDC4' for label in true_labels]
    nx.draw_networkx_nodes(G, pos, node_color=colors_true,
                           node_size=500, alpha=0.8, ax=ax1)
    nx.draw_networkx_edges(G, pos, alpha=0.3, ax=ax1)
    nx.draw_networkx_labels(G, pos, font_size=8, ax=ax1)
    ax1.set_title('True Labels\n(Ground Truth)', fontsize=14, fontweight='bold')
    ax1.axis('off')

    # 右图: 预测标签
    colors_pred = ['#FF6B6B' if label == 0 else '#4ECDC4' for label in pred_labels]
    nx.draw_networkx_nodes(G, pos, node_color=colors_pred,
                           node_size=500, alpha=0.8, ax=ax2)
    nx.draw_networkx_edges(G, pos, alpha=0.3, ax=ax2)
    nx.draw_networkx_labels(G, pos, font_size=8, ax=ax2)
    ax2.set_title('Predicted Labels\n(GCN Prediction)', fontsize=14, fontweight='bold')
    ax2.axis('off')

    # 图例
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#FF6B6B', label='Mr. Hi Club'),
                       Patch(facecolor='#4ECDC4', label='Officer Club')]
    fig.legend(handles=legend_elements, loc='upper center',
               ncol=2, fontsize=12, frameon=True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gcn_vis.png'), dpi=150, bbox_inches='tight')
    print("图可视化已保存!")

output_dir = 'gcn_outputs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# ==================== 第六步: 主函数 ====================
if __name__ == "__main__":
    print("=" * 60)
    print("  GCN (Graph Convolutional Network) Demo")
    print("  哼!这可是我精心准备的代码!")
    print("=" * 60)

    # 设置随机种子,保证结果可复现
    torch.manual_seed(42)
    np.random.seed(42)

    # 训练模型
    model, losses, accuracies = train_gcn()

    print("\n" + "=" * 60)
    print("  训练完成!快去看看生成的图片吧!")
    print("  1. gcn_training.png - 训练曲线")
    print("  2. gcn_visualization.png - 图可视化")
    print("=" * 60)

    # 哼!我的代码写得这么好,你应该能看懂吧?
    # 要是还不懂的话...那就...那就再问我嘛!(小声)
