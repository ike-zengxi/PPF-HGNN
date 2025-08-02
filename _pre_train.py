import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
import numpy as np
import gc
from tqdm import tqdm
import pandas as pd

# 设置随机种子确保可复现性
torch.manual_seed(42)
np.random.seed(42)


# -------------------------- 数据集加载 --------------------------
class CTDHeteroGraphDataset:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.load_data()
        self.build_graph()

    def load_data(self):
        print("加载节点数据...")
        self.chemicals = pd.read_csv(os.path.join(self.data_dir, "chemicals_list.tsv"), sep='\t')
        self.diseases = pd.read_csv(os.path.join(self.data_dir, "diseases_list.tsv"), sep='\t')
        self.genes = pd.read_csv(os.path.join(self.data_dir, "genes_list.tsv"), sep='\t')

        print("加载边数据...")
        self.chem_disease_edges = pd.read_csv(os.path.join(self.data_dir, "chemicals_diseases.tsv"), sep='\t')
        self.chem_gene_edges = pd.read_csv(os.path.join(self.data_dir, "chemicals_genes.tsv"), sep='\t')
        self.gene_disease_edges = pd.read_csv(os.path.join(self.data_dir, "genes_diseases.tsv"), sep='\t')

        # 创建ID到索引的映射
        self.chemical_to_idx = {chem: i for i, chem in enumerate(self.chemicals['ChemicalID'])}
        self.disease_to_idx = {dis: i for i, dis in enumerate(self.diseases['DiseaseID'])}
        self.gene_to_idx = {gene: i for i, gene in enumerate(self.genes['GeneID'])}

        print(f"数据统计:")
        print(f"  化学物质: {len(self.chemicals)}")
        print(f"  疾病: {len(self.diseases)}")
        print(f"  基因: {len(self.genes)}")
        print(f"  化学-疾病关系: {len(self.chem_disease_edges)}")
        print(f"  化学-基因关系: {len(self.chem_gene_edges)}")
        print(f"  基因-疾病关系: {len(self.gene_disease_edges)}")

    def build_graph(self):
        print("构建异质图...")

        # 定义图的边类型
        graph_data = {
            ('chemical', 'c-d', 'disease'): (
                torch.tensor([self.chemical_to_idx[chem] for chem in self.chem_disease_edges['ChemicalID']]),
                torch.tensor([self.disease_to_idx[dis] for dis in self.chem_disease_edges['DiseaseID']])
            ),
            ('disease', 'd-c', 'chemical'): (
                torch.tensor([self.disease_to_idx[dis] for dis in self.chem_disease_edges['DiseaseID']]),
                torch.tensor([self.chemical_to_idx[chem] for chem in self.chem_disease_edges['ChemicalID']])
            ),
            ('chemical', 'c-g', 'gene'): (
                torch.tensor([self.chemical_to_idx[chem] for chem in self.chem_gene_edges['ChemicalID']]),
                torch.tensor([self.gene_to_idx[gene] for gene in self.chem_gene_edges['GeneID']])
            ),
            ('gene', 'g-c', 'chemical'): (
                torch.tensor([self.gene_to_idx[gene] for gene in self.chem_gene_edges['GeneID']]),
                torch.tensor([self.chemical_to_idx[chem] for chem in self.chem_gene_edges['ChemicalID']])
            ),
            ('gene', 'g-d', 'disease'): (
                torch.tensor([self.gene_to_idx[gene] for gene in self.gene_disease_edges['GeneID']]),
                torch.tensor([self.disease_to_idx[dis] for dis in self.gene_disease_edges['DiseaseID']])
            ),
            ('disease', 'd-g', 'gene'): (
                torch.tensor([self.disease_to_idx[dis] for dis in self.gene_disease_edges['DiseaseID']]),
                torch.tensor([self.gene_to_idx[gene] for gene in self.gene_disease_edges['GeneID']])
            )
        }

        self.graph = dgl.heterograph(graph_data)

        # 初始化节点特征
        self.graph.nodes['chemical'].data['feat'] = torch.randn(len(self.chemicals), 64)
        self.graph.nodes['disease'].data['feat'] = torch.randn(len(self.diseases), 64)
        self.graph.nodes['gene'].data['feat'] = torch.randn(len(self.genes), 64)

        # 初始化元路径起点特征
        self.graph.nodes['chemical'].data['c'] = self.graph.nodes['chemical'].data['feat'].clone()
        self.graph.nodes['disease'].data['d'] = self.graph.nodes['disease'].data['feat'].clone()
        self.graph.nodes['gene'].data['g'] = self.graph.nodes['gene'].data['feat'].clone()

        print(f"图结构: {self.graph}")


# -------------------------- 元路径特征传播 --------------------------
def hg_propagate_feat_dgl(g, num_hops, max_length, metapaths_per_type, echo=False):
    all_ntypes = g.ntypes

    for hop in range(1, max_length):
        reserve_heads = {}
        for ntype in all_ntypes:
            ntype_metapaths = [mp for mp in metapaths_per_type if mp.startswith(ntype)]
            reserve_heads[ntype] = [mp[:hop] for mp in ntype_metapaths if len(mp) > hop]

        # 沿所有边类型传播特征
        for etype in g.etypes:
            stype, _, dtype = g.to_canonical_etype(etype)
            for k in list(g.nodes[stype].data.keys()):
                if len(k) == hop:
                    current_dst_name = f'{dtype}{k}'
                    if (hop == num_hops and current_dst_name not in reserve_heads.get(dtype, [])) or \
                            (hop > num_hops and current_dst_name not in reserve_heads.get(dtype, [])):
                        continue

                    if echo:
                        print(f"传播 {k} 沿 {etype} 到 {current_dst_name}")

                    g.update_all(
                        fn.copy_u(k, 'm'),
                        fn.mean('m', current_dst_name),
                        etype=etype
                    )

        # 清理特征
        for ntype in all_ntypes:
            keep_features = [k for k in reserve_heads.get(ntype, [])]
            keep_features.extend([k for k in g.nodes[ntype].data.keys() if len(k) <= 1 or k.startswith(ntype)])
            remove_features = [k for k in g.nodes[ntype].data.keys() if k not in keep_features]
            for k in remove_features:
                del g.nodes[ntype].data[k]
            if echo and remove_features:
                print(f"移除 {ntype} 的特征: {remove_features}")

        gc.collect()
        if echo:
            print(f"--- 完成 hop={hop} 传播 ---")

    return g


# -------------------------- 预训练模型 --------------------------
class HeteroGNNPretrain(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_metapaths, dropout=0.3):
        super().__init__()
        # 共享特征转换层
        self.conv = nn.Conv1d(in_dim, hidden_dim, kernel_size=1)
        self.norm = nn.LayerNorm(hidden_dim)
        self.shared_fc = nn.Linear(hidden_dim * num_metapaths, hidden_dim)

        # 节点表示输出层
        self.chem_fc = nn.Linear(hidden_dim, hidden_dim)
        self.disease_fc = nn.Linear(hidden_dim, hidden_dim)
        self.gene_fc = nn.Linear(hidden_dim, hidden_dim)

        # 自监督任务头：特征重建（输出维度与输入特征一致）
        self.reconstruct_head = nn.Linear(hidden_dim, in_dim)

        # 自监督任务头：边预测
        self.edge_pred_head = nn.Linear(hidden_dim * 2, 1)

        self.dropout = nn.Dropout(dropout)
        self.act = nn.PReLU()

    def forward(self, chem_feats, disease_feats, gene_feats):
        # 处理各类节点特征
        chem_feat = self._process_features(chem_feats)
        disease_feat = self._process_features(disease_feats)
        gene_feat = self._process_features(gene_feats)

        # 生成节点表示
        chem_repr = self.chem_fc(chem_feat)
        disease_repr = self.disease_fc(disease_feat)
        gene_repr = self.gene_fc(gene_feat)

        return chem_repr, disease_repr, gene_repr

    def _process_features(self, meta_feats):
        # 处理元路径特征
        feats = torch.stack(list(meta_feats.values()), dim=1)  # [B, N, in_dim]
        B, N, C = feats.shape

        x = feats.transpose(1, 2)  # [B, C, N]
        x = self.conv(x)  # [B, hidden_dim, N]
        x = self.norm(x.transpose(1, 2)).transpose(1, 2)  # [B, hidden_dim, N]
        x = self.act(x)
        x = self.dropout(x)

        x = x.transpose(1, 2).reshape(B, -1)  # [B, hidden_dim*N]
        x = self.shared_fc(x)
        x = self.act(x)
        x = self.dropout(x)

        return x

    def reconstruct(self, node_repr):
        """特征重建任务：从节点表示重建原始特征"""
        return self.reconstruct_head(node_repr)

    def predict_edge(self, src_repr, dst_repr):
        """边预测任务：预测两个节点间是否存在边"""
        concat = torch.cat([src_repr, dst_repr], dim=1)
        return self.edge_pred_head(concat).squeeze()


# -------------------------- 预训练函数 --------------------------
def pretrain(data_dir, save_path, num_hops=2, hidden_dim=64, epochs=200, lr=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载数据集
    dataset = CTDHeteroGraphDataset(data_dir)
    g = dataset.graph.clone().to(device)

    # 定义元路径（根据实际元路径数量调整）
    metapaths_per_type = [
        # 化学起点元路径
        'c', 'cd', 'cg',
        # 疾病起点元路径
        'd', 'dc', 'dg',
        # 基因起点元路径
        'g', 'gc', 'gd'
    ]
    print(f"元路径集合: {metapaths_per_type}")
    print(f"每种节点类型的元路径数量: {len(metapaths_per_type) // 3}")

    # 特征传播
    print("开始元路径特征传播...")
    g = hg_propagate_feat_dgl(
        g, num_hops=num_hops,
        max_length=num_hops + 1,
        metapaths_per_type=metapaths_per_type,
        echo=False
    )

    # 提取元路径特征
    chem_feats = {k: v.to(device) for k, v in g.nodes['chemical'].data.items() if k.startswith('c')}
    disease_feats = {k: v.to(device) for k, v in g.nodes['disease'].data.items() if k.startswith('d')}
    gene_feats = {k: v.to(device) for k, v in g.nodes['gene'].data.items() if k.startswith('g')}

    # 确保元路径数量一致
    num_metapaths = min(len(chem_feats), len(disease_feats), len(gene_feats))
    chem_feats = dict(list(chem_feats.items())[:num_metapaths])
    disease_feats = dict(list(disease_feats.items())[:num_metapaths])
    gene_feats = dict(list(gene_feats.items())[:num_metapaths])
    print(f"每种节点类型保留的元路径数量: {num_metapaths}")

    # 准备自监督任务数据
    # 1. 特征重建任务：随机掩码部分节点特征
    mask_ratio = 0.15
    # 为每种节点类型生成相同的掩码（关键修复：确保掩码一致）
    chem_mask = torch.rand(len(next(iter(chem_feats.values()))), device=device) < mask_ratio
    disease_mask = torch.rand(len(next(iter(disease_feats.values()))), device=device) < mask_ratio
    gene_mask = torch.rand(len(next(iter(gene_feats.values()))), device=device) < mask_ratio

    # 2. 边预测任务：使用所有类型的边
    edge_types = [
        ('chemical', 'c-d', 'disease'),
        ('chemical', 'c-g', 'gene'),
        ('gene', 'g-d', 'disease')
    ]
    edge_data = {}
    for etype in edge_types:
        src, dst = g.edges(etype=etype)
        edge_data[etype] = {
            'src': src.to(device),
            'dst': dst.to(device)
        }

    # 初始化模型
    model = HeteroGNNPretrain(
        in_dim=64,
        hidden_dim=hidden_dim,
        num_metapaths=num_metapaths
    ).to(device)

    # 优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    recon_criterion = nn.MSELoss()  # 特征重建损失
    edge_criterion = nn.BCEWithLogitsLoss()  # 边预测损失

    # 预训练循环
    best_loss = float('inf')
    print("开始预训练...")
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # 生成节点表示
        chem_repr, disease_repr, gene_repr = model(chem_feats, disease_feats, gene_feats)

        # 1. 特征重建损失（自监督任务1）
        # 关键修复：分别计算每个元路径的重建损失，再取平均（确保维度匹配）
        recon_loss = 0.0

        # 化学物质特征重建（每个元路径单独计算）
        chem_recon = model.reconstruct(chem_repr[chem_mask])
        for feat in chem_feats.values():
            recon_loss += recon_criterion(chem_recon, feat[chem_mask])
        recon_loss /= len(chem_feats)  # 平均化学物质的元路径损失

        # 疾病特征重建
        disease_recon = model.reconstruct(disease_repr[disease_mask])
        for feat in disease_feats.values():
            recon_loss += recon_criterion(disease_recon, feat[disease_mask])
        recon_loss /= len(disease_feats)  # 平均疾病的元路径损失

        # 基因特征重建
        gene_recon = model.reconstruct(gene_repr[gene_mask])
        for feat in gene_feats.values():
            recon_loss += recon_criterion(gene_recon, feat[gene_mask])
        recon_loss /= len(gene_feats)  # 平均基因的元路径损失

        # 最终重建损失：三种节点类型的平均
        recon_loss /= 3

        # 2. 边预测损失（自监督任务2）
        edge_loss = 0.0
        for etype in edge_types:
            data = edge_data[etype]
            src = data['src']
            dst = data['dst']

            # 正样本
            if etype == ('chemical', 'c-d', 'disease'):
                pos_scores = model.predict_edge(chem_repr[src], disease_repr[dst])
            elif etype == ('chemical', 'c-g', 'gene'):
                pos_scores = model.predict_edge(chem_repr[src], gene_repr[dst])
            else:  # gene-disease
                pos_scores = model.predict_edge(gene_repr[src], disease_repr[dst])
            pos_labels = torch.ones_like(pos_scores, device=device)

            # 负样本（随机采样）
            neg_samples = len(src) // 2
            if etype == ('chemical', 'c-d', 'disease'):
                neg_src = torch.randint(0, len(chem_repr), (neg_samples,), device=device)
                neg_dst = torch.randint(0, len(disease_repr), (neg_samples,), device=device)
            elif etype == ('chemical', 'c-g', 'gene'):
                neg_src = torch.randint(0, len(chem_repr), (neg_samples,), device=device)
                neg_dst = torch.randint(0, len(gene_repr), (neg_samples,), device=device)
            else:
                neg_src = torch.randint(0, len(gene_repr), (neg_samples,), device=device)
                neg_dst = torch.randint(0, len(disease_repr), (neg_samples,), device=device)

            # 计算负样本得分
            if etype == ('chemical', 'c-d', 'disease'):
                neg_scores = model.predict_edge(chem_repr[neg_src], disease_repr[neg_dst])
            elif etype == ('chemical', 'c-g', 'gene'):
                neg_scores = model.predict_edge(chem_repr[neg_src], gene_repr[neg_dst])
            else:
                neg_scores = model.predict_edge(gene_repr[neg_src], disease_repr[neg_dst])
            neg_labels = torch.zeros_like(neg_scores, device=device)

            # 计算当前边类型的损失
            all_scores = torch.cat([pos_scores, neg_scores])
            all_labels = torch.cat([pos_labels, neg_labels])
            edge_loss += edge_criterion(all_scores, all_labels)

        edge_loss /= len(edge_types)  # 平均三种边类型的损失

        # # 总损失：平衡两个自监督任务
        total_loss = 0.7 * recon_loss + 0.3 * edge_loss
        # 反向传播
        total_loss.backward()
        optimizer.step()

        # 学习率调整
        scheduler.step(total_loss)

        # 保存最佳模型
        if total_loss < best_loss:
            best_loss = total_loss
            torch.save(model.state_dict(), save_path)
            improved = "*"
        else:
            improved = ""

        # 打印训练信息
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs} | 总损失: {total_loss.item():.4f} "
                  f"(重建损失: {recon_loss.item():.4f}, 边预测损失: {edge_loss.item():.4f}) {improved}")

        # 清理内存
        del chem_repr, disease_repr, gene_repr
        gc.collect()

    print(f"预训练完成，最佳模型保存至: {save_path}")


# -------------------------- 主函数 --------------------------
if __name__ == "__main__":
    data_dir = r""  # 替换为你的数据目录
    save_path = "pretrained_model.pth"  # 预训练模型保存路径
    pretrain(
        data_dir=data_dir,
        save_path=save_path,
        num_hops=2,
        hidden_dim=256,
        epochs=2000,
        lr=0.001
    )
