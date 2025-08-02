import os
import torch
import torch.nn as nn
import numpy as np
import gc
import logging
import datetime
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score  # 新增指标导入
import dgl
from tqdm import tqdm
import pandas as pd
from pre_train import CTDHeteroGraphPretrainDataset, hg_propagate_feat_dgl, HeteroGNNPretrain

# 微调配置
finetune_config = {
    "gpu_id": 0,
    "epochs": 2000,
    "lr": 0.0005,
    "weight_decay": 1e-5,
    "pretrain_model_path": "pretrained_model.pth",
    "node_data_dir": r"",
    "split_edge_dir": "",
    "save_dir": "./finetune_results",
    "freeze_pretrain": True,
    "hidden_dim": 256,
    "in_dim": 64
}


# 提示层
class PromptLayer(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.prompt = nn.Parameter(torch.randn(1, in_dim))
        self.fc = nn.Linear(in_dim * 2, 1)
        self.act = nn.PReLU()

    def forward(self, u_repr, v_repr):
        u_prompted = u_repr + self.prompt
        v_prompted = v_repr + self.prompt

        return self.fc(self.act(torch.cat([u_prompted, v_prompted], dim=1))).squeeze()


# 微调模型
class PromptFinetuneModel(nn.Module):
    def __init__(self, pretrain_model, in_dim):
        super().__init__()
        self.pretrain_model = pretrain_model
        self.prompt_cd = PromptLayer(in_dim)
        self.prompt_cg = PromptLayer(in_dim)
        self.prompt_gd = PromptLayer(in_dim)

    def forward(self, chem_feats, disease_feats, gene_feats):
        return self.pretrain_model(chem_feats, disease_feats, gene_feats)

    def predict(self, chem_repr, disease_repr, gene_repr, task_type, src, dst):
        if task_type == "c-d":
            return self.prompt_cd(chem_repr[src], disease_repr[dst])
        elif task_type == "c-g":
            return self.prompt_cg(chem_repr[src], gene_repr[dst])
        elif task_type == "g-d":
            return self.prompt_gd(gene_repr[src], disease_repr[dst])
        else:
            raise ValueError(f"未知任务类型: {task_type}")


# 下游数据集加载类
class CTDHeteroGraphDownstreamDataset:
    def __init__(self, node_data_dir, split_edge_dir):
        self.node_data_dir = node_data_dir
        self.split_edge_dir = split_edge_dir
        self.load_node_data()
        self.load_downstream_edges()
        self.build_graph()

    def load_node_data(self):
        print("加载节点数据（与预训练一致）...")
        self.chemicals = pd.read_csv(os.path.join(self.node_data_dir, "chemicals_list.tsv"), sep='\t', dtype=str)
        self.diseases = pd.read_csv(os.path.join(self.node_data_dir, "diseases_list.tsv"), sep='\t', dtype=str)
        self.genes = pd.read_csv(os.path.join(self.node_data_dir, "genes_list.tsv"), sep='\t', dtype=str)

        # 节点ID映射保持与预训练一致
        self.chemical_to_idx = {chem: i for i, chem in enumerate(self.chemicals['ChemicalID'])}
        self.disease_to_idx = {dis: i for i, dis in enumerate(self.diseases['DiseaseID'])}
        self.gene_to_idx = {gene: i for i, gene in enumerate(self.genes['GeneID'])}

        print(f"节点统计: 化学物质{len(self.chemicals)}, 疾病{len(self.diseases)}, 基因{len(self.genes)}")

    def load_downstream_edges(self):
        print("加载划分后的下游边数据（30%）...")
        # 下游训练边（10%）
        self.chem_disease_train = pd.read_csv(
            os.path.join(self.split_edge_dir, "chem_disease_down_train.tsv"), sep='\t', dtype=str
        )
        self.chem_gene_train = pd.read_csv(
            os.path.join(self.split_edge_dir, "chem_gene_down_train.tsv"), sep='\t', dtype=str
        )
        self.gene_disease_train = pd.read_csv(
            os.path.join(self.split_edge_dir, "gene_disease_down_train.tsv"), sep='\t', dtype=str
        )

        # 下游验证边（10%）
        self.chem_disease_val = pd.read_csv(
            os.path.join(self.split_edge_dir, "chem_disease_down_val.tsv"), sep='\t', dtype=str
        )
        self.chem_gene_val = pd.read_csv(
            os.path.join(self.split_edge_dir, "chem_gene_down_val.tsv"), sep='\t', dtype=str
        )
        self.gene_disease_val = pd.read_csv(
            os.path.join(self.split_edge_dir, "gene_disease_down_val.tsv"), sep='\t', dtype=str
        )

        # 下游测试边（10%）
        self.chem_disease_test = pd.read_csv(
            os.path.join(self.split_edge_dir, "chem_disease_down_test.tsv"), sep='\t', dtype=str
        )
        self.chem_gene_test = pd.read_csv(
            os.path.join(self.split_edge_dir, "chem_gene_down_test.tsv"), sep='\t', dtype=str
        )
        self.gene_disease_test = pd.read_csv(
            os.path.join(self.split_edge_dir, "gene_disease_down_test.tsv"), sep='\t', dtype=str
        )

        print(f"下游边统计:")
        print(
            f"  化学-疾病: 训练{len(self.chem_disease_train)}, 验证{len(self.chem_disease_val)}, 测试{len(self.chem_disease_test)}")
        print(
            f"  化学-基因: 训练{len(self.chem_gene_train)}, 验证{len(self.chem_gene_val)}, 测试{len(self.chem_gene_test)}")
        print(
            f"  基因-疾病: 训练{len(self.gene_disease_train)}, 验证{len(self.gene_disease_val)}, 测试{len(self.gene_disease_test)}")

    def build_graph(self):
        print("构建下游图（仅含30%下游边）...")
        graph_data = {
            # 化学-疾病边（下游）
            ('chemical', 'c-d', 'disease'): (
                torch.tensor([self.chemical_to_idx[chem] for chem in self.chem_disease_train['ChemicalID']] +
                             [self.chemical_to_idx[chem] for chem in self.chem_disease_val['ChemicalID']] +
                             [self.chemical_to_idx[chem] for chem in self.chem_disease_test['ChemicalID']]),
                torch.tensor([self.disease_to_idx[dis] for dis in self.chem_disease_train['DiseaseID']] +
                             [self.disease_to_idx[dis] for dis in self.chem_disease_val['DiseaseID']] +
                             [self.disease_to_idx[dis] for dis in self.chem_disease_test['DiseaseID']])
            ),
            ('disease', 'd-c', 'chemical'): (
                torch.tensor([self.disease_to_idx[dis] for dis in self.chem_disease_train['DiseaseID']] +
                             [self.disease_to_idx[dis] for dis in self.chem_disease_val['DiseaseID']] +
                             [self.disease_to_idx[dis] for dis in self.chem_disease_test['DiseaseID']]),
                torch.tensor([self.chemical_to_idx[chem] for chem in self.chem_disease_train['ChemicalID']] +
                             [self.chemical_to_idx[chem] for chem in self.chem_disease_val['ChemicalID']] +
                             [self.chemical_to_idx[chem] for chem in self.chem_disease_test['ChemicalID']])
            ),
            # 化学-基因边（下游）
            ('chemical', 'c-g', 'gene'): (
                torch.tensor([self.chemical_to_idx[chem] for chem in self.chem_gene_train['ChemicalID']] +
                             [self.chemical_to_idx[chem] for chem in self.chem_gene_val['ChemicalID']] +
                             [self.chemical_to_idx[chem] for chem in self.chem_gene_test['ChemicalID']]),
                torch.tensor([self.gene_to_idx[gene] for gene in self.chem_gene_train['GeneID']] +
                             [self.gene_to_idx[gene] for gene in self.chem_gene_val['GeneID']] +
                             [self.gene_to_idx[gene] for gene in self.chem_gene_test['GeneID']])
            ),
            ('gene', 'g-c', 'chemical'): (
                torch.tensor([self.gene_to_idx[gene] for gene in self.chem_gene_train['GeneID']] +
                             [self.gene_to_idx[gene] for gene in self.chem_gene_val['GeneID']] +
                             [self.gene_to_idx[gene] for gene in self.chem_gene_test['GeneID']]),
                torch.tensor([self.chemical_to_idx[chem] for chem in self.chem_gene_train['ChemicalID']] +
                             [self.chemical_to_idx[chem] for chem in self.chem_gene_val['ChemicalID']] +
                             [self.chemical_to_idx[chem] for chem in self.chem_gene_test['ChemicalID']])
            ),
            # 基因-疾病边（下游）
            ('gene', 'g-d', 'disease'): (
                torch.tensor([self.gene_to_idx[gene] for gene in self.gene_disease_train['GeneID']] +
                             [self.gene_to_idx[gene] for gene in self.gene_disease_val['GeneID']] +
                             [self.gene_to_idx[gene] for gene in self.gene_disease_test['GeneID']]),
                torch.tensor([self.disease_to_idx[dis] for dis in self.gene_disease_train['DiseaseID']] +
                             [self.disease_to_idx[dis] for dis in self.gene_disease_val['DiseaseID']] +
                             [self.disease_to_idx[dis] for dis in self.gene_disease_test['DiseaseID']])
            ),
            ('disease', 'd-g', 'gene'): (
                torch.tensor([self.disease_to_idx[dis] for dis in self.gene_disease_train['DiseaseID']] +
                             [self.disease_to_idx[dis] for dis in self.gene_disease_val['DiseaseID']] +
                             [self.disease_to_idx[dis] for dis in self.gene_disease_test['DiseaseID']]),
                torch.tensor([self.gene_to_idx[gene] for gene in self.gene_disease_train['GeneID']] +
                             [self.gene_to_idx[gene] for gene in self.gene_disease_val['GeneID']] +
                             [self.gene_to_idx[gene] for gene in self.gene_disease_test['GeneID']])
            )
        }

        self.graph = dgl.heterograph(graph_data)

        # 初始化节点特征（与预训练一致）
        self.graph.nodes['chemical'].data['feat'] = torch.randn(len(self.chemicals), 64)
        self.graph.nodes['disease'].data['feat'] = torch.randn(len(self.diseases), 64)
        self.graph.nodes['gene'].data['feat'] = torch.randn(len(self.genes), 64)

        # 初始化元路径起点特征
        self.graph.nodes['chemical'].data['c'] = self.graph.nodes['chemical'].data['feat'].clone()
        self.graph.nodes['disease'].data['d'] = self.graph.nodes['disease'].data['feat'].clone()
        self.graph.nodes['gene'].data['g'] = self.graph.nodes['gene'].data['feat'].clone()

        print(f"下游图结构: {self.graph}")

    def get_task_edges(self, task_type, split_type):
        if task_type == 'c-d':
            if split_type == 'train':
                edges = self.chem_disease_train
            elif split_type == 'val':
                edges = self.chem_disease_val
            else:
                edges = self.chem_disease_test
            src_col, dst_col = 'ChemicalID', 'DiseaseID'
            src_map, dst_map = self.chemical_to_idx, self.disease_to_idx
        elif task_type == 'c-g':
            if split_type == 'train':
                edges = self.chem_gene_train
            elif split_type == 'val':
                edges = self.chem_gene_val
            else:
                edges = self.chem_gene_test
            src_col, dst_col = 'ChemicalID', 'GeneID'
            src_map, dst_map = self.chemical_to_idx, self.gene_to_idx
        elif task_type == 'g-d':
            if split_type == 'train':
                edges = self.gene_disease_train
            elif split_type == 'val':
                edges = self.gene_disease_val
            else:
                edges = self.gene_disease_test
            src_col, dst_col = 'GeneID', 'DiseaseID'
            src_map, dst_map = self.gene_to_idx, self.disease_to_idx
        else:
            raise ValueError(f"未知任务类型: {task_type}")

        # 转换为节点索引
        src = [src_map[edges.iloc[i][src_col]] for i in range(len(edges))]
        dst = [dst_map[edges.iloc[i][dst_col]] for i in range(len(edges))]
        return torch.tensor(src), torch.tensor(dst)


def finetune():
    # 设备设置
    device = torch.device(f"cuda:{finetune_config['gpu_id']}" if torch.cuda.is_available() else 'cpu')
    os.makedirs(finetune_config['save_dir'], exist_ok=True)

    # 日志配置
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    logfile = logging.FileHandler(os.path.join(finetune_config['save_dir'], f'finetune_{ts}.log'))
    logfile.setFormatter(logging.Formatter('%(asctime)s: %(message)s'))
    logger.addHandler(logfile)
    logger.addHandler(logging.StreamHandler())

    # 加载下游数据集
    dataset = CTDHeteroGraphDownstreamDataset(
        node_data_dir=finetune_config['node_data_dir'],
        split_edge_dir=finetune_config['split_edge_dir']
    )
    g = dataset.graph.to(device)
    logger.info(f"下游微调图结构: {g}")

    # 元路径传播
    metapaths_per_type = [
        'c', 'cd', 'cg', 'cdc', 'cgc', 'cdg',
        'd', 'dc', 'dg', 'dcd', 'dcg', 'dgd',
        'g', 'gc', 'gd', 'gcg', 'gcd', 'gdg'
    ]
    g = hg_propagate_feat_dgl(
        g, num_hops=2,
        max_length=3,
        metapaths_per_type=metapaths_per_type
    )

    # 提取元路径特征
    chem_feats = {k: v.to(device) for k, v in g.nodes['chemical'].data.items() if k.startswith('c')}
    disease_feats = {k: v.to(device) for k, v in g.nodes['disease'].data.items() if k.startswith('d')}
    gene_feats = {k: v.to(device) for k, v in g.nodes['gene'].data.items() if k.startswith('g')}
    num_metapaths = min(len(chem_feats), len(disease_feats), len(gene_feats))
    chem_feats = dict(list(chem_feats.items())[:num_metapaths])
    disease_feats = dict(list(disease_feats.items())[:num_metapaths])
    gene_feats = dict(list(gene_feats.items())[:num_metapaths])
    logger.info(f"元路径数量: {num_metapaths}")

    # 加载预训练模型
    pretrain_model = HeteroGNNPretrain(
        in_dim=finetune_config['in_dim'],
        hidden_dim=finetune_config['hidden_dim'],
        num_metapaths=num_metapaths
    ).to(device)
    pretrain_model.load_state_dict(torch.load(finetune_config['pretrain_model_path'], map_location=device))
    logger.info("预训练模型加载完成")

    # 冻结预训练参数
    if finetune_config['freeze_pretrain']:
        for param in pretrain_model.parameters():
            param.requires_grad = False
        logger.info("已冻结预训练模型参数，仅训练提示层")

    # 初始化微调模型
    model = PromptFinetuneModel(pretrain_model, finetune_config['hidden_dim']).to(device)
    logger.info(f"微调模型可训练参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # 准备下游任务数据
    tasks = ['c-d', 'c-g', 'g-d']
    task_data = {}
    for task_type in tasks:
        # 加载训练/验证/测试边
        train_src, train_dst = dataset.get_task_edges(task_type, 'train')
        val_src, val_dst = dataset.get_task_edges(task_type, 'val')
        test_src, test_dst = dataset.get_task_edges(task_type, 'test')

        task_data[task_type] = {
            'train': {'src': train_src.to(device), 'dst': train_dst.to(device)},
            'val': {'src': val_src.to(device), 'dst': val_dst.to(device)},
            'test': {'src': test_src.to(device), 'dst': test_dst.to(device)}
        }
        logger.info(f"任务 {task_type} 数据: 训练{len(train_src)}, 验证{len(val_src)}, 测试{len(test_src)}")

    # 优化器和损失函数
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=finetune_config['lr'],
        weight_decay=finetune_config['weight_decay']
    )
    criterion = nn.BCEWithLogitsLoss()

    # 微调训练循环
    best_val_auc = {t: 0.0 for t in tasks}  # 用AUC作为最佳模型选择指标
    logger.info("开始微调...")
    for epoch in range(finetune_config['epochs']):
        model.train()
        optimizer.zero_grad()
        total_loss = 0.0

        # 生成节点表示
        chem_repr, disease_repr, gene_repr = model(chem_feats, disease_feats, gene_feats)

        # 计算每个任务的训练损失
        for task_type in tasks:
            data = task_data[task_type]['train']
            src, dst = data['src'], data['dst']
            if len(src) == 0:
                continue

            # 正样本
            pos_score = model.predict(chem_repr, disease_repr, gene_repr, task_type, src, dst)
            pos_label = torch.ones_like(pos_score, device=device)

            # 负样本
            neg_samples = len(src)
            if task_type == "c-d":
                neg_src = torch.randint(0, len(chem_repr), (neg_samples,), device=device)
                neg_dst = torch.randint(0, len(disease_repr), (neg_samples,), device=device)
            elif task_type == "c-g":
                neg_src = torch.randint(0, len(chem_repr), (neg_samples,), device=device)
                neg_dst = torch.randint(0, len(gene_repr), (neg_samples,), device=device)
            else:  # g-d
                neg_src = torch.randint(0, len(gene_repr), (neg_samples,), device=device)
                neg_dst = torch.randint(0, len(disease_repr), (neg_samples,), device=device)

            neg_score = model.predict(chem_repr, disease_repr, gene_repr, task_type, neg_src, neg_dst)
            neg_label = torch.zeros_like(neg_score, device=device)

            # 累加损失
            all_scores = torch.cat([pos_score, neg_score])
            all_labels = torch.cat([pos_label, neg_label])
            task_loss = criterion(all_scores, all_labels)
            total_loss += task_loss

        # 反向传播
        total_loss.backward()
        optimizer.step()

        # 验证（每5轮）
        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                chem_repr_val, disease_repr_val, gene_repr_val = model(chem_feats, disease_feats, gene_feats)
                val_metrics = {}

                for task_type in tasks:
                    data = task_data[task_type]['val']
                    src, dst = data['src'], data['dst']
                    if len(src) == 0:
                        continue

                    # 验证集正样本
                    pos_score = model.predict(chem_repr_val, disease_repr_val, gene_repr_val, task_type, src, dst)
                    pos_label = torch.ones_like(pos_score)

                    # 验证集负样本
                    neg_samples = len(src)
                    if task_type == "c-d":
                        neg_src = torch.randint(0, len(chem_repr_val), (neg_samples,), device=device)
                        neg_dst = torch.randint(0, len(disease_repr_val), (neg_samples,), device=device)
                    elif task_type == "c-g":
                        neg_src = torch.randint(0, len(chem_repr_val), (neg_samples,), device=device)
                        neg_dst = torch.randint(0, len(gene_repr_val), (neg_samples,), device=device)
                    else:
                        neg_src = torch.randint(0, len(gene_repr_val), (neg_samples,), device=device)
                        neg_dst = torch.randint(0, len(disease_repr_val), (neg_samples,), device=device)

                    neg_score = model.predict(chem_repr_val, disease_repr_val, gene_repr_val, task_type, neg_src,
                                              neg_dst)
                    neg_label = torch.zeros_like(neg_score)

                    # 计算指标
                    all_scores = torch.cat([pos_score, neg_score]).cpu().numpy()
                    all_labels = torch.cat([pos_label, neg_label]).cpu().numpy()
                    unique_labels = np.unique(all_labels)

                    # 处理标签类别单一的情况
                    if len(unique_labels) <= 1:
                        auc = 0.5
                        aupr = 0.5
                        precision = 0.5
                        recall = 0.5
                        f1 = 0.5
                    else:
                        # 计算AUC和AUPR
                        auc = roc_auc_score(all_labels, all_scores)
                        aupr = average_precision_score(all_labels, all_scores)

                        # 转换为二分类预测（阈值0.5）
                        y_pred = (all_scores >= 0.5).astype(int)

                        # 计算Precision、Recall、F1
                        precision = precision_score(all_labels, y_pred)
                        recall = recall_score(all_labels, y_pred)
                        f1 = f1_score(all_labels, y_pred)

                    val_metrics[task_type] = (auc, aupr, precision, recall, f1)

                    # 保存最佳模型（仍以AUC为标准）
                    if auc > best_val_auc[task_type]:
                        best_val_auc[task_type] = auc
                        torch.save(model.state_dict(),
                                   os.path.join(finetune_config['save_dir'], f'best_{task_type}.pt'))

            # 打印日志
            logger.info(f"Epoch {epoch + 1}/{finetune_config['epochs']} | 总损失: {total_loss.item():.4f}")
            for task_type, (auc, aupr, precision, recall, f1) in val_metrics.items():
                logger.info(
                    f"  任务 {task_type} - "
                    f"AUC: {auc:.4f}, AUPR: {aupr:.4f}, "
                    f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f} "
                    f"(最佳AUC: {best_val_auc[task_type]:.4f})"
                )

        # 清理内存
        del chem_repr, disease_repr, gene_repr
        gc.collect()

    # 测试最佳模型
    logger.info("\n开始测试最佳模型...")
    for task_type in tasks:
        model.load_state_dict(
            torch.load(os.path.join(finetune_config['save_dir'], f'best_{task_type}.pt'), map_location=device))
        model.eval()
        with torch.no_grad():
            chem_repr_test, disease_repr_test, gene_repr_test = model(chem_feats, disease_feats, gene_feats)
            data = task_data[task_type]['test']
            src, dst = data['src'], data['dst']
            if len(src) == 0:
                continue

            # 测试集正样本
            pos_score = model.predict(chem_repr_test, disease_repr_test, gene_repr_test, task_type, src, dst)
            pos_label = torch.ones_like(pos_score)

            # 测试集负样本
            neg_samples = len(src)
            if task_type == "c-d":
                neg_src = torch.randint(0, len(chem_repr_test), (neg_samples,), device=device)
                neg_dst = torch.randint(0, len(disease_repr_test), (neg_samples,), device=device)
            elif task_type == "c-g":
                neg_src = torch.randint(0, len(chem_repr_test), (neg_samples,), device=device)
                neg_dst = torch.randint(0, len(gene_repr_test), (neg_samples,), device=device)
            else:
                neg_src = torch.randint(0, len(gene_repr_test), (neg_samples,), device=device)
                neg_dst = torch.randint(0, len(disease_repr_test), (neg_samples,), device=device)

            neg_score = model.predict(chem_repr_test, disease_repr_test, gene_repr_test, task_type, neg_src, neg_dst)
            neg_label = torch.zeros_like(neg_score)

            # 计算测试指标
            all_scores = torch.cat([pos_score, neg_score]).cpu().numpy()
            all_labels = torch.cat([pos_label, neg_label]).cpu().numpy()
            unique_labels = np.unique(all_labels)

            if len(unique_labels) <= 1:
                test_auc = 0.5
                test_aupr = 0.5
                test_precision = 0.5
                test_recall = 0.5
                test_f1 = 0.5
            else:
                test_auc = roc_auc_score(all_labels, all_scores)
                test_aupr = average_precision_score(all_labels, all_scores)
                y_pred = (all_scores >= 0.5).astype(int)
                test_precision = precision_score(all_labels, y_pred)
                test_recall = recall_score(all_labels, y_pred)
                test_f1 = f1_score(all_labels, y_pred)

            logger.info(
                f"任务 {task_type} 测试结果 - "
                f"AUC: {test_auc:.4f}, AUPR: {test_aupr:.4f}, "
                f"Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}"
            )

    logger.info("微调完成!")


if __name__ == "__main__":
    finetune()