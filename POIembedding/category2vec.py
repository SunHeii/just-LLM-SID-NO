# category2vec.py (恢复为标准的 64 维版)
# model = SentenceTransformer("/home/mysjz/mywork/Models/all-MiniLM-L6-v2")
# import os
# import pandas as pd
# import numpy as np
# from sentence_transformers import SentenceTransformer
# from sklearn.decomposition import PCA
# import argparse
# import pickle
#
# def category2vec(csv_path, output_dir, model_name="all-MiniLM-L6-v2", n_components=64, category_column="category"):
#     # 确保输出目录存在
#     os.makedirs(output_dir, exist_ok=True)
#
#     # 1. 读取基础元数据 CSV
#     print(f"读取元数据文件: {csv_path}")
#     df = pd.read_csv(csv_path,  encoding='utf-8')
#
#     # 容错处理：填充可能为空的分类
#     df[category_column] = df[category_column].fillna('Unknown')
#     categories = df[category_column].unique().tolist()
#     print(f"共发现 {len(categories)} 种独特的 POI 类别")
#
#     # 2. 加载轻量级句子转化大模型
#     print(f"正在加载语义提取模型: {model_name}")
#     # 使用你本地的模型路径
#     model = SentenceTransformer("/home/mysjz/mywork/Models/all-MiniLM-L6-v2")
#
#     # 3. 提取高维稠密文本特征
#     print("正在生成品类文本 Embedding...")
#     embeddings = model.encode(categories, show_progress_bar=True)
#     print(f"初始提取向量维度: {embeddings.shape}")
#
#     # 4. 【恢复修改】：应用 PCA 降维至标准的 64 维
#     if n_components is not None and n_components < embeddings.shape[1]:
#         print(f"正在执行 PCA 降维，目标维度: {n_components} ...")
#         pca = PCA(n_components=n_components)
#         embeddings_reduced = pca.fit_transform(embeddings)
#         print(f"降维后最终向量维度: {embeddings_reduced.shape}")
#         final_embeddings = embeddings_reduced
#     else:
#         final_embeddings = embeddings
#
#     # 构建品类字符串到向量的字典映射
#     category_to_embedding = dict(zip(categories, final_embeddings))
#
#     # 5. 落盘保存
#     npy_path = os.path.join(output_dir, "category_embeddings.npy")
#     np.save(npy_path, final_embeddings)
#
#     # 保持输出文件名为 category_emb.pkl，无缝适配 POI2emb.py
#     pkl_path = os.path.join(output_dir, "category_emb.pkl")
#     with open(pkl_path, 'wb') as f:
#         pickle.dump(category_to_embedding, f)
#
#     print(f"\n 品类向量提取完成！结果已保存至: {output_dir}")
#     print(f"  -  矩阵文件: {npy_path}")
#     print(f"  - 字典文件: {pkl_path} (此文件将被 POI2emb.py 读取)")
#
#
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Generate POI category embeddings for SA-SID")
#
#     # 默认路径配置（你可以通过命令行参数灵活覆盖）
#     parser.add_argument("--csv_path", default="../data/NOLA/poi_info.csv", help="Path to input CSV file")
#     parser.add_argument("--output_dir", default="../data/NOLA/embeddings", help="Output directory for results")
#     parser.add_argument("--model", default="all-MiniLM-L6-v2", help="Sentence transformer model name")
#
#     # ⚠️ 目标维度恢复锁定为 64
#     parser.add_argument("--dim", type=int, default=64, help="Target dimensionality (PCA)")
#     parser.add_argument("--column", default="category", help="Category column name")
#
#     args = parser.parse_args()
#
#     category2vec(
#         csv_path=args.csv_path,
#         output_dir=args.output_dir,
#         model_name=args.model,
#         n_components=args.dim,
#         category_column=args.column
#     )

# 读取元数据文件: ../data/NOLA/poi_info.csv
# 共发现 989 种独特的 POI 类别
# 正在加载语义提取模型: all-MiniLM-L6-v2
# 正在生成品类文本 Embedding...
# Batches: 100%|██████████| 31/31 [00:00<00:00, 60.43it/s]
# 初始提取向量维度: (989, 384)
# 正在执行 PCA 降维，目标维度: 64 ...
# 降维后最终向量维度: (989, 64)
#
#  品类向量提取完成！结果已保存至: ../data/NOLA/
#   -  矩阵文件: ../data/NOLA/category_embeddings.npy
#   - 字典文件: ../data/NOLA/category_emb.pkl (此文件将被 POI2emb.py 读取)
import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import argparse
import pickle


def category2vec(csv_path, output_dir, model_name="all-MiniLM-L6-v2", n_components=64, category_column="category"):
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 1. 读取基础元数据 CSV
    print(f"读取元数据文件: {csv_path}")
    df = pd.read_csv(csv_path, encoding='utf-8')

    # 容错处理：填充可能为空的分类
    df[category_column] = df[category_column].fillna('Unknown')
    original_categories = df[category_column].unique().tolist()
    print(f"共发现 {len(original_categories)} 种原生的 POI 复合类别组合 (如 Yelp 的 A, B, C)")

    # --- SA-SID 核心优化 1: 基础子类别拆解 ---
    base_categories_set = set()
    for cat_str in original_categories:
        # 按逗号拆分，去除首尾空格
        sub_cats = [c.strip() for c in str(cat_str).split(',')]
        base_categories_set.update(sub_cats)

    base_categories = list(base_categories_set)
    print(f"拆解重组后，提取出 {len(base_categories)} 种绝对基础的独立类别 (Base Categories)")
    # ----------------------------------------

    # 2. 加载轻量级句子转化大模型
    print(f"正在加载语义提取模型: {model_name}")
    # 注意：请确保这是你本地可用的路径，或者直接填模型名称让它自动下载
    model_path = "/home/mysjz/mywork/Models/all-MiniLM-L6-v2"
    if not os.path.exists(model_path):
        model_path = model_name  # 回退到在线下载
    model = SentenceTransformer(model_path)

    # 3. 对【基础类别】提取高维稠密文本特征
    print("正在生成基础品类的文本 Embedding...")
    base_embeddings = model.encode(base_categories, show_progress_bar=True)
    print(f"基础向量初始维度: {base_embeddings.shape}")

    # 4. 应用 PCA 降维至标准的 64 维 (仅对基础类别进行)
    if n_components is not None and n_components < base_embeddings.shape[1]:
        print(f"正在执行 PCA 降维，目标维度: {n_components} ...")
        pca = PCA(n_components=n_components)
        base_embeddings_reduced = pca.fit_transform(base_embeddings)
        print(f"降维后基础向量维度: {base_embeddings_reduced.shape}")
    else:
        base_embeddings_reduced = base_embeddings

    # 建立【基础类别】到【降维向量】的快速查询字典
    base_cat_to_emb = dict(zip(base_categories, base_embeddings_reduced))

    # --- SA-SID 核心优化 2: 复合词组切分聚合 (Mean Pooling) ---
    print("正在为原生复合类别执行 Mean Pooling (平均池化) 聚合...")
    category_to_embedding = {}
    final_embeddings_list = []

    for raw_cat in original_categories:
        sub_cats = [c.strip() for c in str(raw_cat).split(',')]
        # 提取出组成该原生类别的所有基础向量
        valid_vecs = [base_cat_to_emb[sc] for sc in sub_cats if sc in base_cat_to_emb]

        if len(valid_vecs) > 0:
            # 多个基础向量按列求平均，代表该 POI 的综合复合属性
            mean_vec = np.mean(valid_vecs, axis=0)
        else:
            mean_vec = np.zeros(n_components, dtype=np.float32)

        category_to_embedding[raw_cat] = mean_vec
        final_embeddings_list.append(mean_vec)
    # -----------------------------------------------------------

    # 5. 落盘保存
    final_embeddings = np.array(final_embeddings_list, dtype=np.float32)
    npy_path = os.path.join(output_dir, "category_embeddings.npy")
    np.save(npy_path, final_embeddings)

    # 保持输出文件名为 category_emb.pkl，完全无缝适配你后续的 POI2emb.py
    pkl_path = os.path.join(output_dir, "category_emb.pkl")
    with open(pkl_path, 'wb') as f:
        pickle.dump(category_to_embedding, f)

    print(f"\n SA-SID 品类向量提取（复合聚合版）完成！结果已保存至: {output_dir}")
    print(f"  -  矩阵文件: {npy_path}")
    print(f"  - 字典文件: {pkl_path} (将被 POI2emb.py 精准读取)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Multi-Category Pooled embeddings for SA-SID")

    parser.add_argument("--csv_path", default="../data/NOLA/poi_info.csv", help="Path to input CSV file")
    parser.add_argument("--output_dir", default="../data/NOLA/embeddings", help="Output directory for results")
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="Sentence transformer model name")
    parser.add_argument("--dim", type=int, default=64, help="Target dimensionality (PCA)")
    parser.add_argument("--column", default="category", help="Category column name")

    args = parser.parse_args()

    category2vec(
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        model_name=args.model,
        n_components=args.dim,
        category_column=args.column
    )