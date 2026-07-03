# import os
# import os.path as osp
# import json
# import pickle
# import ast
# import pandas as pd
# import numpy as np
# from typing import Dict, Any, List, Tuple, Optional
# from sklearn.preprocessing import LabelEncoder
#
# # ==========================================
# # 全局路径与参数配置区
# # ==========================================
# # 请确保这里的路径和你的实际工作区一致
# DATA_DIR = "/home/mysjz/mywork/V2-SID/data/NOLA"
# RAW_PATH = osp.join(DATA_DIR, "NOLA.csv")  # 之前跑完 filter_data 带有情感分数的全量表
# SID_PATH = osp.join(DATA_DIR, "SID/NOLA_SID.csv")  # 我们刚刚生成的密码本
# OUT_DIR = osp.join(DATA_DIR, "LLM_data")  # LLM 训练数据集输出目录
# os.makedirs(OUT_DIR, exist_ok=True)
#
#
# class Config:
#     min_poi_freq = 1
#     min_user_freq = 1
#     train_ratio = 0.8
#     val_ratio = 0.1
#     remove_isolated_24h = False
#     session_time_interval_min = 12 * 60  # 12小时作为切分一个轨迹的阈值
#     ignore_singleton_trajectories = True
#
#
# cfg = Config()
#
# INSTRUCTION = (
#     "Here is a record of a user's POI accesses. Your task is to predict the SID of the next POI the user is likely to visit based on their historical spatio-temporal trajectory and aspect-based sentiment experiences."
# )
# LETTERS = "abcdefghijklmnopqrstuvwxyz"
#
#
# # ==========================================
# # 🧠 核心增强：时序动态交互的阶梯情感映射器
# # ==========================================
# def map_sentiment_to_text(service, env, price, loc, core):
#     """
#     针对单次历史交互，使用多级阈值转化为极度细腻的推荐系统特征词。
#     """
#     excellent = []
#     good = []
#     poor = []
#     terrible = []
#
#     aspects = {
#         "service": service,
#         "environment": env,
#         "price": price,
#         "location": loc,
#         "core experience": core
#     }
#
#     # 时序分数波动大，我们可以设定人工绝对阈值的多级阶梯
#     for aspect, score in aspects.items():
#         if pd.isna(score): continue
#         if score >= 0.5:
#             excellent.append(aspect)  # 极度满意
#         elif score >= 0.2:
#             good.append(aspect)  # 比较满意
#         elif score <= -0.5:
#             terrible.append(aspect)  # 极度不满
#         elif score <= -0.2:
#             poor.append(aspect)  # 比较不满
#
#     if not any([excellent, good, poor, terrible]):
#         return ""  # 纯粹的路过打卡，无明显情感
#
#     parts = []
#     if excellent: parts.append(f"highly impressed by the {', '.join(excellent)}")
#     if good: parts.append(f"satisfied with the {', '.join(good)}")
#     if poor: parts.append(f"disappointed with the {', '.join(poor)}")
#     if terrible: parts.append(f"extremely unhappy about the {', '.join(terrible)}")
#
#     # 例如: " (feeling highly impressed by the environment, but disappointed with the price)"
#     return " (feeling " + ", but ".join(parts) + ")"
#
#
# # ==========================================
# # 第一部分：数据清洗与时序切分
# # ==========================================
# def read_format(file_path: str) -> pd.DataFrame:
#     df = pd.read_csv(file_path)
#     # 统一列名大小写，容错处理
#     col_mapping = {c.lower(): c for c in df.columns}
#     # 确保时间格式正确
#     time_col = col_mapping.get("time", "Time")
#     df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
#     uid_col = col_mapping.get("uid", "UId")
#     df = df.sort_values([uid_col, time_col]).reset_index(drop=True)
#     # 将列名统一重命名为原代码期望的格式
#     rename_dict = {
#         uid_col: "UId", col_mapping.get("pid", "PId"): "PId",
#         col_mapping.get("category", "Category"): "Category",
#         time_col: "Time"
#     }
#     df = df.rename(columns=rename_dict)
#
#     # 将情感列名统一下划线小写，方便后续提取
#     for col in ['service', 'environment', 'price', 'location', 'core_experience']:
#         actual_col = col_mapping.get(col)
#         if actual_col:
#             df = df.rename(columns={actual_col: col})
#         else:
#             df[col] = 0.0  # 兜底，防止缺列报错
#     return df
#
#
# def build_pseudo_sessions(df: pd.DataFrame, session_time_interval_min: int) -> pd.DataFrame:
#     df = df.sort_values(by=['UId', 'Time'], ascending=True).reset_index(drop=True)
#     diffs = df.groupby('UId')['Time'].diff()
#     diffs_min = diffs.dt.total_seconds() / 60.0
#     new_session = diffs.isna() | (diffs_min > session_time_interval_min)
#     df['pseudo_session_trajectory_id'] = new_session.cumsum().astype(int) - 1
#     return df
#
#
# def remove_unseen_user_poi(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
#     train = df[df['SplitTag'] == 'train']
#     val = df[df['SplitTag'] == 'validation']
#     test = df[df['SplitTag'] == 'test']
#     train_users = set(train['UId'])
#     train_pois = set(train['PId'])
#     val = val[val['UId'].isin(train_users) & val['PId'].isin(train_pois)].reset_index(drop=True)
#     test = test[test['UId'].isin(train_users) & test['PId'].isin(train_pois)].reset_index(drop=True)
#     sample = pd.concat([train.reset_index(drop=True), val, test], axis=0).sort_values(['UId', 'Time']).reset_index(
#         drop=True)
#     return {'sample': sample, 'train_sample': train.reset_index(drop=True), 'validate_sample': val, 'test_sample': test}
#
#
# # ==========================================
# # 第二部分：构建携带着情感的用户字典
# # ==========================================
# def build_user_sequence_dict(sample_df: pd.DataFrame) -> dict:
#     user_seq = {}
#     sample_df = sample_df.sort_values(by=["UId", "Time"], ascending=True)
#
#     for uid, g in sample_df.groupby("UId"):
#         user_seq[int(uid)] = {
#             "PIds": g["PId"].astype(int).tolist(),
#             "Times": g["Time"].astype(str).tolist(),
#             # 核心扩容：打包当次交互的情感特征！
#             "Services": g["service"].astype(float).tolist(),
#             "Envs": g["environment"].astype(float).tolist(),
#             "Prices": g["price"].astype(float).tolist(),
#             "Locs": g["location"].astype(float).tolist(),
#             "Cores": g["core_experience"].astype(float).tolist(),
#         }
#     return user_seq
#
#
# def build_session_index_df(split_df: pd.DataFrame, min_start_seq: int = 5, sequence_length: int = 50) -> pd.DataFrame:
#     records = []
#     split_df = split_df.sort_values(["UId", "sequence_id"]).reset_index(drop=True)
#     for session_id, g in split_df.groupby("pseudo_session_trajectory_id"):
#         uids = g["UId"].unique()
#         if len(uids) != 1: continue
#         uid = int(uids[0])
#         start_seq = int(g["sequence_id"].min())
#         end_seq = int(g["sequence_id"].max())
#         if start_seq < min_start_seq: continue
#         start_seq = max(0, end_seq - sequence_length + 1)
#         records.append({"UId": uid, "session_id": int(session_id), "start_seq": start_seq, "end_seq": end_seq})
#     return pd.DataFrame(records)
#
#
# # ==========================================
# # 第三部分：LLM 提示词组装 (The Engine)
# # ==========================================
# def sid_list_to_tokens(sid_list: List[int]) -> str:
#     toks = []
#     for i, v in enumerate(sid_list):
#         if i >= len(LETTERS):
#             toks.append(f"<x{i}_{int(v)}>")
#         else:
#             toks.append(f"<{LETTERS[i]}_{int(v)}>")
#     return "".join(toks)
#
#
# def load_pid_to_tokens(semitic_csv_path: str) -> Dict[int, str]:
#     df = pd.read_csv(semitic_csv_path)
#     pid2tok = {}
#     for _, row in df.iterrows():
#         pid = int(row["pid"])
#         sid_raw = row["sid"]
#         try:
#             sid_list = ast.literal_eval(sid_raw) if isinstance(sid_raw, str) else list(sid_raw)
#             pid2tok[pid] = sid_list_to_tokens([int(x) for x in sid_list])
#         except Exception:
#             pid2tok[pid] = "<unk>"
#     return pid2tok
#
#
# def build_one_example(uid_new, times, pids, services, envs, prices, locs, cores, pid2tok, start_seq, end_seq):
#     if end_seq <= start_seq or end_seq >= len(pids): return None
#
#     # 提取历史轨迹序列
#     hist_pids = pids[start_seq:end_seq]
#     hist_times = times[start_seq:end_seq]
#
#     # 提取对应的历史情感序列
#     h_ser, h_env = services[start_seq:end_seq], envs[start_seq:end_seq]
#     h_pri, h_loc = prices[start_seq:end_seq], locs[start_seq:end_seq]
#     h_cor = cores[start_seq:end_seq]
#
#     target_pid = int(pids[end_seq])
#     target_time = times[end_seq]
#
#     hist_parts = []
#     for i in range(len(hist_pids)):
#         pid, tt = hist_pids[i], hist_times[i]
#         tok = pid2tok.get(int(pid), "<unk>")
#
#         # 调用映射器，将枯燥的数字变为大模型能看懂的情感句子
#         sentiment_text = map_sentiment_to_text(h_ser[i], h_env[i], h_pri[i], h_loc[i], h_cor[i])
#
#         # 完美融合：时间 + 空间ID + (情感状态)
#         hist_parts.append(f"{str(tt)[:16]} visited {tok}{sentiment_text}")
#
#     history_str = ", then ".join(hist_parts)
#     input_text = (
#         f"User_{uid_new} trajectory history: {history_str}.\n"
#         f"Question: At {str(target_time)[:16]}, which Semantic ID is the user most likely to visit next?"
#     )
#
#     return {
#         "instruction": INSTRUCTION,
#         "input": input_text,
#         "output": pid2tok.get(target_pid, "<unk>")
#     }
#
#
# def build_llm_json_from_sessions(session_csv_path, user_sequences, pid2tok, out_json_path):
#     sess_df = pd.read_csv(session_csv_path)
#
#     # 删除这两行局部映射的废代码！
#     # unique_uids = sess_df["UId"].dropna().astype(int).unique().tolist()
#     # uid_map = {int(uid): i for i, uid in enumerate(unique_uids)}
#
#     examples = []
#
#     for _, row in sess_df.iterrows():
#         uid, start_seq, end_seq = int(row["UId"]), int(row["start_seq"]), int(row["end_seq"])
#         user_seq = user_sequences.get(uid) or user_sequences.get(str(uid))
#         if not user_seq: continue
#
#         ex = build_one_example(
#             uid,  # 核心修复：直接传入真实的 uid！不再使用 uid_map[uid]
#             user_seq["Times"], user_seq["PIds"],
#             user_seq["Services"], user_seq["Envs"], user_seq["Prices"], user_seq["Locs"], user_seq["Cores"],
#             pid2tok, start_seq, end_seq
#         )
#         if ex: examples.append(ex)
#
#     with open(out_json_path, "w", encoding="utf-8") as f:
#         json.dump(examples, f, ensure_ascii=False, indent=2)
#     print(f"成功生成带情感推理的 SFT 数据集: {len(examples)} 条 -> {out_json_path}")
#
#
# # ==========================================
# # 主执行流 (Main Pipeline)
# # ==========================================
# if __name__ == "__main__":
#     print("[Step 1] 读取原始带情感特征的交互数据...")
#     df = read_format(RAW_PATH)
#
#     print("[Step 2] 全局时间切分与 Session 划分...")
#     df = df.sort_values("Time").reset_index(drop=True)
#     n = len(df)
#     train_end = int(n * cfg.train_ratio)
#     val_end = int(n * (cfg.train_ratio + cfg.val_ratio))
#     df['SplitTag'] = 'train'
#     df.loc[train_end:val_end - 1, 'SplitTag'] = 'validation'
#     df.loc[val_end:, 'SplitTag'] = 'test'
#
#     df = build_pseudo_sessions(df, cfg.session_time_interval_min)
#     df["sequence_id"] = df.sort_values(['UId', 'Time']).groupby("UId").cumcount().astype("int64")
#
#     result = remove_unseen_user_poi(df)
#     final_df = result['sample']
#
#     # 落盘中间表 (护送情感列一起落盘)
#     keep_cols = ['UId', 'PId', 'Time', 'SplitTag', 'pseudo_session_trajectory_id', 'sequence_id',
#                  'service', 'environment', 'price', 'location', 'core_experience']
#     final_df = final_df[[c for c in keep_cols if c in final_df.columns]]
#     final_df.to_csv(osp.join(OUT_DIR, 'sample.csv'), index=False)
#     final_df[final_df['SplitTag'] == 'train'].to_csv(osp.join(OUT_DIR, 'train_sample.csv'), index=False)
#     final_df[final_df['SplitTag'] == 'validation'].to_csv(osp.join(OUT_DIR, 'validate_sample.csv'), index=False)
#     final_df[final_df['SplitTag'] == 'test'].to_csv(osp.join(OUT_DIR, 'test_sample.csv'), index=False)
#
#     print("[Step 3] 构建高维用户序列字典 (打包情感)...")
#     user_sequence_dict = build_user_sequence_dict(final_df)
#     with open(osp.join(OUT_DIR, "user_sequences.pkl"), "wb") as f:
#         pickle.dump(user_sequence_dict, f)
#
#     train_idx = build_session_index_df(final_df[final_df['SplitTag'] == 'train'])
#     val_idx = build_session_index_df(final_df[final_df['SplitTag'] == 'validation'], min_start_seq=1)
#     test_idx = build_session_index_df(final_df[final_df['SplitTag'] == 'test'], min_start_seq=1)
#
#     train_idx.to_csv(osp.join(OUT_DIR, "train_session.csv"), index=False)
#     val_idx.to_csv(osp.join(OUT_DIR, "val_session.csv"), index=False)
#     test_idx.to_csv(osp.join(OUT_DIR, "test_session.csv"), index=False)
#
#     print("[Step 4] 呼叫 SA-SID 密码本，拼装 LLM 时序问答对...")
#     pid2tok = load_pid_to_tokens(SID_PATH)
#
#     build_llm_json_from_sessions(osp.join(OUT_DIR, "train_session.csv"), user_sequence_dict, pid2tok,
#                                  osp.join(OUT_DIR, "train_llm.json"))
#     build_llm_json_from_sessions(osp.join(OUT_DIR, "val_session.csv"), user_sequence_dict, pid2tok,
#                                  osp.join(OUT_DIR, "val_llm.json"))
#     build_llm_json_from_sessions(osp.join(OUT_DIR, "test_session.csv"), user_sequence_dict, pid2tok,
#                                  osp.join(OUT_DIR, "test_llm.json"))
#
#     print("完美！大模型特征加工车间全线竣工！去查看你的 train_llm.json 吧！")
#
# # [Step 1] 读取原始带情感特征的交互数据...
# # [Step 2] 全局时间切分与 Session 划分...
# # [Step 3] 构建高维用户序列字典 (打包情感)...
# # [Step 4] 呼叫 SA-SID 密码本，拼装 LLM 时序问答对...
# # 成功生成带情感推理的 SFT 数据集: 11895 条 -> /home/mysjz/mywork/V2-SID/data/NOLA/LLM_data/train_llm.json
# # 成功生成带情感推理的 SFT 数据集: 1893 条 -> /home/mysjz/mywork/V2-SID/data/NOLA/LLM_data/val_llm.json
# # 成功生成带情感推理的 SFT 数据集: 1486 条 -> /home/mysjz/mywork/V2-SID/data/NOLA/LLM_data/test_llm.json
# # 完美！大模型特征加工车间全线竣工！去查看你的 train_llm.json 吧！
import os
import os.path as osp
import json
import pickle
import ast
import argparse
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from tqdm import tqdm


# ==========================================
# 全局参数配置区
# ==========================================
class Config:
    min_poi_freq = 1
    min_user_freq = 1
    train_ratio = 0.8
    val_ratio = 0.1
    remove_isolated_24h = False
    session_time_interval_min = 12 * 60  # 12小时作为切分一个轨迹的阈值
    ignore_singleton_trajectories = True
    min_start_seq = 5  # 至少需要 5 个历史记录才进行预测
    sequence_length = 50  # 最大截断历史长度 (避免 OOM)


cfg = Config()

INSTRUCTION = (
    "Here is a record of a user's POI access history. "
    "Based on the historical spatio-temporal trajectory and their aspect-based sentiment experiences, "
    "your task is to predict the Semantic ID of the next POI the user is most likely to visit."
)

LETTERS = "abcdefghijklmnopqrstuvwxyz"


# ==========================================
# 核心增强：时序动态交互的阶梯情感映射器
# ==========================================
def map_sentiment_to_text(service, env, price, loc, core):
    """
    针对单次历史交互，使用多级阈值转化为细腻的情感推荐提示词。
    """
    excellent, good, poor, terrible = [], [], [], []

    aspects = {
        "service": service,
        "environment": env,
        "price": price,
        "location": loc,
        "core experience": core
    }

    # 阶梯式情感映射 (兼容 NaN 和 0.0)
    for aspect, score in aspects.items():
        if pd.isna(score) or score == 0.0:
            continue
        if score >= 0.5:
            excellent.append(aspect)
        elif score >= 0.2:
            good.append(aspect)
        elif score <= -0.5:
            terrible.append(aspect)
        elif score <= -0.2:
            poor.append(aspect)

    if not any([excellent, good, poor, terrible]):
        return ""  # 纯粹的路过打卡

    parts = []
    if excellent: parts.append(f"highly impressed by the {', '.join(excellent)}")
    if good: parts.append(f"satisfied with the {', '.join(good)}")
    if poor: parts.append(f"disappointed with the {', '.join(poor)}")
    if terrible: parts.append(f"extremely unhappy about the {', '.join(terrible)}")

    return " (feeling " + ", but ".join(parts) + ")"


# ==========================================
# 第一部分：数据清洗与时序切分
# ==========================================
def read_format(file_path: str) -> pd.DataFrame:
    print(f"Loading raw data from {file_path}...")
    df = pd.read_csv(file_path, encoding='utf-8')

    col_mapping = {c.lower(): c for c in df.columns}

    time_col = col_mapping.get("time", "Time")
    df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")

    uid_col = col_mapping.get("uid", "UId")
    pid_col = col_mapping.get("pid", "PId")

    df = df.dropna(subset=[uid_col, pid_col, time_col])

    df[uid_col] = df[uid_col].astype(int)
    df[pid_col] = df[pid_col].astype(int)

    df = df.sort_values([uid_col, time_col]).reset_index(drop=True)

    rename_dict = {
        uid_col: "UId",
        pid_col: "PId",
        col_mapping.get("category", "Category"): "Category",
        time_col: "Time"
    }
    df = df.rename(columns=rename_dict)

    for col in ['service', 'environment', 'price', 'location', 'core_experience']:
        actual_col = col_mapping.get(col)
        if actual_col:
            df = df.rename(columns={actual_col: col})
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
        else:
            df[col] = 0.0

    return df


def build_pseudo_sessions(df: pd.DataFrame, session_time_interval_min: int) -> pd.DataFrame:
    df = df.sort_values(by=['UId', 'Time'], ascending=True).reset_index(drop=True)
    diffs = df.groupby('UId')['Time'].diff()
    diffs_min = diffs.dt.total_seconds() / 60.0
    new_session = diffs.isna() | (diffs_min > session_time_interval_min)
    df['pseudo_session_trajectory_id'] = new_session.cumsum().astype(int) - 1
    return df


def remove_unseen_user_poi(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    train = df[df['SplitTag'] == 'train']
    val = df[df['SplitTag'] == 'validation']
    test = df[df['SplitTag'] == 'test']

    train_users = set(train['UId'])
    train_pois = set(train['PId'])

    # 防止冷启动崩溃：剔除在 Train 中没出现过的孤立特征
    val = val[val['UId'].isin(train_users) & val['PId'].isin(train_pois)].reset_index(drop=True)
    test = test[test['UId'].isin(train_users) & test['PId'].isin(train_pois)].reset_index(drop=True)

    sample = pd.concat([train, val, test], axis=0).sort_values(['UId', 'Time']).reset_index(drop=True)
    return {'sample': sample, 'train': train, 'val': val, 'test': test}


# ==========================================
# 第二部分：构建携带着情感的用户字典
# ==========================================
def build_user_sequence_dict(sample_df: pd.DataFrame) -> dict:
    user_seq = {}
    sample_df = sample_df.sort_values(by=["UId", "Time"], ascending=True)

    for uid, g in tqdm(sample_df.groupby("UId"), desc="Building User Histories"):
        user_seq[int(uid)] = {
            "PIds": g["PId"].tolist(),
            "Times": g["Time"].astype(str).tolist(),
            "Services": g["service"].tolist(),
            "Envs": g["environment"].tolist(),
            "Prices": g["price"].tolist(),
            "Locs": g["location"].tolist(),
            "Cores": g["core_experience"].tolist(),
        }
    return user_seq


def build_session_index_df(split_df: pd.DataFrame, min_start_seq: int, sequence_length: int) -> pd.DataFrame:
    # 增加空表保护，防止后续迭代器崩溃
    if split_df.empty:
        return pd.DataFrame(columns=["UId", "session_id", "start_seq", "end_seq"])

    records = []
    split_df = split_df.sort_values(["UId", "sequence_id"]).reset_index(drop=True)

    for session_id, g in split_df.groupby("pseudo_session_trajectory_id"):
        uids = g["UId"].unique()
        if len(uids) != 1: continue
        uid = int(uids[0])

        for end_seq in g["sequence_id"].tolist():
            end_seq = int(end_seq)

            # 全局历史不足不预测，保证LLM输入有足够长度
            if end_seq < min_start_seq:
                continue

            start_seq = max(0, end_seq - sequence_length + 1)

            records.append({
                "UId": uid,
                "session_id": int(session_id),
                "start_seq": start_seq,
                "end_seq": end_seq
            })

    return pd.DataFrame(records, columns=["UId", "session_id", "start_seq", "end_seq"])


# ==========================================
# 第三部分：LLM 提示词组装 (The Engine)
# ==========================================
def sid_list_to_tokens(sid_list: List[int]) -> str:
    if len(sid_list) != 4:
        raise ValueError(f"检测到异常维度的 SID: {sid_list}，必须为 4 维！")
    return f"<a_{sid_list[0]}><b_{sid_list[1]}><c_{sid_list[2]}><d_{sid_list[3]}>"


def load_pid_to_tokens(sid_csv_path: str) -> Dict[int, str]:
    print(f"Loading SID mapping from {sid_csv_path}...")
    df = pd.read_csv(sid_csv_path)
    pid2tok = {}
    for _, row in df.iterrows():
        pid = int(row["pid"])
        sid_raw = row["sid"]
        try:
            sid_list = ast.literal_eval(sid_raw) if isinstance(sid_raw, str) else list(sid_raw)
            pid2tok[pid] = sid_list_to_tokens([int(x) for x in sid_list])
        except Exception:
            pid2tok[pid] = "<unk>"
    return pid2tok


def build_one_example(uid_new, times, pids, services, envs, prices, locs, cores, pid2tok, start_seq, end_seq):
    if end_seq <= start_seq or end_seq >= len(pids):
        return None

    # 提取历史序列
    hist_pids = pids[start_seq:end_seq]
    hist_times = times[start_seq:end_seq]
    h_ser, h_env = services[start_seq:end_seq], envs[start_seq:end_seq]
    h_pri, h_loc = prices[start_seq:end_seq], locs[start_seq:end_seq]
    h_cor = cores[start_seq:end_seq]

    # Target (待预测的 Next POI)
    target_pid = int(pids[end_seq])
    target_time = times[end_seq]
    target_tok = pid2tok.get(target_pid, "<unk>")

    # 丢弃无效目标
    if target_tok == "<unk>": return None

    hist_parts = []
    for i in range(len(hist_pids)):
        pid, tt = hist_pids[i], hist_times[i]
        tok = pid2tok.get(int(pid), "<unk>")
        sentiment_text = map_sentiment_to_text(h_ser[i], h_env[i], h_pri[i], h_loc[i], h_cor[i])
        hist_parts.append(f"[{str(tt)[:16]}] visited {tok}{sentiment_text}")

    history_str = ", then ".join(hist_parts)

    input_text = (
        f"User_{uid_new} trajectory history: {history_str}.\n"
        f"Question: At [{str(target_time)[:16]}], which Semantic ID is the user most likely to visit next?"
    )

    return {
        "instruction": INSTRUCTION,
        "input": input_text,
        "output": target_tok
    }


def build_llm_json_from_sessions(session_df, user_sequences, pid2tok, out_json_path, desc="Processing"):
    examples = []
    if session_df.empty:
        print(f"⚠️ 警告: {desc} 分配到的有效切分数据为空！")
    else:
        for _, row in tqdm(session_df.iterrows(), total=len(session_df), desc=desc):
            uid, start_seq, end_seq = int(row["UId"]), int(row["start_seq"]), int(row["end_seq"])
            user_seq = user_sequences.get(uid)

            if not user_seq: continue

            ex = build_one_example(
                uid, user_seq["Times"], user_seq["PIds"],
                user_seq["Services"], user_seq["Envs"], user_seq["Prices"], user_seq["Locs"], user_seq["Cores"],
                pid2tok, start_seq, end_seq
            )
            if ex: examples.append(ex)

    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(examples, f, ensure_ascii=False, indent=2)
    print(f"生成带情感推理的 SFT 数据集: {len(examples)} 条 -> {out_json_path}")


# ==========================================
# 主执行流 (Main Pipeline)
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="LBSN Preprocess & SFT Prompt Generation")
    parser.add_argument("--raw_path", type=str, default="/home/mysjz/mywork/V2-SID/data/NOLA/NOLA.csv", help="Path to raw POI interactions")
    parser.add_argument("--sid_path", type=str, default="/home/mysjz/mywork/V2-SID/data/NOLA/SID/NOLA_SID.csv",
                        help="Path to generated SA-SID mapping")
    parser.add_argument("--out_dir", type=str, default="/home/mysjz/mywork/V2-SID/data/NOLA/LLM_data", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("\n[Step 1] 读取原始带情感特征的交互数据...")
    df = read_format(args.raw_path)

    print("\n[Step 2] 【单一用户】生命周期进行 80/10/10 划分...")
    df = df.sort_values(['UId', 'Time']).reset_index(drop=True)

    seq_idx = df.groupby('UId').cumcount()
    user_counts = df.groupby('UId')['UId'].transform('count')

    train_end = np.floor(user_counts * cfg.train_ratio).astype(int)
    val_end = np.floor(user_counts * (cfg.train_ratio + cfg.val_ratio)).astype(int)

    valid_mask = user_counts >= 3
    val_end = np.where(valid_mask & (val_end == user_counts), user_counts - 1, val_end)
    train_end = np.where(valid_mask & (train_end >= val_end), val_end - 1, train_end)

    df['SplitTag'] = 'train'
    df.loc[(seq_idx >= train_end) & (seq_idx < val_end), 'SplitTag'] = 'validation'
    df.loc[(seq_idx >= val_end), 'SplitTag'] = 'test'

    print("\n[Step 2.5] 清洗冷启动与断层重排...")
    # 第一步：先过滤冷启动没见过的 POI
    result_dfs = remove_unseen_user_poi(df)
    final_df = result_dfs['sample']

    # 核心救火点：清理无效数据后，必须重新计算连续的 sequence_id 和 session
    # 否则原有的序列会因为丢弃行导致错位，进而使得下方预测代码越界判定为 Invalid！
    final_df = final_df.sort_values(['UId', 'Time']).reset_index(drop=True)
    final_df = build_pseudo_sessions(final_df, cfg.session_time_interval_min)
    final_df["sequence_id"] = final_df.groupby("UId").cumcount().astype(int)

    # 拿着洗干净且重新排过序的完整表，重新抓取 Train / Val / Test
    train_df = final_df[final_df['SplitTag'] == 'train'].reset_index(drop=True)
    val_df = final_df[final_df['SplitTag'] == 'validation'].reset_index(drop=True)
    test_df = final_df[final_df['SplitTag'] == 'test'].reset_index(drop=True)

    print(f"数据清洗后留存: Train({len(train_df)}), Val({len(val_df)}), Test({len(test_df)})")

    # 落盘中间表
    keep_cols = ['UId', 'PId', 'Time', 'SplitTag', 'pseudo_session_trajectory_id', 'sequence_id',
                 'service', 'environment', 'price', 'location', 'core_experience']
    final_df = final_df[[c for c in keep_cols if c in final_df.columns]]
    final_df.to_csv(osp.join(args.out_dir, 'sample.csv'), index=False)
    train_df.to_csv(osp.join(args.out_dir, 'train_sample.csv'), index=False)
    val_df.to_csv(osp.join(args.out_dir, 'validate_sample.csv'), index=False)
    test_df.to_csv(osp.join(args.out_dir, 'test_sample.csv'), index=False)

    print("\n[Step 3] 构建高维用户序列字典 (打包情感)...")
    user_sequence_dict = build_user_sequence_dict(final_df)
    with open(osp.join(args.out_dir, "user_sequences.pkl"), "wb") as f:
        pickle.dump(user_sequence_dict, f)

    # 用更新后的 df 创建索引
    train_idx = build_session_index_df(train_df, min_start_seq=cfg.min_start_seq, sequence_length=cfg.sequence_length)
    val_idx = build_session_index_df(val_df, min_start_seq=1, sequence_length=cfg.sequence_length)
    test_idx = build_session_index_df(test_df, min_start_seq=1, sequence_length=cfg.sequence_length)

    train_idx.to_csv(osp.join(args.out_dir, "train_session.csv"), index=False)
    val_idx.to_csv(osp.join(args.out_dir, "val_session.csv"), index=False)
    test_idx.to_csv(osp.join(args.out_dir, "test_session.csv"), index=False)

    print("\n[Step 4] 呼叫 SA-SID 密码本，拼装 LLM 时序问答对...")
    pid2tok = load_pid_to_tokens(args.sid_path)

    build_llm_json_from_sessions(train_idx, user_sequence_dict, pid2tok, osp.join(args.out_dir, "train_llm.json"),
                                 "Train Data")
    build_llm_json_from_sessions(val_idx, user_sequence_dict, pid2tok, osp.join(args.out_dir, "val_llm.json"),
                                 "Val Data")
    build_llm_json_from_sessions(test_idx, user_sequence_dict, pid2tok, osp.join(args.out_dir, "test_llm.json"),
                                 "Test Data")

    print("\n完美！大模型特征加工车间全线竣工！")


if __name__ == "__main__":
    main()

# Step 1] 读取原始带情感特征的交互数据...
# Loading raw data from /home/mysjz/mywork/V2-SID/data/NOLA/NOLA.csv...
#
# [Step 2] 🔴按【单一用户】生命周期进行 80/10/10 划分...
#
# [Step 2.5] 🟢清洗冷启动与断层重排...
# 数据清洗后留存: Train(26401), Val(3218), Test(4233)
#
# [Step 3] 构建高维用户序列字典 (打包情感)...
# Building User Histories: 100%|██████████| 1734/1734 [00:00<00:00, 6672.14it/s]
#
# [Step 4] 呼叫 SA-SID 密码本，拼装 LLM 时序问答对...
# Loading SID mapping from /home/mysjz/mywork/V2-SID/data/NOLA/SID/NOLA_SID.csv...
# Train Data: 100%|██████████| 17731/17731 [00:00<00:00, 19786.33it/s]
# Val Data:   0%|          | 0/3218 [00:00<?, ?it/s]✅ 生成带情感推理的 SFT 数据集: 17731 条 -> /home/mysjz/mywork/V2-SID/data/NOLA/LLM_data/train_llm.json
# Val Data: 100%|██████████| 3218/3218 [00:00<00:00, 15381.29it/s]
# ✅ 生成带情感推理的 SFT 数据集: 3218 条 -> /home/mysjz/mywork/V2-SID/data/NOLA/LLM_data/val_llm.json
# Test Data: 100%|██████████| 4233/4233 [00:00<00:00, 15903.68it/s]
# ✅ 生成带情感推理的 SFT 数据集: 4233 条 -> /home/mysjz/mywork/V2-SID/data/NOLA/LLM_data/test_llm.json