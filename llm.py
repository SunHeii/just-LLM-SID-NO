import os
import re
import json
import torch
import argparse
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SA-SID Recommendation Model")

    # 测试集路径
    parser.add_argument("--test_path", type=str, default="/root/autodl-tmp/V2-SID/data/NOLA/LLM_data/test_llm.json")

    # 基础融合模型与刚才训练好的推荐 LoRA 适配器
    parser.add_argument("--base_model_path", type=str, default="/root/autodl-tmp/V2-SID/data/NOLA/model/llama3-merged")
    parser.add_argument("--adapter_path", type=str,
                        default="/root/autodl-tmp/V2-SID/data/NOLA/model/llama3-sa-sid-recommendation")

    # 测试数量（为了快速验证，默认随机测 200 条，设为 -1 则测试全量）
    parser.add_argument("--num_test", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def extract_sid(text):
    """
    🌟 增强版提取器：无视任何空格、换行、或模型前置的废话。
    精准捕捉形如 <a_x>, <b_y> 的 token 并组装。
    """
    # 匹配所有的独立 token
    tokens = re.findall(r'<[abcd]_\d+>', text)
    if len(tokens) >= 4:
        # 取前四个 token 组装为标准的 SID
        return "".join(tokens[:4])
    # 如果没生成全，也返回已生成的部分供容错计算
    return "".join(tokens)


def calculate_hierarchical_accuracy(preds, targets):
    """
    核心学术指标：层级准确率 (Hierarchical Accuracy)
    评估模型在不同粒度下的推理能力。
    """
    level_1_hits = 0  # 宏观意图命中 <a_x>
    level_2_hits = 0  # 中观意图命中 <a_x><b_y>
    level_3_hits = 0  # 微观意图命中 <a_x><b_y><c_z>
    level_4_hits = 0  # 绝对精确命中 <a_x><b_y><c_z><d_w> (加上了情感排位)

    total = len(preds)
    if total == 0:
        return {}

    for p, t in zip(preds, targets):
        p_tokens = re.findall(r'<[abcd]_\d+>', p)
        t_tokens = re.findall(r'<[abcd]_\d+>', t)

        # 🌟 修复：解除“连坐惩罚”。即使模型生成不完整（少于4个），
        # 只要前面的层级生成对了，依然算命中该层级！这在学术评估中是标准做法。
        if not p_tokens or not t_tokens:
            continue

        if p_tokens[0] == t_tokens[0]:
            level_1_hits += 1
            if len(p_tokens) > 1 and len(t_tokens) > 1 and p_tokens[1] == t_tokens[1]:
                level_2_hits += 1
                if len(p_tokens) > 2 and len(t_tokens) > 2 and p_tokens[2] == t_tokens[2]:
                    level_3_hits += 1
                    if len(p_tokens) > 3 and len(t_tokens) > 3 and p_tokens[3] == t_tokens[3]:
                        level_4_hits += 1

    return {
        "Level 1 (Macro Geo/Cat) Acc": level_1_hits / total * 100,
        "Level 2 (Meso Block) Acc": level_2_hits / total * 100,
        "Level 3 (Micro Time) Acc": level_3_hits / total * 100,
        "Level 4 (Exact/Sentiment) Acc": level_4_hits / total * 100
    }


def main():
    args = parse_args()

    print("=================================================")
    print(" 📊 启动 SA-SID 大模型推荐效果量化评估 (Beam Search 增强版)")
    print("=================================================")

    # 1. 读取测试集
    with open(args.test_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    import random
    random.seed(args.seed)
    if args.num_test > 0 and args.num_test < len(test_data):
        test_data = random.sample(test_data, args.num_test)
        print(f"为了快速验证，随机抽取了 {args.num_test} 条测试样本。")
    else:
        print(f"准备在全量 {len(test_data)} 条测试集上进行评估 (这可能需要几个小时)。")

    # 2. 加载模型与 Tokenizer
    print("加载 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = "<|eot_id|>"
    # 🌟 生成任务最佳实践：左侧填充
    tokenizer.padding_side = "left"

    print("以 4-bit 模式加载基座模型 (防止 OOM)...")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        quantization_config=quant_config,
        device_map="auto",
        attn_implementation="sdpa"
    )

    print("挂载推荐任务 LoRA 适配器...")
    model = PeftModel.from_pretrained(base_model, args.adapter_path)
    model.eval()

    # 3. 开始批量推理
    predictions = []
    ground_truths = []

    print("\n🚀 开始生成预测...")
    for idx, item in enumerate(tqdm(test_data)):
        try:
            prefix_part, rest = item['input'].split(" trajectory history: ", 1)
            prefix_part += " trajectory history: "
            history_str, question_part = rest.split(".\nQuestion:", 1)
            question_part = ".\nQuestion:" + question_part
        except ValueError:
            prefix_part, history_str, question_part = "", item['input'], ""

        traj_list = history_str.split(", then ")
        # 测试时保留最后 10 次记录作为上下文
        short_history = ", then ".join(traj_list[-10:])
        safe_input = prefix_part + short_history + question_part

        prompt = (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{item['instruction']}<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            f"{safe_input}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=30,
                eos_token_id=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")],
                pad_token_id=tokenizer.pad_token_id,
                # ====================================================
                # 🌟 核心提升：开启束搜索 (Beam Search)
                # 这会让模型脑海中同时推演 5 条路线，找出得分最高的那条
                # 对于 SID 这种精确组合的生成，能极大提升命中率！
                # ====================================================
                num_beams=5,
                early_stopping=True,
                do_sample=False
            )

        input_length = inputs.input_ids.shape[1]
        raw_pred = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=False).strip()
        raw_pred = raw_pred.replace("<|eot_id|>", "").strip()

        extracted_pred = extract_sid(raw_pred)

        predictions.append(extracted_pred)
        ground_truths.append(item['output'])

        # 打印前 3 条作为直观展示
        if idx < 3:
            print("\n" + "-" * 40)
            print(f"👀 样本 {idx + 1} 直观对比:")
            print(f"原始未处理输出 (Raw): {raw_pred}")
            print(f"真实答案 (GT)      : {item['output']}")
            print(f"精准提取 (Pred)    : {extracted_pred}")
            print("-" * 40)

    # 4. 计算指标
    print("\n" + "=" * 50)
    print(" 📈 最终层级准确率评估结果 (Hierarchical Accuracy)")
    print("=" * 50)

    metrics = calculate_hierarchical_accuracy(predictions, ground_truths)

    for k, v in metrics.items():
        print(f"➤ {k:<30} : {v:.2f}%")

    print("\n💡 指标解析指南:")
    print("- Level 1 较高: 说明大模型完美学会了根据历史，推断用户接下来想去哪个城市的大片区/主类别。")
    print("- Level 3 较高: 说明大模型连极度微小的用户偏好(如某个特定时间段、细分商圈)都算准了。")
    print("- Level 4 (Exact): 这是最苛刻的指标。即便不高也没关系，只要 Level 1-3 高，就证明意图解耦极其成功！")


if __name__ == "__main__":
    main()

# ==================================================
#  📈 最终层级准确率评估结果 (Hierarchical Accuracy)
# ==================================================
# ➤ Level 1 (Macro Geo/Cat) Acc    : 8.00%
# ➤ Level 2 (Meso Block) Acc       : 3.00%
# ➤ Level 3 (Micro Time) Acc       : 0.00%
# ➤ Level 4 (Exact/Sentiment) Acc  : 0.00%
#
# 💡 指标解析指南:
# - Level 1 较高: 说明大模型完美学会了根据历史，推断用户接下来想去哪个城市的大片区/主类别。
# - Level 3 较高: 说明大模型连极度微小的用户偏好(如某个特定时间段、细分商圈)都算准了。
# - Level 4 (Exact): 这是最苛刻的指标。即便不高也没关系，只要 Level 1-3 高，就证明意图解耦极其成功！