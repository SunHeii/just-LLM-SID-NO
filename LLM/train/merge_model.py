import os
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def parse_args():
    parser = argparse.ArgumentParser(description="Merge Alignment LoRA with Base Llama-3 Model")

    # 1. 原始未受污染的 Llama-3 基座
    parser.add_argument("--base_model_path", type=str, default="/root/autodl-tmp/Llama-3-8B-Instruct",
                        help="Path to the original base model")
    # 2. 刚才 align_sid.py 训练完保存的带新词表的 LoRA 目录
    parser.add_argument("--lora_path", type=str, default="/root/autodl-tmp/V2-SID/data/NOLA/model/align_lora",
                        help="Path to the trained LoRA and extended tokenizer")
    # 3. 合并后输出的完整大模型目录 (将被 sft_after_alignment.py 使用)
    parser.add_argument("--output_path", type=str, default="/root/autodl-tmp/V2-SID/data/NOLA/model/llama3-merged",
                        help="Path to save the fully merged model")

    return parser.parse_args()


def main():
    args = parse_args()
    print("=================================================")
    print(" 🔄 开始执行 SA-SID 基座模型与对齐权重融合")
    print("=================================================")

    # =========================================================
    # 🌟 核心步骤 1：必须从 lora_path 加载扩充后的 Tokenizer
    # 因为原版基座里没有 <a_53>, <d_1> 这些词！
    # =========================================================
    print(f"Loading Extended Tokenizer from {args.lora_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.lora_path, trust_remote_code=True)

    # =========================================================
    # 🌟 核心步骤 2：加载原始基座模型 (内存防爆机制)
    # 融合不需要 GPU，在 CPU 上用 bfloat16 加载可极大降低内存(RAM)压力
    # =========================================================
    print(f"Loading Base Model from {args.base_model_path}...")
    is_bf16_supported = torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False
    compute_dtype = torch.bfloat16 if is_bf16_supported else torch.float16

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        torch_dtype=compute_dtype,
        device_map="cpu",  # 强制放在 CPU 上，给 GPU 显存腾地方
        low_cpu_mem_usage=True,  # 🌟 核心优化：分块加载，防止 AutoDL 主机内存爆满被 Kill
        trust_remote_code=True
    )

    # =========================================================
    # 🌟 核心步骤 3：调整基座 Embedding 大小以匹配新 Tokenizer
    # 如果漏了这一步，下一步挂载 LoRA 时会直接报 Tensor shape mismatch 错误！
    # =========================================================
    print(f"Resizing base model token embeddings to {len(tokenizer)}...")
    base_model.resize_token_embeddings(len(tokenizer))

    # =========================================================
    # 🌟 核心步骤 4：挂载 LoRA 权重并执行物理合并
    # =========================================================
    print(f"Loading PEFT/LoRA adapter from {args.lora_path}...")
    model = PeftModel.from_pretrained(base_model, args.lora_path)

    print("Merging LoRA weights into base model (this may take a few minutes)...")
    merged_model = model.merge_and_unload()

    # =========================================================
    # 🌟 核心步骤 5：保存全新的大模型
    # =========================================================
    os.makedirs(args.output_path, exist_ok=True)
    print(f"Saving merged model to {args.output_path}...")

    # safe_serialization=True 保证输出为主流的 .safetensors 格式，加载速度更快
    merged_model.save_pretrained(args.output_path, safe_serialization=True)
    tokenizer.save_pretrained(args.output_path)

    print("\n🎉 融合完美收官！一个内置了 SA-SID 地理与情感常识的全新 Llama-3 已诞生！")
    print("👉 下一步：你可以直接运行 sft_after_alignment.py 开始时序推荐训练了！")


if __name__ == "__main__":
    main()