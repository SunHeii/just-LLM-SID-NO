import os
import re
import argparse
import torch
from datasets import load_dataset

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)

# 开启显存碎片整理
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"


def parse_args():
    parser = argparse.ArgumentParser(description="SA-SID LLM Alignment Finetuning (Industrial Ready)")
    parser.add_argument("--data_path", type=str, default="/root/autodl-tmp/V2-SID/data/NOLA/alignment/train_align.json")
    parser.add_argument("--val_path", type=str, default="/root/autodl-tmp/V2-SID/data/NOLA/alignment/val_align.json")
    parser.add_argument("--model_name_or_path", type=str, default="/root/autodl-tmp/Llama-3-8B-Instruct")
    parser.add_argument("--output_dir", type=str, default="/root/autodl-tmp/V2-SID/data/NOLA/model/align_lora")

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)  # 对齐阶段可以保持较高的学习率
    parser.add_argument("--cutoff_len", type=int, default=1024)
    parser.add_argument("--wandb_proj", type=str, default="SA-SID-Alignment")
    return parser.parse_args()


def extract_sa_sid_tokens(dataset):
    print("正在自动扫描并提取新的 SA-SID 符号...")
    new_tokens = set()
    for item in dataset:
        found = re.findall(r'<[abcd]_\d+>', item['output'])
        new_tokens.update(found)
    new_tokens = sorted(list(new_tokens))
    print(f"成功提取 {len(new_tokens)} 个全新语义符号。")
    return new_tokens


def preprocess_dataset(dataset, tokenizer, cutoff_len):
    """
    对齐阶段的数据预处理：对提问部分打上 -100 掩码，只让模型学习生成 SID
    """

    def tokenize_and_mask(example):
        prompt = (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{example['instruction']}<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            f"{example['input']}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        response = f"{example['output']}<|eot_id|>"

        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        response_ids = tokenizer.encode(response, add_special_tokens=False)

        input_ids = prompt_ids + response_ids
        labels = [-100] * len(prompt_ids) + response_ids

        # =============================================================
        # 核心截断逻辑修复：保住注意力汇聚点 (Attention Sink)
        # =============================================================
        if len(input_ids) > cutoff_len:
            bos_id = input_ids[0]
            input_ids = [bos_id] + input_ids[-(cutoff_len - 1):]
            labels = [-100] + labels[-(cutoff_len - 1):]

        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels
        }

    return dataset.map(
        tokenize_and_mask,
        remove_columns=dataset.column_names,
        desc="Tokenizing and Masking Alignment Labels"
    )


def main():
    args = parse_args()

    # 关闭 wandb
    os.environ["WANDB_DISABLED"] = "true"

    print("=================================================")
    print(" 🚀 启动 SA-SID 语义对齐微调 (工业完善版)")
    print("=================================================")

    print(f"Loading data from {args.data_path}...")
    dataset = load_dataset("json", data_files={"train": args.data_path, "val": args.val_path})
    train_data = dataset["train"]
    val_data = dataset["val"]

    print(f"Loading Tokenizer from {args.model_name_or_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = "<|eot_id|>"
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    tokenizer.padding_side = "right"

    # 扩充词表：将 SA-SID 新符号加入
    new_tokens = extract_sa_sid_tokens(train_data)
    tokenizer.add_tokens(new_tokens)

    print("开始预处理数据集掩码...")
    train_dataset = preprocess_dataset(train_data, tokenizer, args.cutoff_len)
    val_dataset = preprocess_dataset(val_data, tokenizer, args.cutoff_len)

    # 🌟 自适应硬件浮点精度，防溢出报错
    is_bf16_supported = torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if is_bf16_supported else torch.float16

    print("Loading Base Model in 4-bit NormalFloat...")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True
    )

    attn_impl = "sdpa"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        quantization_config=quant_config,
        device_map="auto",
        attn_implementation=attn_impl,
        trust_remote_code=True
    )

    # 调整 Embedding 大小以容纳新 Token
    model.resize_token_embeddings(len(tokenizer))
    model = prepare_model_for_kbit_training(model)

    # 配置 LoRA：针对对齐任务，必须把词表层加入可训练！
    peft_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        modules_to_save=["embed_tokens", "lm_head"],  # SA-SID 绝对核心
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)

    # 🌟 极其危险的 Bug 修复：取消了原先的注释！
    # 强制确保 embed_tokens 能够吃到梯度，否则新加的 131 个词永远是随机数！
    if hasattr(model, "base_model") and hasattr(model.base_model.model.model, "embed_tokens"):
        model.base_model.model.model.embed_tokens.weight.requires_grad_(True)

    model.print_trainable_parameters()

    # 显式声明 label_pad_token_id，防止 Padding 污染 Loss
    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
        return_tensors="pt"
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        bf16=is_bf16_supported,
        fp16=not is_bf16_supported,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,

        # 防爆盘与 OOM 护甲
        load_best_model_at_end=False,
        save_total_limit=2,

        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        report_to="none",
        optim="paged_adamw_32bit",
        remove_unused_columns=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},

        # 🌟 引入 NEFTune 噪声，显著缓解模型背书过拟合，提高对空间和情感的泛化性
        neftune_noise_alpha=5.0
    )

    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args,
        data_collator=collator
    )

    print("🔥 开始执行 SA-SID Alignment 训练 (工业防爆版)...")
    import warnings
    warnings.filterwarnings("ignore", message=".*Token indices sequence length.*")

    trainer.train()

    # 安全地将模型保存到最终目录
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"🎉 训练完美收官！带有 SA-SID 词表的 LoRA 权重已安全保存至：{args.output_dir}")


if __name__ == "__main__":
    main()


# /root/miniconda3/envs/llm/bin/python /root/autodl-tmp/V2-SID/LLM/train/align_sid.py
# =================================================
#  🚀 启动 SA-SID 语义对齐微调 (工业完善版)
# =================================================
# Loading data from /root/autodl-tmp/V2-SID/data/NOLA/alignment/train_align.json...
# Loading Tokenizer from /root/autodl-tmp/Llama-3-8B-Instruct...
# 正在自动扫描并提取新的 SA-SID 符号...
# 成功提取 117 个全新语义符号。
# 开始预处理数据集掩码...
# Tokenizing and Masking Alignment Labels: 100%|██████████| 767/767 [00:00<00:00, 2297.91 examples/s]
# Tokenizing and Masking Alignment Labels: 100%|██████████| 43/43 [00:00<00:00, 1871.51 examples/s]
# Loading Base Model in 4-bit NormalFloat...
# Loading weights: 100%|██████████| 291/291 [00:04<00:00, 62.58it/s]
# The new embeddings will be initialized from a multivariate normal distribution that has old embeddings' mean and covariance. As described in this article: https://nlp.stanford.edu/~johnhew/vocab-expansion.html. To disable this, use `mean_resizing=False`
# The new lm_head weights will be initialized from a multivariate normal distribution that has old embeddings' mean and covariance. As described in this article: https://nlp.stanford.edu/~johnhew/vocab-expansion.html. To disable this, use `mean_resizing=False`
# /root/miniconda3/envs/llm/lib/python3.10/site-packages/peft/tuners/tuners_utils.py:1348: UserWarning: Model has `tie_word_embeddings=True` and a tied layer is part of the adapter, but `ensure_weight_tying` is not set to True. This can lead to complications, for example when merging the adapter or converting your model to formats other than safetensors. Check the discussion here: https://github.com/huggingface/peft/issues/2777
#   warnings.warn(msg)
# trainable params: 1,135,517,696 || all params: 9,166,737,408 || trainable%: 12.3874
# warmup_ratio is deprecated and will be removed in v5.2. Use `warmup_steps` instead.
# 🔥 开始执行 SA-SID Alignment 训练 (工业防爆版)...
#  56%|█████▌    | 10/18 [04:14<03:22, 25.27s/it]{'loss': '8.757', 'grad_norm': '1.079', 'learning_rate': '0.0001092', 'epoch': '1.667'}
# 100%|██████████| 18/18 [07:38<00:00, 25.32s/it]/root/miniconda3/envs/llm/lib/python3.10/site-packages/peft/utils/save_and_load.py:386: UserWarning: Setting `save_embedding_layers` to `True` as the embedding layer has been resized during finetuning.
#   warnings.warn(
# {'train_runtime': '485.3', 'train_samples_per_second': '4.742', 'train_steps_per_second': '0.037', 'train_loss': '6.095', 'epoch': '3'}
# 100%|██████████| 18/18 [08:05<00:00, 26.96s/it]
# /root/miniconda3/envs/llm/lib/python3.10/site-packages/peft/utils/save_and_load.py:386: UserWarning: Setting `save_embedding_layers` to `True` as the embedding layer has been resized during finetuning.
#   warnings.warn(
# 🎉 训练完美收官！带有 SA-SID 词表的 LoRA 权重已安全保存至：/root/autodl-tmp/V2-SID/data/NOLA/model/align_lora
#
# 进程已结束，退出代码为 0
