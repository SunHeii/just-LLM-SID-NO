# import os
# import argparse
# import torch
# from datasets import load_dataset
#
# from transformers import (
#     AutoModelForCausalLM,
#     AutoTokenizer,
#     TrainingArguments,
#     Trainer,  # 继续使用原生高稳定 Trainer
#     DataCollatorForSeq2Seq,
#     BitsAndBytesConfig
# )
# from peft import (
#     LoraConfig,
#     get_peft_model,
#     prepare_model_for_kbit_training
# )
#
# # 开启显存碎片整理
# os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
#
#
# def parse_args():
#     parser = argparse.ArgumentParser(description="SA-SID Sequential Recommendation SFT")
#
#     # 🌟 注意：这里使用的是携带长段历史轨迹的 LLM_data
#     parser.add_argument("--data_path", type=str, default="/root/autodl-tmp/V2-SID/data/NOLA/LLM_data/train_llm.json")
#     parser.add_argument("--val_path", type=str, default="/root/autodl-tmp/V2-SID/data/NOLA/LLM_data/val_llm.json")
#
#     # 🌟 注意：基座模型变成了我们刚刚 Merge 出来的融合模型！
#     parser.add_argument("--model_name_or_path", type=str,
#                         default="/root/autodl-tmp/V2-SID/data/NOLA/llama3-merged")
#
#     # 最终推荐模型的输出路径
#     parser.add_argument("--output_dir", type=str,
#                         default="/root/autodl-tmp/V2-SID/data/NOLA/sid-recommendation")
#
#     # =========================================================================
#     # =========================================================================
#     parser.add_argument("--batch_size", type=int, default=4)
#     parser.add_argument("--grad_accum", type=int, default=16)
#
#     parser.add_argument("--epochs", type=int, default=3)
#     parser.add_argument("--lr", type=float, default=1e-4)  # 推荐任务可以稍微降低学习率
#
#     # 🌟 轨迹历史较长，截断长度适当放宽到 ，防止截断用户的有用历史
#     parser.add_argument("--cutoff_len", type=int, default=1024)
#     return parser.parse_args()
#
#
# def preprocess_dataset(dataset, tokenizer, cutoff_len):
#     """
#     复用原生掩码机制：只对大模型预测出的 Next POI 计算 Loss
#     """
#
#     def tokenize_and_mask(example):
#         prompt = (
#             "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
#             f"{example['instruction']}<|eot_id|>"
#             "<|start_header_id|>user<|end_header_id|>\n\n"
#             f"{example['input']}<|eot_id|>"
#             "<|start_header_id|>assistant<|end_header_id|>\n\n"
#         )
#         response = f"{example['output']}<|eot_id|>"
#
#         prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
#         response_ids = tokenizer.encode(response, add_special_tokens=False)
#
#         input_ids = prompt_ids + response_ids
#         labels = [-100] * len(prompt_ids) + response_ids
#
#         if len(input_ids) > cutoff_len:
#             input_ids = input_ids[:cutoff_len]
#             labels = labels[:cutoff_len]
#
#         return {
#             "input_ids": input_ids,
#             "attention_mask": [1] * len(input_ids),
#             "labels": labels
#         }
#
#     return dataset.map(
#         tokenize_and_mask,
#         remove_columns=dataset.column_names,
#         desc="Tokenizing and Masking SFT Labels"
#     )
#
#
# def main():
#     args = parse_args()
#
#     # 关闭 wandb
#     os.environ["WANDB_DISABLED"] = "true"
#
#     print("=================================================")
#     print(" 🚀 启动 SA-SID 时序推荐监督微调 (Sequential SFT)")
#     print("=================================================")
#
#     print(f"Loading data from {args.data_path}...")
#     dataset = load_dataset("json", data_files={"train": args.data_path, "val": args.val_path})
#     train_data = dataset["train"]
#     val_data = dataset["val"]
#
#     print(f"Loading Tokenizer from merged model {args.model_name_or_path}...")
#     tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
#
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = "<|eot_id|>"
#         tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
#     tokenizer.padding_side = "right"
#
#     print("开始预处理时序轨迹数据集掩码...")
#     train_dataset = preprocess_dataset(train_data, tokenizer, args.cutoff_len)
#     val_dataset = preprocess_dataset(val_data, tokenizer, args.cutoff_len)
#
#     print("Loading Merged Model in 4-bit NormalFloat...")
#     quant_config = BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_quant_type="nf4",
#         bnb_4bit_compute_dtype=torch.bfloat16,
#         bnb_4bit_use_double_quant=True
#     )
#
#     attn_impl = "sdpa"
#
#     model = AutoModelForCausalLM.from_pretrained(
#         args.model_name_or_path,
#         quantization_config=quant_config,
#         device_map="auto",
#         attn_implementation=attn_impl,
#         trust_remote_code=True
#     )
#
#     model = prepare_model_for_kbit_training(model)
#
#     # 🌟 这次是纯推理推荐任务，我们不需要再更新 embed_tokens 词表了！
#     # 所以去掉了 modules_to_save 参数，让参数量回归到经典的极小 LoRA 级别 (约 8000万参数)
#     peft_config = LoraConfig(
#         r=32,
#         lora_alpha=64,
#         target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
#         lora_dropout=0.05,
#         bias="none",
#         task_type="CAUSAL_LM",
#     )
#     model = get_peft_model(model, peft_config)
#     model.print_trainable_parameters()
#
#     collator = DataCollatorForSeq2Seq(
#         tokenizer=tokenizer,
#         model=model,
#         padding=True,
#         return_tensors="pt"
#     )
#
#     training_args = TrainingArguments(
#         output_dir=args.output_dir,
#         per_device_train_batch_size=args.batch_size,
#         per_device_eval_batch_size=args.batch_size,
#         gradient_accumulation_steps=args.grad_accum,
#         num_train_epochs=args.epochs,
#         learning_rate=args.lr,
#         bf16=True,
#         logging_steps=10,
#         eval_strategy="steps",
#         eval_steps=100,
#         save_strategy="steps",
#         save_steps=100,
#         load_best_model_at_end=False,  # 关闭回载机制，防止 OOM
#         save_total_limit=2,  # 只保留最近两个 Checkpoint
#         warmup_ratio=0.05,
#         lr_scheduler_type="cosine",
#         report_to="none",
#         optim="paged_adamw_8bit",  # 保持 8bit 优化器，极致压缩显存
#         remove_unused_columns=False,
#         gradient_checkpointing=True,
#         gradient_checkpointing_kwargs={'use_reentrant': False}
#     )
#
#     trainer = Trainer(
#         model=model,
#         train_dataset=train_dataset,
#         eval_dataset=val_dataset,
#         args=training_args,
#         data_collator=collator
#     )
#
#     print("🔥 开始执行终局之战：SA-SID 序列推荐监督微调...")
#     import warnings
#     warnings.filterwarnings("ignore", message=".*Token indices sequence length.*")
#
#     trainer.train()
#
#     trainer.save_model(args.output_dir)
#     tokenizer.save_pretrained(args.output_dir)
#     print(f"🎉 大功告成！支持情感的下一兴趣点推荐模型已降生于：{args.output_dir}")
#
#
# if __name__ == "__main__":
#     main()

import os
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
    parser = argparse.ArgumentParser(description="SA-SID Sequential Recommendation SFT")

    parser.add_argument("--data_path", type=str, default="/root/autodl-tmp/V2-SID/data/NOLA/LLM_data/train_llm.json")
    parser.add_argument("--val_path", type=str, default="/root/autodl-tmp/V2-SID/data/NOLA/LLM_data/val_llm.json")

    parser.add_argument("--model_name_or_path", type=str,
                        default="/root/autodl-tmp/V2-SID/data/NOLA/model/llama3-merged")
    parser.add_argument("--output_dir", type=str,
                        default="/root/autodl-tmp/V2-SID/data/NOLA/model/llama3-sa-sid-recommendation")

    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--grad_accum", type=int, default=16)

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-5)

    parser.add_argument("--cutoff_len", type=int, default=1024)
    return parser.parse_args()


def preprocess_dataset(dataset, tokenizer, cutoff_len):
    """
    🌟 核心重构：语义级历史轨迹折叠 (Semantic History Truncation)
    完美保护 Llama-3 Chat 模板，丢弃远古记录，保留近期核心轨迹。
    """

    def tokenize_and_mask(example):
        instruction = example['instruction']
        raw_input = example['input']
        output = example['output']

        # 1. 结构化拆解：精准剥离出“历史轨迹”和“静态问题/指令”
        try:
            # 提取前缀
            prefix_part, rest = raw_input.split(" trajectory history: ", 1)
            prefix_part += " trajectory history: "
            # 提取轨迹段与尾部问题
            history_str, question_part = rest.split(".\nQuestion:", 1)
            question_part = ".\nQuestion:" + question_part
        except ValueError:
            # 容错机制：如果格式匹配失败，退化处理
            prefix_part, history_str, question_part = "", raw_input, ""

        # 轨迹切分成独立记录
        traj_list = history_str.split(", then ")

        # 2. 组装“不可或缺”的静态结构部分
        sys_prompt = (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{instruction}<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            f"{prefix_part}"
        )
        qa_prompt = f"{question_part}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        resp_prompt = f"{output}<|eot_id|>"

        # 提前 Tokenize 静态层，计算固定开销
        sys_ids = tokenizer.encode(sys_prompt, add_special_tokens=False)
        qa_ids = tokenizer.encode(qa_prompt, add_special_tokens=False)
        resp_ids = tokenizer.encode(resp_prompt, add_special_tokens=False)

        # 3. 计算留给“历史轨迹”的 Token 余额
        static_len = len(sys_ids) + len(qa_ids) + len(resp_ids)
        available_len = cutoff_len - static_len

        # 4. 倒序装载：优先保留最新的打卡记录
        history_ids = []
        # reversed() 确保从最新的(尾部)打卡记录开始塞入
        for i, traj in enumerate(reversed(traj_list)):
            if i == 0:
                traj_tokens = tokenizer.encode(traj, add_special_tokens=False)
            else:
                traj_tokens = tokenizer.encode(", then " + traj, add_special_tokens=False)

            # 检查余额是否足够
            if len(history_ids) + len(traj_tokens) <= available_len:
                # 既然是倒序遍历，新进来的记录要放在最前面
                history_ids = traj_tokens + history_ids
            else:
                # 余额不足，直接跳出。更早的轨迹被自然抛弃！
                # 极端情况兜底：如果连 1 条最新记录都塞不下，就对这一条执行截断
                if len(history_ids) == 0:
                    history_ids = traj_tokens[-available_len:] if available_len > 0 else []
                break

        # 5. 无缝拼装完整的 Input 与 Label
        prompt_ids = sys_ids + history_ids + qa_ids
        input_ids = prompt_ids + resp_ids

        # 标签掩码：只对最终的 response_ids 计算 Loss，其他全部设为 -100
        labels = [-100] * len(prompt_ids) + resp_ids

        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels
        }

    return dataset.map(
        tokenize_and_mask,
        remove_columns=dataset.column_names,
        desc="Tokenizing & Semantic Truncation"
    )


def main():
    args = parse_args()

    # 关闭 wandb
    os.environ["WANDB_DISABLED"] = "true"

    print("=================================================")
    print(" 🚀 启动 SA-SID 时序推荐监督微调 (语义动态截断版)")
    print("=================================================")

    print(f"Loading data from {args.data_path}...")
    dataset = load_dataset("json", data_files={"train": args.data_path, "val": args.val_path})
    train_data = dataset["train"]
    val_data = dataset["val"]

    print(f"Loading Tokenizer from merged model {args.model_name_or_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = "<|eot_id|>"
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    tokenizer.padding_side = "right"

    print("开始预处理时序轨迹数据集掩码...")
    train_dataset = preprocess_dataset(train_data, tokenizer, args.cutoff_len)
    val_dataset = preprocess_dataset(val_data, tokenizer, args.cutoff_len)

    # 自适应硬件浮点精度
    is_bf16_supported = torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if is_bf16_supported else torch.float16

    print("Loading Merged Model in 4-bit...")
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

    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
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
        load_best_model_at_end=False,
        save_total_limit=2,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        report_to="none",
        optim="paged_adamw_8bit",
        remove_unused_columns=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        # 引入 NEFTune 噪声，显著提升 SFT 推荐模型的泛化推理能力
        neftune_noise_alpha=5.0
    )

    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args,
        data_collator=collator
    )

    print("🔥 开始执行终局之战：SA-SID 序列推荐监督微调...")
    import warnings
    warnings.filterwarnings("ignore", message=".*Token indices sequence length.*")

    trainer.train()

    # 训练完成，安全保存
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"🎉 大功告成！支持情感的下一兴趣点推荐模型已降生于：{args.output_dir}")

    # =========================================================
    # 自动化推理测试模块 (Live Inference Test)
    # =========================================================
    print("\n" + "=" * 50)
    print(" 🔍 开始自动抽取验证集样本，检验模型预测效果...")
    print("=" * 50)

    # 清理一下显存，防止 OOM
    torch.cuda.empty_cache()
    model.eval()

    # 随机挑一条验证集数据进行截断预测演示
    test_sample = val_data[0]

    # 为了测试效果，我们在这里也要用同样的切分方法保证测试集能够输入
    try:
        prefix_part, rest = test_sample['input'].split(" trajectory history: ", 1)
        prefix_part += " trajectory history: "
        history_str, question_part = rest.split(".\nQuestion:", 1)
        question_part = ".\nQuestion:" + question_part
    except ValueError:
        prefix_part, history_str, question_part = "", test_sample['input'], ""

    traj_list = history_str.split(", then ")
    # 测试时直接保留最后 10 次记录，保证绝对不会超出显存
    short_history = ", then ".join(traj_list[-10:])
    safe_input = prefix_part + short_history + question_part

    eval_prompt = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{test_sample['instruction']}<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{safe_input}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )

    inputs = tokenizer(eval_prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=15,
            eos_token_id=tokenizer.convert_tokens_to_ids("<|eot_id|>"),
            pad_token_id=tokenizer.pad_token_id,
            temperature=0.1,
            do_sample=False
        )

    input_length = inputs.input_ids.shape[1]
    prediction = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)

    print(f"\n📝 【用户近期核心输入 Prompt】:\n{safe_input[:300]} ... (已折叠)")
    print(f"\n✅ 【真实下一个地点 Ground Truth】: {test_sample['output']}")
    print(f"🤖 【大模型预测结果 Prediction】:   {prediction.strip()}")
    print("\n💡 如果预测结果与真实地点在宏观层(如前两位)或精确匹配，说明模型已成功建立地理情感常识！")


if __name__ == "__main__":
    main()