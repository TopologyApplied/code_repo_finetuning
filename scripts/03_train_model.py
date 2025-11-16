"""
修复版模型微调脚本
核心改进:
1. 鲁棒的标签掩码（只学习assistant的回答）- 最终、最鲁棒修正版
2. 解决 QwenTokenizer 没有 im_end_id 属性的兼容性问题。
3. 修复 TypeError: '<=' not supported between instances of 'float' and 'str' 问题。
"""
import os
import json
import yaml
import torch
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
import numpy as np


@dataclass
class ModelArguments:
    """模型参数"""
    model_name_or_path: str = field(default="Qwen/Qwen3-8B")
    use_lora: bool = field(default=True)
    lora_r: int = field(default=64)  
    lora_alpha: int = field(default=128)  
    lora_dropout: float = field(default=0.05)
    lora_target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj", 
            "gate_proj", "up_proj", "down_proj"
        ]
    )


@dataclass
class DataArguments:
    """数据参数"""
    data_dir: str = field(default="./data/training_data")
    max_length: int = field(default=1024)
    preprocessing_num_workers: int = field(default=32)


class SampleInspectionCallback(TrainerCallback):
    """训练样本检查回调"""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.checked = False
    
    def on_step_begin(self, args, state, control, **kwargs):
        """在第一步开始时检查样本"""
        if not self.checked and state.global_step == 0:
            print("\n" + "="*60)
            print("🔍 Inspecting training samples...")
            print("="*60)
            self.checked = True


class QwenFineTunerFixed:
    """Qwen模型微调器 - 修复版"""
    config_path = Path(__file__).parent.parent / "config" / "default_config.yaml"

    def __init__(self, config_path: str = config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.model_args = ModelArguments(
            model_name_or_path=self.config['model']['base_model']
        )
        self.data_args = DataArguments(
            data_dir=self.config['dataset']['output_dir']
        )
        
        self.output_dir = Path(self.config['training']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.tokenizer = None
        self.model = None
        self.train_dataset = None
        self.eval_dataset = None
        # 新增属性：用于安全存储 im_end_id
        self.im_end_token_id = None 
        
    def load_tokenizer_and_model(self):
        """加载tokenizer和模型"""
        print(f"Loading tokenizer from {self.model_args.model_name_or_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_args.model_name_or_path,
            trust_remote_code=True,
            padding_side='right'
        )
        
        # 安全获取 im_end_id (修复 im_end_id 属性错误)
        try:
             # Qwen token ID 是 151644
             self.im_end_token_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
             if self.im_end_token_id is None:
                 raise ValueError("Could not convert <|im_end|> token to ID.")
        except Exception as e:
             print(f"Warning: Could not get <|im_end|> ID, trying fallback: {e}")
             self.im_end_token_id = self.tokenizer.eos_token_id
        print(f"Using im_end_id: {self.im_end_token_id}")

        
        # 设置pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
        if self.tokenizer.chat_template is None:
             print("Warning: Qwen chat template not found. Using default template logic.")
        
        print(f"Loading model from {self.model_args.model_name_or_path}")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_args.model_name_or_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            use_cache=False,
            low_cpu_mem_usage=True
        )
        
        # 准备LoRA
        print("Preparing model for LoRA training...")
        if self.model_args.use_lora:
             
            print("Applying LoRA configuration")
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.model_args.lora_r,
                lora_alpha=self.model_args.lora_alpha,
                lora_dropout=self.model_args.lora_dropout,
                target_modules=self.model_args.lora_target_modules,
                bias="none",
                inference_mode=False,
            )
            
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
            self.model.train()
            
            # 验证
            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f"✓ Trainable parameters: {trainable:,}")
    
    def load_and_preprocess_data(self):
        """加载和预处理数据"""
        print("Loading datasets...")
        
        data_files = {
            'train': str(Path(self.data_args.data_dir) / 'train.jsonl'),
            'validation': str(Path(self.data_args.data_dir) / 'val.jsonl'),
        }
        
        raw_datasets = load_dataset('json', data_files=data_files)
        
        print("Preprocessing datasets...")
        self.train_dataset = raw_datasets['train'].map(
            self._preprocess_function,
            batched=True,
            num_proc=self.data_args.preprocessing_num_workers,
            remove_columns=raw_datasets['train'].column_names,
            desc="Preprocessing train dataset"
        )
        
        self.eval_dataset = raw_datasets['validation'].map(
            self._preprocess_function,
            batched=True,
            num_proc=self.data_args.preprocessing_num_workers,
            remove_columns=raw_datasets['validation'].column_names,
            desc="Preprocessing validation dataset"
        )
        
        # 过滤过长样本
        print("Filtering samples...")
        self.train_dataset = self.train_dataset.filter(
            lambda x: x is not None and len(x['input_ids']) <= self.data_args.max_length
        )
        self.eval_dataset = self.eval_dataset.filter(
            lambda x: x is not None and len(x['input_ids']) <= self.data_args.max_length
        )
        
        print(f"✓ Train samples: {len(self.train_dataset)}")
        print(f"✓ Validation samples: {len(self.eval_dataset)}")
        
        # 检查第一个样本
        if len(self.train_dataset) > 0:
            self._inspect_sample(self.train_dataset[0])
    
    def _preprocess_function(self, examples):
        """预处理函数 - 最终、最鲁棒修正版标签掩码"""
        model_inputs = {
            "input_ids": [],
            "attention_mask": [],
            "labels": []
        }
        
        for conversations in examples['conversations']:
            try:
                # 1. 完整对话文本
                full_text = self.tokenizer.apply_chat_template(
                    conversations,
                    tokenize=False,
                    add_generation_prompt=False 
                )
                
                # 找到最后一个 Assistant 消息的索引
                last_assistant_index = next((i for i, msg in reversed(list(enumerate(conversations))) if msg['role'] == 'assistant'), -1)
                
                if last_assistant_index == -1:
                    print("Warning: Skipping conversation with no assistant reply.")
                    continue
                
                # 构造 "仅问题" 的对话列表: 包含所有消息直到最后一个 Assistant 消息之前
                prompt_messages = conversations[:last_assistant_index]
                # 加上最后一个 Assistant 消息的 Role Prompt (例如 <|im_start|>assistant\n)
                prompt_messages.append({"role": "assistant", "content": ""}) 
                
                prompt_text = self.tokenizer.apply_chat_template(
                    prompt_messages,
                    tokenize=False,
                    add_generation_prompt=False 
                )
                
                # 3. 分词: 完整对话
                tokenized_full = self.tokenizer(
                    full_text,
                    max_length=self.data_args.max_length,
                    truncation=True,
                    padding=False,
                )
                
                # 4. 分词: 仅问题部分 (获取答案起始点)
                tokenized_prompt = self.tokenizer(
                    prompt_text,
                    max_length=self.data_args.max_length,
                    truncation=True,
                    padding=False,
                )
                
                input_ids = tokenized_full['input_ids']
                labels = input_ids.copy()
                
                # 答案内容的起始索引 = 仅问题部分的长度
                answer_start_index = len(tokenized_prompt['input_ids'])
                
                if answer_start_index >= len(labels):
                    print(f"Warning: Answer start index {answer_start_index} exceeds or matches total length {len(labels)}. Skipping.")
                    return None

                # 5. 标签掩码:
                # 掩盖掉答案起始点之前的所有 tokens
                labels[:answer_start_index] = [-100] * answer_start_index
                
                # 确保最后一个 token (通常是 EOS/PAD 或 <|im_end|>) 也被掩盖
                if len(labels) > 0:
                    last_token_id = labels[-1]
                    
                    # 检查是否是 EOS/PAD token
                    if last_token_id != -100 and last_token_id == self.tokenizer.eos_token_id:
                        labels[-1] = -100
                    
                    # 检查是否是 Qwen 的 <|im_end|> token (使用安全存储的 ID)
                    if self.im_end_token_id is not None and last_token_id != -100 and last_token_id == self.im_end_token_id:
                        labels[-1] = -100
                
                model_inputs["input_ids"].append(input_ids)
                model_inputs["attention_mask"].append(tokenized_full['attention_mask'])
                model_inputs["labels"].append(labels)
            
            except Exception as e:
                import sys
                import traceback
                traceback.print_exc(file=sys.stdout)
                print(f"Error processing conversation: {e}")
                return None
                
        return model_inputs
    
    # ... (_inspect_sample 方法保持不变)
    def _inspect_sample(self, sample):
        """检查样本质量"""
        print("\n" + "="*60)
        print("🔍 Sample Inspection (AFTER FINAL, MOST ROBUST FIXES)")
        print("="*60)
        
        input_ids = sample['input_ids']
        labels = sample['labels']
        
        # 解码
        input_text = self.tokenizer.decode(input_ids, skip_special_tokens=False)
        
        # 统计
        total_tokens = len(input_ids)
        masked_tokens = sum(1 for l in labels if l == -100)
        learning_tokens = total_tokens - masked_tokens
        
        print(f"Total tokens: {total_tokens}")
        print(f"Masked tokens (prompt/padding): {masked_tokens} ({masked_tokens/total_tokens*100:.1f}%)")
        print(f"Learning tokens (assistant): {learning_tokens} ({learning_tokens/total_tokens*100:.1f}%)")
        
        # 显示前200个token的掩码情况
        print("\n📊 First 200 tokens masking pattern:")
        preview_len = min(200, len(labels))
        mask_preview = ''.join(['█' if labels[i] == -100 else '░' for i in range(preview_len)])
        
        # 找到第一个学习 token 和第一个掩码 token
        first_learn_idx = next((i for i, l in enumerate(labels) if l != -100), -1)
        
        if first_learn_idx != -1:
             print(f"First 10 tokens: {self.tokenizer.decode(input_ids[:10], skip_special_tokens=False)}")
             print(f"First learning token index: {first_learn_idx}")
             print(f"First learning token: {self.tokenizer.decode(input_ids[first_learn_idx])}")
             # 打印学习内容周围的 tokens
             start = max(0, first_learn_idx - 5)
             end = min(len(input_ids), first_learn_idx + 5)
             print(f"Around learning start: {self.tokenizer.decode(input_ids[start:end], skip_special_tokens=False)}")

        print(mask_preview)
        print("█ = masked (prompt/padding) | ░ = learning (assistant)")
        
        # 显示学习内容示例
        learning_ids = [input_ids[i] for i in range(len(labels)) if labels[i] != -100]
        if learning_ids:
            learning_text = self.tokenizer.decode(learning_ids[:100], skip_special_tokens=True)
            print(f"\n📝 Learning content preview:")
            print(f"{learning_text[:200]}...")
        
        print("="*60 + "\n")
    
    def train(self):
        """训练模型"""
        print("Setting up training arguments...")
        
        # 改进的训练配置
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=self.config['training']['num_epochs'],
            
            # 批次配置
            per_device_train_batch_size=2, 
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=8, 
            
            # 学习率
            learning_rate=float(self.config['training']['learning_rate']), # <--- 修复: 强制类型转换 float
            warmup_ratio=float(self.config['training']['warmup_ratio']), # <--- 修复: 强制类型转换 float
            lr_scheduler_type="cosine",
            
            # 优化器
            optim="adamw_torch",
            weight_decay=float(self.config['training']['weight_decay']), # <--- 修复: 强制类型转换 float
            max_grad_norm=float(self.config['training']['max_grad_norm']), # <--- 修复: 强制类型转换 float
            
            # 日志和保存
            logging_steps=10,
            save_steps=100,
            eval_steps=100,
            save_total_limit=3,
            
            # 评估
            eval_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            
            # 精度
            bf16=True,
            bf16_full_eval=True,
            
            # DeepSpeed
            deepspeed="../config/deepspeed_zero3.json",
            
            # 其他
            report_to=["tensorboard"],
            logging_dir=str(self.output_dir / "logs"),
            remove_unused_columns=False,
            dataloader_pin_memory=True,
            dataloader_num_workers=0,
            logging_first_step=True,
            logging_nan_inf_filter=True,
        )
        
        # Data collator
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            label_pad_token_id=-100,
            padding=True,
        )
        
        # Callbacks
        callbacks = [SampleInspectionCallback(self.tokenizer)]
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            callbacks=callbacks,
        )
        
        # 训练前验证
        print("\n" + "="*60)
        print("Pre-training Validation")
        print("="*60)
        print(f"✓ Model in training mode: {self.model.training}")
        
        lora_params = sum(p.numel() for n, p in self.model.named_parameters() 
                         if p.requires_grad and 'lora' in n.lower())
        print(f"✓ LoRA parameters: {lora_params:,}")
        
        # 开始训练
        print("\n" + "="*60)
        print("Starting Training")
        print("="*60)
        
        train_result = trainer.train()
        
        # 保存
        print("\nSaving model...")
        trainer.save_model(str(self.output_dir / "final_model"))
        
        # 保存指标
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        
        # 评估
        print("\nEvaluating...")
        eval_metrics = trainer.evaluate()
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)
        
        print("\n✓ Training completed!")
        return trainer


def main():
    """主函数"""
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
         os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    if 'TOKENIZERS_PARALLELISM' not in os.environ:
         os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    if 'PYTORCH_CUDA_ALLOC_CONF' not in os.environ:
         os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    print("="*60)
    print("Qwen3-8B Fine-tuning - Fixed Version (Label Masking/LoRA Params Improved)")
    print("="*60)
    print()
    
    finetuner = QwenFineTunerFixed()
    finetuner.load_tokenizer_and_model()
    finetuner.load_and_preprocess_data()
    trainer = finetuner.train()
    
    print("\n" + "="*60)
    print("✓ Fine-tuning Complete!")
    print(f"Model saved to: {finetuner.output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()

