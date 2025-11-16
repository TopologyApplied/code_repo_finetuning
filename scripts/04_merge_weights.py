"""
独立的LoRA权重合并脚本
用于训练完成后合并LoRA权重到基础模型
"""
import torch
import os
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import yaml


def merge_lora_weights(config_path: str = "../config/default_config.yaml"):
    """合并LoRA权重到基础模型"""
    # 加载配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    base_model_path = config['model']['base_model']
    output_dir = Path(config['training']['output_dir'])
    lora_adapter_path = output_dir / "final_model"
    merged_model_path = output_dir / "merged_model"
    
    print("="*60)
    print("LoRA Weights Merging")
    print("="*60)
    print(f"Base model: {base_model_path}")
    print(f"LoRA adapter: {lora_adapter_path}")
    print(f"Output: {merged_model_path}")
    print()
    
    # 检查adapter是否存在
    if not lora_adapter_path.exists():
        print(f"❌ LoRA adapter not found at {lora_adapter_path}")
        print("Please check the training output directory.")
        return
    
    # 加载tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True
    )
    
    # 加载基础模型 - 不使用device_map避免与DeepSpeed冲突
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    
    # 加载LoRA adapter
    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(
        base_model,
        str(lora_adapter_path),
        torch_dtype=torch.bfloat16
    )
    
    # 合并权重
    print("Merging LoRA weights into base model...")
    merged_model = model.merge_and_unload()
    
    # 保存合并后的模型
    print(f"Saving merged model to {merged_model_path}...")
    merged_model_path.mkdir(parents=True, exist_ok=True)
    
    merged_model.save_pretrained(
        str(merged_model_path),
        safe_serialization=True  # 使用safetensors格式
    )
    
    tokenizer.save_pretrained(str(merged_model_path))
    
    print("✓ Merge completed successfully!")
    print()
    print("="*60)
    print(f"Merged model saved to: {merged_model_path}")
    print("="*60)
    print()
    print("Next steps:")
    print("1. Test the model: python test_model.py")
    print("2. Evaluate: python 4_model_evaluator.py")
    print("3. Generate report: python 5_generate_report.py")


if __name__ == "__main__":
    # 确保没有DeepSpeed环境变量干扰
    if 'MASTER_ADDR' in os.environ:
        del os.environ['MASTER_ADDR']
    if 'MASTER_PORT' in os.environ:
        del os.environ['MASTER_PORT']
    if 'RANK' in os.environ:
        del os.environ['RANK']
    if 'LOCAL_RANK' in os.environ:
        del os.environ['LOCAL_RANK']
    if 'WORLD_SIZE' in os.environ:
        del os.environ['WORLD_SIZE']
    
    merge_lora_weights()
