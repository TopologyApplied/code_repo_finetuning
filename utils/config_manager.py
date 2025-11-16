"""
更新config.yaml中的代码仓库配置
"""
import yaml
from pathlib import Path
import sys


def update_repository_config(repo_url: str, local_path: str = None):
    """更新config.yaml中的仓库配置"""
    config_file = Path(__file__).parent.parent / "config" / "default_config.yaml"
    #config_file = Path("config.yaml")
    
    if not config_file.exists():
        print("❌ 错误: 找不到 config.yaml")
        return False
    
    # 读取现有配置
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 提取仓库名
    repo_name = Path(repo_url).stem
    
    # 更新仓库配置
    if local_path is None:
        local_path = f"./repos/{repo_name}"
    
    config['repository']['url'] = repo_url
    config['repository']['local_path'] = local_path
    
    # 备份原配置
    #backup_file = config_file.with_suffix('.yaml.backup')
    backup_file = config_file.parent / (config_file.stem + '.backup.yaml')

    with open(backup_file, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
    
    print(f"✓ 原配置已备份至: {backup_file}")
    
    # 保存新配置
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
    
    print(f"✓ 配置已更新:")
    print(f"  仓库名称: {repo_name}")
    print(f"  仓库URL: {repo_url}")
    print(f"  本地路径: {local_path}")
    
    return True


def main():
    """主函数"""
    print("="*70)
    print("代码仓库配置更新工具")
    print("="*70)
    print()
    
    if len(sys.argv) > 1:
        repo_url = sys.argv[1]
        local_path = sys.argv[2] if len(sys.argv) > 2 else None
    else:
        print("请输入新的代码仓库URL:")
        print("示例: https://github.com/gamosoft/NoteDiscovery.git")
        print()
        repo_url = input("仓库URL: ").strip()
        
        if not repo_url:
            print("❌ 未输入URL,已取消")
            return
        
        print()
        use_default_path = input("使用默认本地路径? (y/n, 默认y): ").strip().lower()
        
        if use_default_path == 'n':
            local_path = input("本地路径: ").strip()
        else:
            local_path = None
    
    print()
    success = update_repository_config(repo_url, local_path)
    
    if success:
        print()
        print("="*70)
        print("✅ 配置更新成功!")
        print()
        print("下一步:")
        print("  1. 运行知识检测: python test_base_model_knowledge.py")
        print("  2. 如果检测通过,开始训练:")
        print("     python 1_repository_analyzer.py")
        print("     python 2_data_generator.py")
        print("     deepspeed --num_gpus=2 3_model_finetuner_v4_OOM_FIX.py")
        print("="*70)


if __name__ == "__main__":
    main()
