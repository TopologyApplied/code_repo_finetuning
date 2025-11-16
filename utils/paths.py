"""
统一的路径管理模块
"""
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

# 配置文件路径
CONFIG_DIR = PROJECT_ROOT / "config"
DEFAULT_CONFIG = CONFIG_DIR / "default_config.yaml"
DEEPSPEED_CONFIG = CONFIG_DIR / "deepspeed_zero3.json"

# 数据路径
DATA_DIR = PROJECT_ROOT / "data"
REPO_ANALYSIS_FILE = DATA_DIR / "repository_analysis.json"
TRAINING_DATA_DIR = DATA_DIR / "training_data"

# 输出路径
OUTPUT_DIR = PROJECT_ROOT / "output"
MODEL_OUTPUT_DIR = OUTPUT_DIR / "finetuned_model"

# 仓库路径
REPOS_DIR = PROJECT_ROOT / "repos"

def get_repo_path(repo_name: str) -> Path:
    """获取特定仓库路径"""
    return REPOS_DIR / repo_name