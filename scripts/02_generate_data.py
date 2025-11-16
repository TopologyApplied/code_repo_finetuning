"""
修复版训练数据生成器
核心改进:
1. 直接基于代码内容生成准确的问答对
2. 不依赖LLM生成（避免循环依赖）
3. 使用模板化方法确保数据质量
4. 优化项目概览问题，使其更具项目特色
"""
import json
import yaml
import random
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass, field # <--- 修复: dataclass 位于 dataclasses 模块
import re
from collections import defaultdict


@dataclass
class TrainingSample:
    """训练样本"""
    conversations: List[Dict[str, str]]
    metadata: Dict[str, Any]


class FixedDataGenerator:
    """修复版数据生成器 - 基于规则和模板"""
    
    def __init__(self, config_path: str = "../config/default_config.yaml", 
                 analysis_path: str = "../data/repository_analysis.json"):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        try:
            with open(analysis_path, 'r', encoding='utf-8') as f:
                self.analysis_data = json.load(f)
        except FileNotFoundError:
            print(f"❌ 警告: 找不到分析文件 {analysis_path}。请先运行分析器。")
            self.analysis_data = {'code_elements': [], 'project_context': {}}

        self.code_elements = self.analysis_data.get('code_elements', [])
        self.project_context = self.analysis_data.get('project_context', {})
        self.project_name = self.project_context.get('project_name', 'Laddr')
        
        self.training_samples = []
    
    def generate_training_data(self):
        """生成训练数据"""
        print(f"Generating training data for {self.project_name}...")
        
        # 1. 代码解释任务（基于docstring + 代码结构）
        print("Generating code explanation samples...")
        self._generate_code_explanation_samples()
        
        # 2. API使用示例（基于函数签名 + docstring）
        print("Generating API usage samples...")
        self._generate_api_usage_samples()
        
        # 3. 项目概览问答（基于统计和结构信息）
        print("Generating project overview samples...")
        self._generate_project_overview_samples()
        
        # 4. 代码定位任务（"在哪个文件中..."）
        print("Generating code location samples...")
        self._generate_code_location_samples()
        
        print(f"Total samples generated: {len(self.training_samples)}")
    
    def _generate_code_explanation_samples(self):
        """生成代码解释样本 - 基于真实代码和docstring"""
        # 选择有docstring的元素
        candidates = [e for e in self.code_elements 
                     if e.get('docstring') and len(e.get('code', '')) > 50]
        
        for element in candidates[:300]:  # 增加数量限制
            name = element['name']
            docstring = element['docstring']
            filepath = element['filepath']
            element_type = element['type']
            code = element.get('code', '')
            
            # 提取函数签名
            signature = self._extract_signature(code, element_type)
            
            # 问题模板
            questions = [
                f"请解释 {self.project_name} 中 `{name}` 的作用。",
                f"{self.project_name} 的 `{name}` 是做什么的？",
                f"在 {self.project_name} 项目中，`{name}` 有什么功能？",
            ]
            question = random.choice(questions)
            
            # 构建高质量答案（基于真实信息）
            answer_parts = []
            
            # 1. 基本信息
            answer_parts.append(f"`{name}` 是 {self.project_name} 项目中的一个 {self._type_to_cn(element_type)}，位于 `{filepath}`。")
            
            # 2. 功能描述（来自docstring）
            if docstring:
                # 清理docstring
                clean_doc = self._clean_docstring(docstring)
                answer_parts.append(f"\n**功能描述**：\n{clean_doc}")
            
            # 3. 函数签名（如果有）
            if signature:
                answer_parts.append(f"\n**函数签名**：\n```python\n{signature}\n```")
            
            # 4. 参数说明（如果有）
            params = element.get('parameters', [])
            if params and len(params) > 0:
                param_desc = "\n**参数**：\n"
                for param in params[:5]:  # 最多5个参数
                    param_name = param.get('name', 'unknown')
                    param_type = param.get('type', 'Any')
                    # 尝试从 docstring 中提取参数描述，如果没有则使用类型
                    param_desc_from_doc = self._extract_param_desc(docstring, param_name)
                    if param_desc_from_doc:
                        param_info = f"- `{param_name}` ({param_type}): {param_desc_from_doc}\n"
                    else:
                        param_info = f"- `{param_name}` ({param_type})\n"

                    param_desc += param_info
                answer_parts.append(param_desc)
            
            # 5. 返回值（如果有）
            return_type = element.get('return_type')
            if return_type:
                answer_parts.append(f"\n**返回值**：`{return_type}`")
            
            answer = ''.join(answer_parts)
            
            self.training_samples.append(TrainingSample(
                conversations=[
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer}
                ],
                metadata={
                    "task_type": "code_explanation",
                    "element_name": name,
                    "filepath": filepath
                }
            ))
    
    def _generate_api_usage_samples(self):
        """生成API使用示例 - 基于函数签名"""
        # 选择公共函数/方法
        candidates = [e for e in self.code_elements 
                     if e['type'] in ['function', 'method']
                     and not e['name'].startswith('_')  # 排除私有方法
                     and e.get('parameters')]
        
        for element in candidates[:150]: # 增加数量限制
            name = element['name']
            params = element.get('parameters', [])
            filepath = element['filepath']
            docstring = element.get('docstring', '')
            
            question = f"如何在 {self.project_name} 中使用 `{name}` 函数？"
            
            # 构建使用示例
            answer_parts = []
            answer_parts.append(f"`{name}` 位于 `{filepath}`，使用方法如下：")
            
            # 生成示例代码
            param_names = [p['name'] for p in params if p['name'] != 'self']
            if param_names:
                example_code = f"{name}("
                param_examples = []
                for p in param_names[:5]:  # 最多5个参数
                    param_examples.append(f"{p}=...")
                example_code += ", ".join(param_examples)
                example_code += ")"
                
                answer_parts.append(f"\n```python\n{example_code}\n```")
            
            # 参数说明
            if params:
                answer_parts.append("\n**参数说明**：")
                for param in params[:5]:
                    if param['name'] != 'self':
                        param_type = param.get('type', 'Any')
                        
                        param_desc_from_doc = self._extract_param_desc(docstring, param['name'])
                        
                        answer_parts.append(f"\n- `{param['name']}`: {param_type}")
                        if param_desc_from_doc:
                             answer_parts[-1] += f" - {param_desc_from_doc}" # 追加描述
            
            # 添加docstring提示
            if docstring:
                clean_doc = self._clean_docstring(docstring)[:200]
                if clean_doc:
                    answer_parts.append(f"\n\n**功能简述**：{clean_doc}...")
            
            answer = ''.join(answer_parts)
            
            self.training_samples.append(TrainingSample(
                conversations=[
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer}
                ],
                metadata={
                    "task_type": "api_usage",
                    "element_name": name
                }
            ))
    
    def _generate_project_overview_samples(self):
        """生成项目概览问答 - 基于统计信息"""
        stats = self.analysis_data.get('statistics', {})
        description = self.project_context.get('description', '')
        techs = self.project_context.get('main_technologies', [])
        file_type_counts = self.analysis_data.get('statistics', {}).get('file_type_counts', {})
        
        # --- 问题1: 项目主要功能 (更具项目特色) ---
        q1_list = [
            f"请用一句话描述 {self.project_name} 项目的主要功能。",
            f"{self.project_name} 是一个什么样的项目？",
            f"简单介绍一下 {self.project_name} 项目。"
        ]
        q1 = random.choice(q1_list)
        
        a1_parts = [
            f"{self.project_name} 是一个 Python 项目。"
        ]
        
        if description:
            # 修复：确保项目描述清晰
            a1_parts.append(f"\n**核心目标**：\n{description}")
        else:
             a1_parts.append("\n**核心目标**：此项目旨在提供一个可扩展的多代理系统框架（Agent Framework），支持任务规划、工具调用、消息队列和数据库集成等功能。")
        
        # 添加技术栈
        if techs:
            a1_parts.append(f"\n\n**主要技术栈**：{', '.join(techs[:5])}等。")
        
        a1_parts.append(f"\n\n项目包含 {stats.get('total_elements', 0)} 个代码元素，主要由 {stats.get('classes', 0)} 个类和 {stats.get('functions', 0) + stats.get('methods', 0)} 个函数/方法构成。")
        
        a1 = ''.join(a1_parts)
        
        self.training_samples.append(TrainingSample(
            conversations=[
                {"role": "user", "content": q1},
                {"role": "assistant", "content": a1}
            ],
            metadata={"task_type": "project_overview"}
        ))
        
        # --- 问题2: 项目结构 ---
        q2_list = [
            f"{self.project_name} 的项目结构是怎样的？",
            f"请列举 {self.project_name} 的核心模块。",
        ]
        q2 = random.choice(q2_list)
        
        a2_parts = [f"{self.project_name} 项目包含以下主要部分：\n"]
        
        # 获取主要模块
        modules = self.project_context.get('key_modules', [])
        if modules:
            a2_parts.append("\n**核心模块**：\n")
            for mod in modules[:10]:
                a2_parts.append(f"- `{mod}`\n")
        else:
            a2_parts.append("\n**核心模块**：\n- `core` (核心逻辑，如Agent Runtime, Tooling, Config)\n- `cli` (命令行接口)\n- `llms` (LLM后端实现)\n")

        # 优化文件类型展示
        if file_type_counts:
             file_stats = ', '.join(f'{k.lstrip(".").upper()}: {v}' for k, v in file_type_counts.items() if k not in ['.other'])
             a2_parts.append(f"\n**主要文件类型统计**：{file_stats}")
        
        a2 = ''.join(a2_parts)
        
        self.training_samples.append(TrainingSample(
            conversations=[
                {"role": "user", "content": q2},
                {"role": "assistant", "content": a2}
            ],
            metadata={"task_type": "project_structure"}
        ))
        
        # --- 问题3: 核心类/函数 ---
        top_elements = sorted(self.code_elements, 
                             key=lambda x: x.get('complexity', 0), 
                             reverse=True)[:10]
        
        q3 = f"{self.project_name} 中有哪些核心类和函数？"
        a3_parts = [f"{self.project_name} 的核心组件包括（基于复杂度和重要性）：\n"]
        
        for elem in top_elements:
            name = elem['name']
            filepath = elem['filepath']
            elem_type = self._type_to_cn(elem['type'])
            
            doc = elem.get('docstring', '')
            short_doc = self._clean_docstring(doc).split('\n')[0][:80].strip()
            
            line = f"\n- `{name}` ({elem_type})：位于 `{filepath}`"
            if short_doc:
                line += f" - {short_doc}..."
            a3_parts.append(line)
        
        if len(top_elements) > 0:
             a3 = ''.join(a3_parts)
             self.training_samples.append(TrainingSample(
                conversations=[
                    {"role": "user", "content": q3},
                    {"role": "assistant", "content": a3}
                ],
                metadata={"task_type": "core_components"}
            ))
    
    def _generate_code_location_samples(self):
        """生成代码定位任务"""
        # 选择不同文件中的元素
        file_elements = defaultdict(list)
        for elem in self.code_elements:
            # 排除非核心的__init__
            if elem['name'] == '__init__' and 'module' not in elem['type']:
                continue
            file_elements[elem['filepath']].append(elem)
        
        for filepath, elements in list(file_elements.items())[:50]:
            # 随机选择1-3个元素
            selected = random.sample(elements, min(3, len(elements)))
            
            for elem in selected:
                name = elem['name']
                elem_type = self._type_to_cn(elem['type'])
                
                question = f"在 {self.project_name} 中，`{name}` {elem_type}在哪个文件里？"
                
                # 答案优化：更简洁，减少冗余信息，模型只需学习路径
                answer = f"`{name}` 位于 `{filepath}`。"
                
                self.training_samples.append(TrainingSample(
                    conversations=[
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": answer}
                    ],
                    metadata={
                        "task_type": "code_location",
                        "element_name": name,
                        "filepath": filepath
                    }
                ))
    
    def _extract_signature(self, code: str, element_type: str) -> str:
        """提取函数/类签名"""
        if not code:
            return ""
        
        lines = code.strip().split('\n')
        signature_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            signature_lines.append(line)
            
            # 提取函数/方法定义行
            if element_type in ['function', 'method'] and (line.startswith('def ') or line.startswith('async def ')):
                # 兼容多行函数签名
                if not line.endswith(':'):
                    continue
                return '\n'.join(signature_lines)
            
            # 提取类定义行
            if element_type == 'class' and line.startswith('class '):
                 if not line.endswith(':'):
                    continue
                 return '\n'.join(signature_lines)

            # 避免包含函数/方法体
            if line.endswith((':')) and not line.startswith(('def ', 'class ')):
                break

        # 仅返回前几行，确保只包含定义
        return '\n'.join(signature_lines[:5])
    
    def _clean_docstring(self, docstring: str) -> str:
        """清理docstring"""
        if not docstring:
            return ""
        
        # 移除多余空白
        lines = docstring.strip().split('\n')
        cleaned = []
        for line in lines:
            line = line.strip()
            if line:
                cleaned.append(line)
        
        return ' '.join(cleaned)

    def _extract_param_desc(self, docstring: str, param_name: str) -> str:
        """从 docstring 中尝试提取参数描述"""
        if not docstring:
            return ""
        # 匹配各种格式的参数描述，例如 Args: key: The cache key.
        match = re.search(rf"(?:Args|Parameters|Params):\s*(?:[\n\r]\s*-)?\s*`?{re.escape(param_name)}`?\s*[:\-]\s*(.*)", docstring, re.IGNORECASE)
        if match:
             desc = match.group(1).split('\n')[0].strip()
             return desc if desc else "无描述"
        return ""
    
    def _type_to_cn(self, element_type: str) -> str:
        """元素类型转中文"""
        mapping = {
            'function': '函数',
            'method': '方法',
            'class': '类',
            'variable': '变量',
            'module': '模块'
        }
        return mapping.get(element_type, element_type)
    
    def save_training_data(self):
        """保存训练数据"""
        output_dir = Path(self.config['dataset']['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 打乱
        random.shuffle(self.training_samples)
        
        # 分割
        total = len(self.training_samples)
        train_size = int(total * 0.8)
        val_size = int(total * 0.1)
        
        if total < 10: # 如果样本太少，平均分配
             train_size = max(1, total // 2)
             val_size = max(1, (total - train_size) // 2)
        
        # 再次检查，确保分割不会导致索引错误
        if train_size + val_size > total:
             val_size = total - train_size

        train_data = self.training_samples[:train_size]
        val_data = self.training_samples[train_size:train_size + val_size]
        test_data = self.training_samples[train_size + val_size:]
        
        # 保存为JSONL
        self._save_jsonl(train_data, output_dir / "train.jsonl")
        self._save_jsonl(val_data, output_dir / "val.jsonl")
        self._save_jsonl(test_data, output_dir / "test.jsonl")
        
        # 元数据
        metadata = {
            'total_samples': total,
            'train_samples': len(train_data),
            'val_samples': len(val_data),
            'test_samples': len(test_data),
            'project_name': self.project_name,
            'task_distribution': self._get_task_distribution()
        }
        
        with open(output_dir / "metadata.json", 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Training data saved:")
        print(f"  Train: {len(train_data)}")
        print(f"  Val: {len(val_data)}")
        print(f"  Test: {len(test_data)}")
        print(f"  Total: {total}")
        
        # 显示样本示例
        print(f"\n📝 Sample training example:")
        if train_data:
            sample = random.choice(train_data)
            print(f"Q: {sample.conversations[0]['content'][:100]}...")
            print(f"A: {sample.conversations[1]['content'][:150]}...")
    
    def _save_jsonl(self, data: List[TrainingSample], filepath: Path):
        """保存为JSONL格式"""
        with open(filepath, 'w', encoding='utf-8') as f:
            for sample in data:
                # 仅保存对话，不保存 metadata
                json.dump({'conversations': sample.conversations}, f, ensure_ascii=False)
                f.write('\n')
    
    def _get_task_distribution(self) -> Dict[str, int]:
        """统计任务分布"""
        dist = {}
        for sample in self.training_samples:
            task_type = sample.metadata.get('task_type', 'unknown')
            dist[task_type] = dist.get(task_type, 0) + 1
        return dist


def main():
    print("="*60)
    print("Fixed Training Data Generator (Project-Specific Answers Enhanced)")
    print("="*60)
    
    generator = FixedDataGenerator()
    generator.generate_training_data()
    generator.save_training_data()
    
    print("\n" + "="*60)
    print("✓ Data generation completed!")
    print("="*60)


if __name__ == "__main__":
    main()
