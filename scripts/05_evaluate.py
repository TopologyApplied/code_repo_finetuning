"""
基础模型 vs 微调模型对比测试 - 增强版
重点测试项目特定知识
核心改进: 
1. 增强 Repo-Specific 评估权重
2. 提高对 project name, file/class name 提及的奖励
"""
import os
import json
import yaml
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class TestCase:
    """测试用例"""
    type: str  # repo_specific, code_specific, general
    question: str
    category: str  # overview, architecture, implementation, etc.
    reference_files: List[str] = None  # 参考的代码文件


class ModelComparator:
    """模型对比器"""
    
    def __init__(self, config_path: str = "../config/default_config.yaml"):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 加载分析数据
        analysis_path = "./data/repository_analysis.json"
        try:
            with open(analysis_path, 'r', encoding='utf-8') as f:
                self.analysis_data = json.load(f)
        except FileNotFoundError:
            print(f"❌ 警告: 找不到分析文件 {analysis_path}。使用默认空数据。")
            self.analysis_data = {'code_elements': [], 'project_context': {'project_name': 'Laddr'}}

        self.project_name = self.analysis_data.get('project_context', {}).get('project_name', 'Laddr')
        self.code_elements = self.analysis_data.get('code_elements', [])
        
        self.base_model = None
        self.base_tokenizer = None
        self.finetuned_model = None
        self.finetuned_tokenizer = None
        
    def load_models(self):
        """加载基础模型和微调模型"""
        base_model_path = self.config['model']['base_model']
        finetuned_model_path = Path(self.config['training']['output_dir']) / "merged_model"
        
        # 检查 merged model 是否存在
        if not finetuned_model_path.exists():
            print(f"❌ 错误: 找不到合并后的模型 {finetuned_model_path}。请先运行 merge_lora.py。")
            return
        
        print("Loading base model...")
        self.base_tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            trust_remote_code=True
        )
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        self.base_model.eval()
        
        print("Loading fine-tuned model...")
        self.finetuned_tokenizer = AutoTokenizer.from_pretrained(
            str(finetuned_model_path),
            trust_remote_code=True
        )
        self.finetuned_model = AutoModelForCausalLM.from_pretrained(
            str(finetuned_model_path),
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        self.finetuned_model.eval()
        
        print("Models loaded successfully")
    
    def generate_response(self, model, tokenizer, question: str, max_new_tokens: int = 1024) -> str:
        """生成回答"""
        messages = [
            # Qwen 默认无 system prompt
            {"role": "user", "content": question}
        ]
        
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id
            )
        
        # Qwen 模型会生成 <|im_start|>assistant\n 和答案
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], 
                                   skip_special_tokens=True)
        # 移除可能的角色标识
        if response.startswith("assistant"):
            response = response[len("assistant"):].strip()
        
        return response.strip()
    
    def create_test_cases(self) -> List[TestCase]:
        """创建测试用例 - 基于实际代码内容"""
        test_cases = []
        
        # 1. 项目概览问题 (使用项目实际信息)
        test_cases.extend([
            TestCase(
                type="repo_specific",
                question=f"{self.project_name} 项目的主要功能是什么?",
                category="overview"
            ),
            TestCase(
                type="repo_specific",
                question=f"请介绍 {self.project_name} 的架构设计。",
                category="architecture"
            ),
            TestCase(
                type="repo_specific",
                question=f"{self.project_name} 中有哪些核心模块?",
                category="modules"
            ),
        ])
        
        # 2. 具体代码相关问题
        # 找几个关键函数/类
        key_elements = sorted(self.code_elements, 
                             key=lambda x: x.get('complexity', 0), 
                             reverse=True)[:5]
        
        for element in key_elements:
            name = element['name']
            filepath = element['filepath']
            
            test_cases.append(TestCase(
                type="code_specific",
                question=f"{self.project_name} 中 `{name}` 的作用是什么?",
                category="code_understanding",
                reference_files=[filepath]
            ))
            
            if element['type'] in ['function', 'method'] and not name.startswith('_'):
                test_cases.append(TestCase(
                    type="code_specific",
                    question=f"如何使用 {self.project_name} 的 `{name}`?",
                    category="code_usage",
                    reference_files=[filepath]
                ))

            # 增加代码定位问题
            test_cases.append(TestCase(
                 type="code_location",
                 question=f"`{name}` 在 {self.project_name} 哪个文件里?",
                 category="code_location",
                 reference_files=[filepath]
             ))
        
        # 3. 通用软件工程问题 (对照组)
        test_cases.extend([
            TestCase(
                type="general",
                question="什么是代码重构?",
                category="general_knowledge"
            ),
            TestCase(
                type="general",
                question="如何设计一个可扩展的软件架构?",
                category="general_knowledge"
            ),
        ])
        
        return test_cases
    
    def evaluate_response(self, question: str, response: str, test_type: str) -> Dict:
        """评估回答质量 - 增强版"""
        score = 0.0
        
        # 1. 长度合理性 (50-1000字符)
        length = len(response)
        # 避免回答太短（幻觉或拒绝回答）或太长（填充无关信息）
        if 50 <= length <= 1000:
            score += 0.3
        elif 20 <= length < 50 or length > 1000:
            score += 0.1
        
        # 2. 提及项目名称 (repo_specific, code_specific, code_location)
        has_repo_mention = self.project_name.lower() in response.lower()
        if test_type in ["repo_specific", "code_specific", "code_location"]:
            if has_repo_mention:
                score += 0.25 # 基础分数
            
            # 检查是否提及问题中的核心元素 (如函数名、类名)
            question_words = [q.strip('`?!.,') for q in question.split() if q.strip('`?!.,') not in ['Laddr', self.project_name, '请解释', '在', '如何', '中', '的', '作用', '是什么', '哪个', '文件', '里', '使用', '函数', '方法', '类']]
            
            # 检查是否提到任何代码元素
            code_element_mentions = 0
            for elem in self.code_elements[:50]: # 只检查前50个，防止计算量过大
                if f"`{elem['name']}`".lower() in response.lower():
                     code_element_mentions += 1
            
            if code_element_mentions > 0:
                score += 0.25 # 提及代码元素，强烈暗示学习到了新知识
            
            # 检查是否提及文件路径 (code_location or code_specific)
            if test_type in ["code_location", "code_specific"] and any(f for f in self.code_elements[:50] if f['filepath'] in response):
                 score += 0.1 # 提到文件路径，也是强相关性
        
        # 3. 具体性 (vs 模糊回答)
        uncertainty_phrases = ['可能', '大概', '或许', '不确定', '无法确认', '没有找到', '没有广泛知名']
        has_uncertainty = any(phrase in response for phrase in uncertainty_phrases)
        
        # 如果是项目特定/代码特定的问题，避免不确定性
        if test_type in ["repo_specific", "code_specific", "code_location"] and not has_uncertainty:
             score += 0.1 
        # 如果是通用问题，不惩罚不确定性
        
        # 4. 代码块/路径 (code_usage, code_location)
        if test_type in ["code_specific", "code_location"]:
            if '```' in response or '`' in response: # 代码块或反引号标记
                score += 0.1
        
        # 标准化到0-1
        score = min(1.0, score)
        
        # 针对通用问题 (对照组) 设定上限
        if test_type == "general":
             score = min(score, 0.4) # 通用问题得分不应超过 0.4，因为我们主要优化的是特定知识
        
        return {
            'score': score,
            'length': length,
            'has_repo_mention': has_repo_mention,
            'has_uncertainty': has_uncertainty,
            'specifics_count': response.count('`'),
            'quality': 'excellent' if score >= 0.8 else 'good' if score >= 0.6 else 'fair' if score >= 0.3 else 'poor'
        }
    
    def run_comparison(self):
        """运行对比测试"""
        if self.finetuned_model is None:
             print("无法运行对比：模型加载失败。")
             return None

        print(f"\nRunning comparison tests for {self.project_name}...")
        
        test_cases = self.create_test_cases()
        results = []
        
        for i, test in enumerate(test_cases, 1):
            print(f"\n[{i}/{len(test_cases)}] Testing: {test.question}")
            
            # 基础模型回答
            print("  Generating base model response...")
            base_response = self.generate_response(
                self.base_model, 
                self.base_tokenizer, 
                test.question
            )
            
            # 微调模型回答
            print("  Generating fine-tuned model response...")
            finetuned_response = self.generate_response(
                self.finetuned_model,
                self.finetuned_tokenizer,
                test.question
            )
            
            # 评估
            base_eval = self.evaluate_response(test.question, base_response, test.type)
            finetuned_eval = self.evaluate_response(test.question, finetuned_response, test.type)
            
            improvement = finetuned_eval['score'] - base_eval['score']
            
            results.append({
                'test_case': asdict(test),
                'base_response': base_response,
                'base_evaluation': base_eval,
                'finetuned_response': finetuned_response,
                'finetuned_evaluation': finetuned_eval,
                'improvement': improvement
            })
            
            print(f"  Base score: {base_eval['score']:.2f} | Fine-tuned score: {finetuned_eval['score']:.2f} | Improvement: {improvement:+.2f}")
        
        return results
    
    # ... (generate_report 函数不变)
    def generate_report(self, results: List[Dict]):
        """生成对比报告"""
        total_tests = len(results)
        
        base_scores = [r['base_evaluation']['score'] for r in results]
        finetuned_scores = [r['finetuned_evaluation']['score'] for r in results]
        improvements = [r['improvement'] for r in results]
        
        overall_base = sum(base_scores) / len(base_scores)
        overall_finetuned = sum(finetuned_scores) / len(finetuned_scores)
        overall_improvement = sum(improvements) / len(improvements)
        
        # 按类型统计
        type_stats = {}
        for result in results:
            test_type = result['test_case']['type']
            if test_type not in type_stats:
                type_stats[test_type] = {
                    'count': 0,
                    'base_scores': [],
                    'finetuned_scores': []
                }
            type_stats[test_type]['count'] += 1
            type_stats[test_type]['base_scores'].append(result['base_evaluation']['score'])
            type_stats[test_type]['finetuned_scores'].append(result['finetuned_evaluation']['score'])
        
        type_statistics = {}
        for test_type, stats in type_stats.items():
            type_statistics[test_type] = {
                'count': stats['count'],
                'avg_base': sum(stats['base_scores']) / len(stats['base_scores']),
                'avg_finetuned': sum(stats['finetuned_scores']) / len(stats['finetuned_scores']),
                'avg_improvement': (sum(stats['finetuned_scores']) - sum(stats['base_scores'])) / len(stats['base_scores'])
            }
        
        # 判定
        if overall_improvement > 0.15:
            conclusion = "✓ 微调效果显著"
        elif overall_improvement > 0.05:
            conclusion = "⚠️ 微调有提升"
        else:
            conclusion = "✗ 微调效果不明显或失败"
        
        report = {
            'metadata': {
                'repo_name': self.project_name,
                'test_date': datetime.now().isoformat(),
                'base_model': self.config['model']['base_model'],
                'finetuned_model': str(Path(self.config['training']['output_dir']) / "merged_model")
            },
            'summary': {
                'total_tests': total_tests,
                'overall_base_score': overall_base,
                'overall_finetuned_score': overall_finetuned,
                'overall_improvement': overall_improvement,
                'conclusion': conclusion
            },
            'type_statistics': type_statistics,
            'detailed_results': results
        }
        
        # 保存报告
        output_path = f"./comparison_report_{self.project_name}.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'='*60}")
        print("Comparison Report")
        print(f"{'='*60}")
        print(f"Project: {self.project_name}")
        print(f"Total Tests: {total_tests}")
        print(f"Base Model Score: {overall_base:.3f}")
        print(f"Fine-tuned Model Score: {overall_finetuned:.3f}")
        print(f"Improvement: {overall_improvement:+.3f} ({overall_improvement/overall_base*100:+.1f}%)")
        print(f"\nConclusion: {conclusion}")
        print(f"\nDetailed report saved to: {output_path}")
        print(f"{'='*60}")
        
        return report


def main():
    """主函数"""
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
         os.environ['CUDA_VISIBLE_DEVICES'] = '0' # 确保单个 GPU 也能运行
    
    print("="*60)
    print("Base vs Fine-tuned Model Comparison (Enhanced Evaluation)")
    print("="*60)
    
    comparator = ModelComparator()
    comparator.load_models()
    
    results = comparator.run_comparison()
    
    if results is not None:
         report = comparator.generate_report(results)
    
    print("\nComparison completed!")


if __name__ == "__main__":
    main()
