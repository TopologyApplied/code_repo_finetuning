"""
增强代码仓库分析器 - 提取实际代码内容用于深度学习
生成包含真实代码上下文的训练数据
"""
import os
import ast
import json
import re
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple, Set
from dataclasses import dataclass, asdict
import yaml
# 尝试导入 git
try:
    from git import Repo
    GIT_AVAILABLE = True
except ImportError:
    print("Warning: python-git not installed. Cloning will be skipped.")
    GIT_AVAILABLE = False

from collections import defaultdict
import hashlib


@dataclass
class CodeElement:
    """代码元素数据结构 - 增强版"""
    type: str  
    name: str
    filepath: str
    start_line: int
    end_line: int
    code: str
    docstring: str
    dependencies: List[str]
    complexity: int
    business_context: str
    # 新增字段
    imports: List[str]  # 导入的模块
    called_functions: List[str]  # 调用的函数
    parent_class: str  # 所属类（针对方法）
    decorators: List[str]  # 装饰器
    parameters: List[Dict[str, str]]  # 参数列表
    return_type: str  # 返回类型


@dataclass
class CodePattern:
    """代码模式 - 用于生成高质量训练样本"""
    pattern_type: str  # implementation, usage, interaction
    description: str
    code_snippet: str
    context: str
    related_elements: List[str]


@dataclass
class ProjectContext:
    """项目上下文信息"""
    project_name: str
    description: str
    main_technologies: List[str]
    architecture_style: str
    key_modules: List[str]
    dependencies: Dict[str, str]


class RepositoryAnalyzer:
    """增强代码仓库分析器"""
    
    def __init__(self, config_path: str = "../config/default_config.yaml"):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.repo_path = Path(self.config['repository']['local_path'])
        self.languages = self.config['repository']['languages']
        self.exclude_dirs = set(self.config['repository']['exclude_dirs'])
        
        self.code_elements = []
        self.code_patterns = []
        self.project_structure = {}
        self.project_context = None
        self.file_imports = defaultdict(set)  # 文件级导入
        self.function_calls_graph = defaultdict(set)  # 函数调用图
        
    def clone_repository(self):
        """克隆或更新代码仓库"""
        if not GIT_AVAILABLE:
            print("❌ 错误: 缺少 python-git 库。请手动克隆仓库或安装 'pip install gitpython'。")
            return

        repo_url = self.config['repository']['url']
        
        if self.repo_path.exists():
            print(f"Repository already exists at {self.repo_path}")
            try:
                repo = Repo(self.repo_path)
                if not repo.git_dir:
                    print(f"Warning: {self.repo_path} exists but is not a Git repository. Skipping pull.")
                    return
                
                print("Attempting to pull latest changes...")
                repo.remotes.origin.pull()
            except Exception as e:
                print(f"Warning: Could not pull updates for {self.repo_path}: {e}")
        else:
            print(f"Cloning repository from {repo_url} to {self.repo_path}")
            self.repo_path.parent.mkdir(parents=True, exist_ok=True)
            Repo.clone_from(repo_url, self.repo_path)
        
    def analyze_repository(self):
        """分析整个代码仓库 - 增强版"""
        if not self.repo_path.exists():
            print(f"❌ 错误: 仓库路径 {self.repo_path} 不存在。请先克隆仓库。")
            return

        print("Analyzing repository structure...")
        self._build_project_structure()
        
        print("Extracting project context...")
        self._extract_project_context()
        
        print("Extracting code elements with full context...")
        self._extract_code_elements()
        
        print("Building function call graph...")
        self._build_call_graph()
        
        print("Extracting code patterns...")
        self._extract_code_patterns()
        
        print(f"Found {len(self.code_elements)} code elements")
        print(f"Extracted {len(self.code_patterns)} code patterns")
        
    def _extract_project_context(self):
        """提取项目上下文信息"""
        project_name = self.repo_path.name
        
        # 读取README
        readme_content = ""
        for readme_name in ['README.md', 'README.rst', 'README.txt', 'README']:
            readme_path = self.repo_path / readme_name
            if readme_path.exists():
                try:
                    with open(readme_path, 'r', encoding='utf-8') as f:
                        readme_content = f.read()[:2000]  # 前2000字符
                    break
                except:
                    pass
        
        # 提取主要技术和完整依赖
        technologies = set()
        dependencies_dict = {}
        
        # 辅助函数：解析依赖行
        def parse_dependency_line(line):
             line = line.strip()
             if not line or line.startswith('#'):
                 return
             
             match = re.match(r'([a-zA-Z0-9_-]+)([=<>!~]+[0-9.a-zA-Z*]+)?', line)
             if match:
                 pkg = match.group(1).strip()
                 version = match.group(2) if match.group(2) else "any"
                 dependencies_dict[pkg] = version
                 technologies.add(pkg)

        # 从requirements.txt提取
        req_file = self.repo_path / 'requirements.txt'
        if req_file.exists():
            try:
                with open(req_file, 'r') as f:
                    for line in f:
                        parse_dependency_line(line)
            except Exception as e:
                print(f"Warning: Could not read requirements.txt: {e}")
        
        # 从setup.py提取
        setup_file = self.repo_path / 'setup.py'
        if setup_file.exists():
            try:
                with open(setup_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # 提取 install_requires
                    import_match = re.findall(r'install_requires\s*=\s*\[(.*?)\]', content, re.DOTALL)
                    if import_match:
                        for pkg_line in re.findall(r'["\']([^"\']+)["\']', import_match[0]):
                            parse_dependency_line(pkg_line)
            except Exception as e:
                print(f"Warning: Could not parse setup.py: {e}")
        
        # 架构风格推断
        architecture_style = "modular"
        if (self.repo_path / 'microservices').exists() or (self.repo_path / 'services').exists():
            architecture_style = "microservices"
        elif any((self.repo_path / item).is_dir() and item not in self.exclude_dirs for item in ['app', 'src', 'core']):
            architecture_style = "layered"
        
        # 关键模块
        key_modules = []
        for item in self.repo_path.iterdir():
            if item.is_dir() and item.name not in self.exclude_dirs:
                if (item / '__init__.py').exists() or (item / 'main.py').exists():
                    key_modules.append(item.name)
        
        self.project_context = ProjectContext(
            project_name=project_name,
            description=readme_content[:500] if readme_content else f"{project_name} Python project",
            main_technologies=sorted(list(technologies))[:10],
            architecture_style=architecture_style,
            key_modules=key_modules[:10],
            dependencies=dependencies_dict
        )
        
    def _build_project_structure(self):
        """构建项目结构树"""
        for root, dirs, files in os.walk(self.repo_path):
            dirs[:] = [d for d in dirs if d not in self.exclude_dirs]
            
            try:
                rel_root = Path(root).relative_to(self.repo_path)
            except ValueError:
                continue

            current = self.project_structure
            parts = rel_root.parts

            temp_current = self.project_structure
            for part in parts:
                if part not in temp_current:
                    temp_current[part] = {}
                temp_current = temp_current[part]
            current = temp_current

            for file in files:
                if file.endswith('.py') or file.endswith('.md') or file.endswith('.json') or file.endswith('.yml') or file.endswith('.yaml'):
                    current[file] = str((rel_root / file))

    
    def _extract_code_elements(self):
        """提取代码元素 - 增强版"""
        python_files = list(self.repo_path.rglob("*.py"))
        
        for filepath in python_files:
            if any(excluded in filepath.parts for excluded in self.exclude_dirs):
                continue
            
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    source_code = f.read()
                
                # 提取导入
                imports = self._extract_imports(source_code)
                self.file_imports[str(filepath)] = imports
                
                # 解析代码元素
                elements = self._parse_python_file(filepath, source_code)
                self.code_elements.extend(elements)
                
            except Exception as e:
                print(f"Error parsing {filepath}: {e}")
    
    def _extract_imports(self, source_code: str) -> Set[str]:
        """提取导入模块"""
        imports = set()
        try:
            tree = ast.parse(source_code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.add(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.add(node.module)
        except:
            pass
        return imports
    
    # 核心修复点: 解决 'argument of type ... is not iterable' 错误
    def _unparse_node(self, node: ast.AST) -> str:
        """安全地 unparse AST 节点，兼容性修复 'argument of type ... is not iterable'"""
        if sys.version_info >= (3, 9):
            try:
                # 尝试使用官方 unparse
                return ast.unparse(node)
            except Exception:
                pass
        
        # 自行实现简易 unparse，只提取关键名称，避免递归错误
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._unparse_node(node.value)}.{node.attr}"
        elif isinstance(node, ast.Subscript):
            return f"{self._unparse_node(node.value)}[{self._unparse_node(node.slice.value if hasattr(node.slice, 'value') else node.slice)}]" # 兼容不同版本 Slice
        # 兼容不同版本的 Constant/Num/Str
        elif hasattr(ast, 'Constant') and isinstance(node, ast.Constant):
             return str(node.value)
        elif isinstance(node, (ast.Str, ast.Num, ast.Bytes)):
             return str(node.s if isinstance(node, ast.Str) else node.n)
        elif isinstance(node, (ast.List, ast.Tuple)):
            return f"[{', '.join(self._unparse_node(e) for e in node.elts)}]"
        elif isinstance(node, ast.Call):
            return f"{self._unparse_node(node.func)}(...)"
        
        return str(node) # 最终fallback


    def _parse_python_file(self, filepath: Path, source_code: str) -> List[CodeElement]:
        """解析Python文件 - 增强版"""
        elements = []
        
        try:
            tree = ast.parse(source_code)
            
            # 使用列表来跟踪 FunctionDef
            all_function_defs = {}
            class_methods = set()

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    element = self._extract_class_enhanced(node, filepath, source_code)
                    if element:
                        elements.append(element)
                    
                    # 提取类方法
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            method = self._extract_method_enhanced(item, node, filepath, source_code)
                            if method:
                                elements.append(method)
                                class_methods.add(item.name) # 标记为方法

                elif isinstance(node, ast.FunctionDef):
                    all_function_defs[node.name] = (node, elements)
            
            # 提取非方法函数
            for func_name, (node, _) in all_function_defs.items():
                if func_name not in class_methods:
                    element = self._extract_function_enhanced(node, filepath, source_code)
                    if element:
                        elements.append(element)
        
        except SyntaxError as e:
            print(f"Syntax error in {filepath}: {e}")
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
        
        return elements
    
    def _extract_function_enhanced(self, node: ast.FunctionDef, filepath: Path, 
                                   source_code: str) -> CodeElement:
        """提取函数信息 - 增强版"""
        lines = source_code.split('\n')
        start_line = node.lineno
        end_line = node.end_lineno if hasattr(node, 'end_lineno') and node.end_lineno else start_line + 1
        
        code = '\n'.join(lines[start_line-1:end_line])
        docstring = ast.get_docstring(node) or ""
        
        # 提取装饰器
        decorators = [self._get_decorator_name(dec) for dec in node.decorator_list]
        
        # 提取参数
        parameters = []
        for arg in node.args.args:
            param_info = {'name': arg.arg}
            param_info['type'] = self._unparse_node(arg.annotation) if arg.annotation else 'Any'
            parameters.append(param_info)
        
        # 提取返回类型
        return_type = None
        if node.returns:
            return_type = self._unparse_node(node.returns)
        else:
             return_type = 'None' # 默认 None
        
        # 提取依赖和调用
        dependencies = []
        called_functions = []
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    called_functions.append(child.func.id)
                    dependencies.append(child.func.id)
                elif isinstance(child.func, ast.Attribute):
                    called_functions.append(child.func.attr)
                    dependencies.append(child.func.attr)
        
        # 提取导入使用
        imports_used = []
        file_imports = self.file_imports.get(str(filepath), set())
        for imp in file_imports:
            if imp in code:
                imports_used.append(imp)
            for alias_name, original_name in self._get_import_aliases(lines).items():
                if original_name == imp and alias_name in code:
                     imports_used.append(imp)

        
        complexity = self._calculate_complexity(node)
        business_context = self._extract_business_context(docstring, code)
        
        return CodeElement(
            type="function",
            name=node.name,
            filepath=str(filepath.relative_to(self.repo_path)),
            start_line=start_line,
            end_line=end_line,
            code=code,
            docstring=docstring,
            dependencies=list(set(dependencies)),
            complexity=complexity,
            business_context=business_context,
            imports=list(set(imports_used)),
            called_functions=list(set(called_functions)),
            parent_class="",
            decorators=decorators,
            parameters=parameters,
            return_type=return_type
        )
    
    def _get_import_aliases(self, lines: List[str]) -> Dict[str, str]:
        """获取导入别名 (alias: original_name)"""
        aliases = {}
        for line in lines:
            match_import = re.match(r'^\s*import\s+([\w.]+)\s+as\s+(\w+)', line)
            if match_import:
                aliases[match_import.group(2)] = match_import.group(1)
            # 也可以解析 from ... import name as alias
            match_from = re.match(r'^\s*from\s+[\w.]+\s+import\s+([\w]+)\s+as\s+(\w+)', line)
            if match_from:
                 aliases[match_from.group(2)] = match_from.group(1)
        return aliases
        
    def _extract_class_enhanced(self, node: ast.ClassDef, filepath: Path, 
                               source_code: str) -> CodeElement:
        """提取类信息 - 增强版"""
        lines = source_code.split('\n')
        start_line = node.lineno
        end_line = node.end_lineno if hasattr(node, 'end_lineno') and node.end_lineno else start_line + 1
        
        # 提取类定义（不含方法体）
        class_def_end = start_line
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                class_def_end = item.lineno - 1
                break
            class_def_end = item.end_lineno if hasattr(item, 'end_lineno') and item.end_lineno else class_def_end
        
        class_def_end = min(class_def_end, len(lines))
        
        code = '\n'.join(lines[start_line-1:min(class_def_end, start_line + 20)])
        docstring = ast.get_docstring(node) or ""
        
        # 提取基类
        dependencies = []
        for base in node.bases:
            dependencies.append(self._unparse_node(base)) # 使用修复后的 _unparse_node
        
        # 提取装饰器
        decorators = [self._get_decorator_name(dec) for dec in node.decorator_list]
        
        complexity = len(node.body)
        business_context = self._extract_business_context(docstring, code)
        
        return CodeElement(
            type="class",
            name=node.name,
            filepath=str(filepath.relative_to(self.repo_path)),
            start_line=start_line,
            end_line=end_line,
            code=code,
            docstring=docstring,
            dependencies=dependencies,
            complexity=complexity,
            business_context=business_context,
            imports=[],
            called_functions=[],
            parent_class="",
            decorators=decorators,
            parameters=[],
            return_type=""
        )
    
    def _extract_method_enhanced(self, node: ast.FunctionDef, class_node: ast.ClassDef,
                                filepath: Path, source_code: str) -> CodeElement:
        """提取类方法信息 - 增强版"""
        element = self._extract_function_enhanced(node, filepath, source_code)
        if element:
            element.type = "method"
            element.name = f"{class_node.name}.{node.name}"
            element.parent_class = class_node.name
        return element
    
    def _get_decorator_name(self, decorator) -> str:
        """获取装饰器名称"""
        if isinstance(decorator, ast.Name):
            return decorator.id
        elif isinstance(decorator, ast.Call):
            if isinstance(decorator.func, ast.Name):
                return decorator.func.id
            elif isinstance(decorator.func, ast.Attribute):
                return decorator.func.attr
        return ""
    
    def _calculate_complexity(self, node: ast.AST) -> int:
        """计算代码复杂度"""
        complexity = 1
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler,
                                ast.With, ast.Assert, ast.BoolOp)):
                complexity += 1
        
        return complexity
    
    def _extract_business_context(self, docstring: str, code: str) -> str:
        """提取业务上下文关键词"""
        keywords = []
        
        text = (docstring + " " + code).lower()
        business_terms = re.findall(
            r'\b(validate|process|handle|create|update|delete|'
            r'authenticate|authorize|calculate|generate|parse|'
            r'transform|filter|query|save|load|fetch|send|'
            r'init|initialize|setup|config|configure|agent|tool|database|storage|cache|message|queue|runtime|llm)\b', 
            text
        )
        keywords.extend(business_terms)
        
        return ", ".join(list(set(keywords))[:5])
    
    def _build_call_graph(self):
        """构建函数调用图"""
        for element in self.code_elements:
            if element.type in ['function', 'method']:
                for called in element.called_functions:
                    self.function_calls_graph[element.name].add(called)
    
    def _extract_code_patterns(self):
        """提取代码模式用于训练"""
        
        # 模式1: 类的完整实现示例
        for element in self.code_elements:
            if element.type == 'class' and element.docstring:
                pattern = CodePattern(
                    pattern_type="class_implementation",
                    description=f"类 {element.name} 的实现",
                    code_snippet=element.code,
                    context=f"文件: {element.filepath}\n文档: {element.docstring[:200]}",
                    related_elements=[element.name]
                )
                self.code_patterns.append(pattern)
        
        # 模式2: 函数实现和用法
        for element in self.code_elements:
            if element.type in ['function', 'method'] and len(element.code) > 50 and element.docstring:
                # 查找调用此函数的地方
                callers = [e.name for e in self.code_elements 
                          if element.name in e.called_functions or element.name.split('.')[-1] in e.called_functions]
                
                pattern = CodePattern(
                    pattern_type="function_implementation",
                    description=f"{element.type} {element.name} 的实现和用法",
                    code_snippet=element.code,
                    context=f"文件: {element.filepath}\n"
                            f"参数: {', '.join(p['name'] for p in element.parameters)}\n"
                            f"调用者: {', '.join(callers[:3]) if callers else '无'}",
                    related_elements=[element.name] + callers[:3]
                )
                self.code_patterns.append(pattern)
        
        # 模式3: 模块间交互
        module_interactions = defaultdict(list)
        for element in self.code_elements:
            if element.imports:
                for imp in element.imports:
                    module_interactions[imp].append(element)
        
        for module, elements in module_interactions.items():
            if len(elements) >= 2 and module not in ['typing', 'os', 'sys', 're', 'json', 'yaml']:
                pattern = CodePattern(
                    pattern_type="module_interaction",
                    description=f"核心模块 {module} 的使用方式",
                    code_snippet="\n\n".join([e.code[:200] for e in elements[:3]]),
                    context=f"被 {len(elements)} 个组件使用",
                    related_elements=[e.name for e in elements[:5]]
                )
                self.code_patterns.append(pattern)
    
    def get_element_with_context(self, element_name: str) -> Dict[str, Any]:
        """获取代码元素及其完整上下文"""
        element = None
        for e in self.code_elements:
            if e.name == element_name or element_name in e.name:
                element = e
                break
        
        if not element:
            return None
        
        # 获取相关元素
        related = []
        for dep in element.dependencies:
            for e in self.code_elements:
                if dep in e.name:
                    related.append(e)
                    break
        
        # 获取调用者
        callers = [e for e in self.code_elements if element.name in e.called_functions or element.name.split('.')[-1] in e.called_functions]
        
        return {
            'element': asdict(element),
            'related_elements': [asdict(r) for r in related[:5]],
            'callers': [asdict(c) for c in callers[:5]],
            'file_context': {
                'imports': list(self.file_imports.get(element.filepath, set()))
            }
        }
    
    def save_analysis(self, output_path: str = "../data/repository_analysis.json"):
        """保存分析结果 - 增强版"""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # 统计文件类型
        file_type_counts = defaultdict(int)
        for root, dirs, files in os.walk(self.repo_path):
            dirs[:] = [d for d in dirs if d not in self.exclude_dirs]
            for file in files:
                ext = Path(file).suffix
                if ext:
                    file_type_counts[ext] += 1
                else:
                    file_type_counts['.other'] += 1


        analysis_data = {
            'project_context': asdict(self.project_context) if self.project_context else {},
            'project_structure': self.project_structure,
            'code_elements': [asdict(e) for e in self.code_elements],
            'code_patterns': [asdict(p) for p in self.code_patterns],
            'statistics': {
                'total_elements': len(self.code_elements),
                'functions': len([e for e in self.code_elements if e.type == 'function']),
                'classes': len([e for e in self.code_elements if e.type == 'class']),
                'methods': len([e for e in self.code_elements if e.type == 'method']),
                'code_patterns': len(self.code_patterns),
                'file_type_counts': dict(file_type_counts),
            },
            'call_graph': {k: list(v) for k, v in self.function_calls_graph.items()}
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(analysis_data, f, indent=2, ensure_ascii=False)
        
        print(f"Enhanced analysis saved to {output_path}")
        return analysis_data


if __name__ == "__main__":
    analyzer = RepositoryAnalyzer()
    analyzer.clone_repository() 
    analyzer.analyze_repository()
    analyzer.save_analysis()
