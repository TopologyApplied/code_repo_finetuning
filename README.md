---
language:
- zh
- en
license: apache-2.0
library_name: transformers
tags:
- code
- qwen
- lora
- repository-understanding
- code-assistant
- fine-tuning
- multi-agent-systems
base_model: Qwen/Qwen3-8B
datasets:
- custom
metrics:
- accuracy
- code_understanding
pipeline_tag: text-generation
model-index:
- name: code_repo_finetuning
  results:
  - task:
      type: text-generation
      name: Code Repository Understanding
    metrics:
    - type: accuracy
      value: 71.5
      name: Overall Score
    - type: improvement
      value: 22.1
      name: Improvement over Base Model
---

# Finetune any base model (e.g. Qwen3-8B) on any given code repository

## Model Description

This model is a fine-tuned version of [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) specifically trained to understand and answer questions about any given private or new project repository, for example, [Laddr](https://github.com/AgnetLabs/Laddr) - a framework for building scalable multi-agent systems.

The fine-tuning was performed using **LoRA (Low-Rank Adaptation)** with an innovative training data generation approach that **does not rely on LLM-generated synthetic data**, avoiding circular dependencies and hallucination issues.

### Key Features

- ✅ **Project-Specific Knowledge**: Deep understanding of Laddr's architecture, codebase, and APIs
- ✅ **Code Location**: Accurately locates functions, classes, and modules (+30% improvement)
- ✅ **Code Understanding**: Explains code functionality with detailed context (+19.3% improvement)
- ✅ **Maintains General Abilities**: Retains base model's general knowledge capabilities
- ✅ **Zero Hallucination Training Data**: Generated from real code via AST parsing, not LLM synthesis

## Model Details

### Base Model
- **Model**: Qwen/Qwen3-8B
- **Parameters**: 8 Billion
- **Architecture**: Transformer-based causal language model

### Fine-tuning Specifications
- **Method**: LoRA (Low-Rank Adaptation)
- **LoRA Rank**: 64
- **LoRA Alpha**: 128
- **Target Modules**: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- **Training Framework**: DeepSpeed ZeRO-3
- **Precision**: BF16
- **Epochs**: 3
- **Training Samples**: 650+
- **Training Time**: ~2-3 hours on 2x GPUs (48GB each)

### Training Data

The training dataset was **automatically generated** from the Laddr repository using:
- **Python AST parsing** for code structure extraction
- **Real docstrings** and code comments
- **Function signatures** and parameter information
- **Call graph relationships**
- **Project statistics** and module structure

**Data Composition**:
- Code Explanation: 300+ samples (46%)
- API Usage: 150+ samples (23%)
- Code Location: 100+ samples (15%)
- Project Overview: 50+ samples (8%)
- Design Proposals: 50+ samples (8%)

**Data Split**:
- Training: 80% (520+ samples)
- Validation: 10% (65+ samples)
- Test: 10% (65+ samples)

## Performance

### Overall Results

| Metric | Base Model | Fine-tuned | Improvement |
|--------|-----------|-----------|-------------|
| **Overall Score** | 49.4% | 71.5% | **+22.1%** ✅ |
| Code Location | 60.0% | 90.0% | **+30.0%** ⭐ |
| Code Understanding | 59.3% | 78.6% | +19.3% |
| Project Overview | 35.0% | 51.7% | +16.7% |
| General Knowledge | 10.0% | 30.0% | +20.0% |

### Detailed Performance by Task Type

**Code Location Tasks** (+30.0%):
- Accurately identifies file locations of functions/classes
- Provides complete file paths with line numbers
- Eliminates uncertainty in location queries

**Code Understanding Tasks** (+19.3%):
- Explains code functionality with context
- Includes function signatures and parameters
- Extracts and presents real docstrings

**Project Overview Tasks** (+16.7%):
- Describes project purpose and architecture
- Lists technology stack and dependencies
- Provides project statistics and structure

**General Knowledge** (+20.0%):
- Maintains ability to answer general programming questions
- Improves conciseness (from verbose to clear)
- Preserves base model capabilities

## Use Cases

### 1. Code Assistant for Laddr Project
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "tensense/code_repo_finetuning"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

question = "Where is the create_system_tools function located in Laddr?"
messages = [{"role": "user", "content": question}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer([text], return_tensors="pt")

outputs = model.generate(**inputs, max_new_tokens=512)
response = tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
print(response)
```

### 2. API Documentation Assistant
Query about function usage, parameters, and return values.

### 3. Codebase Onboarding
Help new developers quickly understand the Laddr project structure and key components.

### 4. Code Review Support
Provide context and explanations for code review processes.

## Example Queries

### Query 1: Code Location
**Question**: "Where is `create_system_tools` located in Laddr?"

**Response**:
> `create_system_tools` is located in `lib/laddr/src/laddr/core/system_tools.py`.
>
> **Function Description**:
> Create system tools for task delegation and artifact management. This function checks for user-provided overrides before registering the default implementations.
>
> **Function Signature**:
> ```python
> def create_system_tools(message_bus, storage_backend=None, agent=None) -> dict[str, tuple[Any, list[str]]]:
> ```
>
> **Parameters**:
> - `message_bus` (Any): Message queue backend
> - `storage_backend` (Any): Optional storage backend
> - `agent` (Any): Optional agent instance
>
> **Returns**: `dict[str, tuple[Any, list[str]]]`

### Query 2: Project Overview
**Question**: "What is Laddr's main functionality?"

**Response**:
> Laddr is a Python project that serves as a framework for building scalable multi-agent systems.
>
> **Main Technologies**: bandit, black, boto3-stubs, build, click, etc.
>
> The project contains 279 code elements, consisting of 67 classes and 212 functions/methods.
>
> **Core Modules**:
> - `core` (279 elements)
> - `cli` (52 elements)
> - `llms` (39 elements)

## Limitations

- **Project-Specific**: Optimized for Laddr project; may not perform as well on other codebases
- **Knowledge Cutoff**: Based on the Laddr repository as of training time (2025-01)
- **Language Focus**: Primarily trained on Python code and English/Chinese documentation
- **Limited General Coding**: While it maintains general knowledge, it's optimized for Laddr-specific queries

## Training Methodology

### Innovation: LLM-Free Training Data Generation

Unlike traditional approaches that use LLMs to generate synthetic training data, this project employs a novel methodology:

1. **AST-Based Code Parsing**: Python Abstract Syntax Tree analysis extracts accurate code structure
2. **Real Documentation**: Utilizes actual docstrings, comments, and code signatures
3. **Call Graph Analysis**: Builds function dependency relationships
4. **Pattern Extraction**: Identifies code patterns (implementation, usage, interaction)
5. **Template-Based QA**: Generates question-answer pairs using templates with real code context

**Benefits**:
- ✅ Avoids circular dependency (using LLM data to train LLM)
- ✅ Eliminates hallucination in training data
- ✅ Ensures factual accuracy
- ✅ Provides complete reasoning traces

### Training Pipeline

```
GitHub Repository
    ↓
[1. Repository Analyzer]
    → Extracts code elements, patterns, call graph
    ↓
[2. Data Generator]
    → Creates QA pairs with code context
    ↓
[3. Model Fine-tuner]
    → LoRA + DeepSpeed ZeRO-3 training
    ↓
[4. LoRA Merger]
    → Merges adapter into base model
    ↓
[5. Model Evaluator]
    → Compares base vs fine-tuned
    ↓
Fine-tuned Model
```

## Extensibility

The training methodology is **repository-agnostic** and can be applied to any codebase:

### Adapt to Your Repository

```bash
# 1. Update configuration
python utils/config_manager.py https://github.com/your-org/your-repo

# 2. Analyze repository
python scripts/01_analyze_repo.py

# 3. Generate training data
python scripts/02_generate_data.py

# 4. Fine-tune model
deepspeed --num_gpus=2 scripts/03_train_model.py

# 5. Merge LoRA weights
python scripts/04_merge_weights.py

# 6. Evaluate
python scripts/05_evaluate.py
```

**Supported Languages** (currently):
- Python (primary)
- Markdown (documentation)

**Extensible to**:
- JavaScript/TypeScript
- Java
- Go
- Rust

## Ethical Considerations

- **Code Attribution**: All training data comes from the open-source Laddr repository
- **License Compliance**: Respects Apache 2.0 license of both base model and Laddr project
- **No Private Data**: Only uses publicly available code
- **Reproducibility**: Complete methodology documented for transparency

## Citation

If you use this model or methodology in your research, please cite:

```bibtex
@misc{qwen3-code-repo-finetuned-2025,
  title={Finetune any base model (e.g. Qwen3-8B) on any given code repository},
  author={Tensense},
  year={2025},
  publisher={HuggingFace},
  url={https://huggingface.co/tensense/code_repo_finetuning}
}
```

## Acknowledgments

- **Base Model**: [Qwen Team](https://huggingface.co/Qwen) for Qwen3-8B
- **Laddr Project**: [AgnetLabs](https://github.com/AgnetLabs/Laddr) for the multi-agent framework
- **Training Framework**: HuggingFace Transformers, DeepSpeed, PEFT (LoRA)

## License

This model is released under the **Apache 2.0 License**, consistent with:
- Qwen3-8B base model license
- Laddr project license

## Model Card Authors

[Tensense]

## Model Card Contact

For questions or issues, please contact:
- Email: xu@tensense.org
- GitHub: [TopologyApplied](https://github.com/TopologyApplied)
- HuggingFace: [tensense](https://huggingface.co/tensense)

---

## Additional Resources

- **Base Model**: [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)
- **Training Code**: [GitHub Repository](https://github.com/TopologyApplied/code_repo_finetuning)
- **Checkpoint & Finetuned Model**: [Huggingface](https://huggingface.co/tensense/code_repo_finetuning)
- **Laddr Project**: [GitHub](https://github.com/AgnetLabs/Laddr)
- **Evaluation Report**: [[Link to comparison_report.json](https://github.com/TopologyApplied/code_repo_finetuning/blob/main/output/comparison_report_Laddr.json)]
- **Design Documentation**: [[Link to design docs](https://github.com/TopologyApplied/code_repo_finetuning/blob/main/代码仓库智能训练数据生成系统_设计文档.md)]

## Version History

### v1.0 (2025-11-15)
- Initial release
- Fine-tuned on Laddr repository
- 650+ training samples
- LoRA rank 64, alpha 128
- 3 epochs training
- Overall improvement: +22.1%

---

**Note**: This is a demonstration of repository-specific fine-tuning methodology. The approach can be adapted to any codebase for creating custom code assistants.
