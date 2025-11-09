# agent/step1_design.py
import os
import sys
import json
from pathlib import Path
import re

try:
    from .nl_llm import LLMClient, LLMError
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent))
    from nl_llm import LLMClient, LLMError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AGENT_DIR = PROJECT_ROOT / "agent"

def load_prompt_from_file(filepath: Path) -> str:
    if not filepath.exists():
        print(f"错误：提示文件未找到于 '{filepath}'。")
        return ""
    return filepath.read_text(encoding="utf-8")

def save_code_to_file(code: str, output_dir: Path, filename: str = "model_main.py") -> Path:
    if "```python" in code:
        code_to_save = code.split("```python", 1)[1].split("```", 1)[0].strip()
    else:
        code_to_save = code.strip()
    
    filepath = output_dir / filename
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(code_to_save)
    print(f"   -> 代码已成功保存到: {filepath}")
    return filepath

def design_new_model(user_request: str, model_name: str):
    print("--- 新模型设计工作流启动 (单文件模式) ---")
    
    output_dir = AGENT_DIR / "generated_models" / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # [EN] TODO: translate comment from Chinese.
    print("\n[阶段一：规划] 正在加载规划器Agent...")
    planner_prompt = load_prompt_from_file(AGENT_DIR / "planner_agent_prompt.md")
    if not planner_prompt: return

    print(f"用户设计请求: '{user_request}'")
    print("调用规划器Agent生成技术蓝图...")
    
    try:
        llm = LLMClient()
        technical_blueprint = llm.chat_generic(system=planner_prompt, user=user_request)
    except LLMError as e:
        print(f"规划器Agent调用失败: {e}")
        return

    print("--- 技术蓝图已生成 ---")
    print(technical_blueprint)
    
    # [EN] TODO: translate comment from Chinese.
    print("\n[阶段二：代码生成] 正在加载代码生成Agent...")
    coder_prompt = load_prompt_from_file(AGENT_DIR / "coder_agent_prompt.md")
    if not coder_prompt: return

    print(f"\n>>> 正在为模型 '{model_name}' 生成完整的训练脚本...")
    
    coder_user_input = f"""
# FULL BLUEPRINT (for context):
{technical_blueprint}

# CURRENT TASK:
Please write the complete, single Python script (`model_main.py`) based on the entire blueprint provided above.
"""
    try:
        llm = LLMClient()
        generated_code = llm.chat_generic(system=coder_prompt, user=coder_user_input)
    except LLMError as e:
        print(f"代码生成Agent调用失败: {e}")
        return
            
    filepath = save_code_to_file(generated_code, output_dir)
    
    # [EN] TODO: translate comment from Chinese.
    init_py_path = output_dir / "__init__.py"
    with open(init_py_path, "w", encoding="utf-8") as f:
        f.write(f"# Entry point for the dynamically generated model: {model_name}\n")
        f.write("from .model_main import run_training\n")
    print(f"   -> 包入口文件已创建: {init_py_path}")

    print("\n--- 新模型设计工作流完成 ---")
    print(f"模型 '{model_name}' 的所有代码已生成在: {output_dir}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python agent/step1_design.py <模型名称> <设计需求>")
        sys.exit(1)
    
    model_name_arg = sys.argv[1]
    user_request_arg = " ".join(sys.argv[2:])
    design_new_model(user_request_arg, model_name_arg)
