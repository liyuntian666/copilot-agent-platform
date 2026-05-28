from prompt import QwenModelPromptHub, PromptModelHub
from prompt.dsv32_model_prompts import DSv32ModelPromptHub
from utils import logger


def create_prompt_hub(model_name: str):
    """根据模型名称创建对应的提示模型Hub"""
    logger.info(f"准备使用模型： {model_name}")
    if "qwen" in model_name.lower():
        return QwenModelPromptHub("",model_name)
    elif "deepseek-v3.2" in model_name.lower():
        return DSv32ModelPromptHub("", model_name)
    else:
        return PromptModelHub("")