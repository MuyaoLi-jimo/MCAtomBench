'''
Author: Muyao 2350076251@qq.com
Date: 2025-03-05 10:56:23
LastEditors: Muyao 2350076251@qq.com
LastEditTime: 2025-03-30 21:22:45
'''


def load_visual_model(checkpoint_path ="",**kwargs):
    if not checkpoint_path:
        raise AssertionError("checkpoint_path is required")
    
    checkpoint_path = checkpoint_path.lower().replace('-','_')
    
    if "mistral" in checkpoint_path:
        LLM_backbone = "mistral"
    elif "vicuna" in checkpoint_path:
        LLM_backbone = "llama-2"
    elif "llama_3" in checkpoint_path or "llama3" in  checkpoint_path:
        LLM_backbone = "llama-3"
    elif "qwen2_vl" in checkpoint_path:
        LLM_backbone = "qwen2_vl"
        
    if 'llava_next' in checkpoint_path or 'llava_v1.6'  in checkpoint_path:
        VLM_backbone = "llava-next"
    elif "qwen2_vl" in checkpoint_path:
        VLM_backbone = "qwen2_vl"
    else:
        raise AssertionError

    return LLM_backbone,VLM_backbone
        