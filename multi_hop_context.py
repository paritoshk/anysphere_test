from typing import List
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def retrieve_multi_hop_context(model, tokenizer, query: str, file_list: List[str]) -> str:
    context = ""
    for file in file_list:
        prompt = f"Context so far:\n{context}\n\nQuery: {query}\n\nRelevant information from {file}:"
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, return_attention_mask=True).to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=100,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        file_context = tokenizer.decode(outputs[0], skip_special_tokens=True)
        context += f"\n\nFrom {file}:\n{file_context[len(prompt):]}"
    
    return context