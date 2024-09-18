import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def detect_bugs(model, tokenizer, code: str) -> str:
    prompt = f"Analyze the following code for potential bugs and suggest improvements:\n\n{code}\n\nBugs and Suggestions:"
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, return_attention_mask=True).to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=200,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    analysis = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return analysis[len(prompt):]