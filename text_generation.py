import torch
from transformers import Cache
from typing import List, Tuple, Dict

def vanilla_edit(model, tokenizer, prompt: str, max_tokens: int) -> str:
    """
    Perform vanilla (non-speculative) text generation.

    Args:
        prompt (str): The input prompt for text generation.
        max_tokens (int): Maximum number of tokens to generate.

    Returns:
        str: The generated text.
    """
    
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, return_attention_mask=True).to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text[len(prompt):]

def speculative_edit(model, tokenizer, prompt: str, max_tokens: int, num_heads: int = 3, codebase_context: Dict[str, str] = None) -> str:
    """
    Perform speculative text generation using multiple heads.

    Args:
        prompt (str): The input prompt for text generation.
        max_tokens (int): Maximum number of tokens to generate.
        num_heads (int): Number of speculative heads to use.

    Returns:
        str: The generated text.
    """
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, return_attention_mask=True).to("cuda")
    input_length = inputs.input_ids.shape[1]
    
    generated_tokens: List[int] = []
    total_tokens = 0
    
    if codebase_context:
        context_prompt = "\n".join([f"File: {file}\n{content}" for file, content in codebase_context.items()])
        context_inputs = tokenizer(context_prompt, return_tensors="pt", padding=True, return_attention_mask=True).to("cuda")
        inputs.input_ids = torch.cat([context_inputs.input_ids, inputs.input_ids], dim=1)
        inputs.attention_mask = torch.cat([context_inputs.attention_mask, inputs.attention_mask], dim=1)
    
    cache = Cache()
    
    while total_tokens < max_tokens:
        with torch.no_grad():
            outputs = model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, past_key_values=cache.to_legacy_tuple())
            logits = outputs.logits[:, -num_heads:]
        
        predicted_tokens = torch.argmax(logits, dim=-1)
        mismatch_index = (predicted_tokens[0] != inputs.input_ids[0, -num_heads:]).nonzero(as_tuple=True)[0]
        
        if len(mismatch_index) == 0:
            generated_tokens.extend(predicted_tokens[0].tolist())
            mismatch_index = num_heads
        else:
            mismatch_index = mismatch_index[0].item()
            generated_tokens.extend(predicted_tokens[0, :mismatch_index].tolist())
        
        new_tokens = tokenizer.decode(predicted_tokens[0, :mismatch_index])
        inputs = tokenizer(tokenizer.decode(inputs.input_ids[0]) + new_tokens, return_tensors="pt", padding=True, return_attention_mask=True).to("cuda")
        
        total_tokens = len(generated_tokens)
        cache.update(outputs.past_key_values)
        
        if total_tokens >= max_tokens:
            break
    
    return tokenizer.decode(generated_tokens[:max_tokens])

def compare_methods(model, tokenizer, prompt: str, max_tokens: int, writer) -> Tuple[str, str, float, float]:
    import time

    start_time = time.time()
    vanilla_output = vanilla_edit(model, tokenizer, prompt, max_tokens)
    vanilla_time = time.time() - start_time

    start_time = time.time()
    speculative_output = speculative_edit(model, tokenizer, prompt, max_tokens)
    speculative_time = time.time() - start_time

    writer.add_scalar('Time_Difference', vanilla_time - speculative_time, max_tokens)

    return vanilla_output, speculative_output, vanilla_time, speculative_time