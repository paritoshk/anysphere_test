import asyncio
import torch

async def perfect_edit_async(model, tokenizer, code: str, instruction: str) -> str:
    prompt = f"Code:\n{code}\n\nInstruction: {instruction}\n\nEdited Code:"
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, return_attention_mask=True).to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=500,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
    
    edited_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return edited_code[len(prompt):]

def perfect_edit(model, tokenizer, code: str, instruction: str) -> str:
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(perfect_edit_async(model, tokenizer, code, instruction))