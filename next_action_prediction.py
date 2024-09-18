def predict_next_action(model, tokenizer, current_state: str) -> str:
    prompt = f"Given the current editor state:\n{current_state}\n\nPredict the next user action:"
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, return_attention_mask=True).to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    predicted_action = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return predicted_action[len(prompt):]