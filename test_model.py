import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def test_model():
    model_path = "/workspace/llama3finetune/model"
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Test prompt
    prompt = "Explain the concept of quantum computing in simple terms."
    
    # Tokenize and generate
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_length=200, num_return_sequences=1)
    
    # Decode and print the result
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Prompt: {prompt}\n")
    print(f"Response: {result}")

if __name__ == "__main__":
    test_model()