import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Tuple, Dict
from torch.utils.tensorboard import SummaryWriter

# Initialize TensorBoard writer
writer = SummaryWriter()

# Load the model and tokenizer
model_path = "/workspace/llama3finetune/model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).cuda()

# Set the model to evaluation mode
model.eval()

def vanilla_edit(prompt: str, max_tokens: int) -> str:
    """
    Perform vanilla (non-speculative) text generation.

    Args:
        prompt (str): The input prompt for text generation.
        max_tokens (int): Maximum number of tokens to generate.

    Returns:
        str: The generated text.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=max_tokens,
            do_sample=False,  # Use greedy decoding
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text[len(prompt):]  # Return only the newly generated text

def speculative_edit(prompt: str, max_tokens: int, num_heads: int = 3, codebase_context: Dict[str, str] = None) -> str:
    """
    Perform speculative text generation using multiple heads and codebase context.

    Args:
        prompt (str): The input prompt for text generation.
        max_tokens (int): Maximum number of tokens to generate.
        num_heads (int): Number of speculative heads to use.
        codebase_context (Dict[str, str]): Additional context from the codebase.

    Returns:
        str: The generated text.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    input_length = inputs.input_ids.shape[1]
    
    generated_tokens: List[int] = []
    total_tokens = 0
    
    # Incorporate codebase context if available
    if codebase_context:
        context_prompt = "\n".join([f"File: {file}\n{content}" for file, content in codebase_context.items()])
        context_inputs = tokenizer(context_prompt, return_tensors="pt").to("cuda")
        inputs = torch.cat([context_inputs.input_ids, inputs.input_ids], dim=1)
    
    while total_tokens < max_tokens:
        # Speculate multiple tokens using the num_heads
        with torch.no_grad():
            outputs = model(input_ids=inputs.input_ids)
            logits = outputs.logits[:, -num_heads:]
        
        # Get the predicted tokens for each head
        predicted_tokens = torch.argmax(logits, dim=-1)
        
        # Find the first mismatch or use all predicted tokens if no mismatch
        mismatch_index = (predicted_tokens[0] != inputs.input_ids[0, -num_heads:]).nonzero(as_tuple=True)[0]
        
        if len(mismatch_index) == 0:
            # All speculated tokens are correct
            generated_tokens.extend(predicted_tokens[0].tolist())
            mismatch_index = num_heads
        else:
            mismatch_index = mismatch_index[0].item()
            generated_tokens.extend(predicted_tokens[0, :mismatch_index].tolist())
        
        # Update input for next iteration
        inputs = tokenizer(tokenizer.decode(inputs.input_ids[0]) + tokenizer.decode(predicted_tokens[0, :mismatch_index]), return_tensors="pt").to("cuda")
        
        total_tokens = len(generated_tokens)
        
        # Log the number of correct speculations
        writer.add_scalar('Correct_Speculations', mismatch_index, total_tokens)
        
        if total_tokens >= max_tokens:
            break
    
    return tokenizer.decode(generated_tokens[:max_tokens])

def compare_methods(prompt: str, max_tokens: int) -> Tuple[str, str, float, float]:
    """
    Compare vanilla and speculative editing methods.

    Args:
        prompt (str): The input prompt for text generation.
        max_tokens (int): Maximum number of tokens to generate.

    Returns:
        Tuple[str, str, float, float]: Vanilla output, speculative output, vanilla time, speculative time.
    """
    import time

    start_time = time.time()
    vanilla_output = vanilla_edit(prompt, max_tokens)
    vanilla_time = time.time() - start_time

    start_time = time.time()
    speculative_output = speculative_edit(prompt, max_tokens)
    speculative_time = time.time() - start_time

    # Log the time differences
    writer.add_scalar('Time_Difference', vanilla_time - speculative_time, max_tokens)

    return vanilla_output, speculative_output, vanilla_time, speculative_time

# Test the functions
if __name__ == "__main__":
    test_prompt = """Please add a single comment

    ```ts
    export default function Visualization() {
      const [instanceIdInputs, setInstanceIdInputs] = createSignal<
        InstanceId[] | null
      >(null);
      const [storedInput, setStoredInput] = createSignal<string>("");
      const [datapointOptions, setDatapointOptions] = createSignal<PropsInstance[]>(
        []
      );
      const [shouldRefreshGold, setShouldRefreshGold] =
        createSignal<boolean>(false);
      const [showGold, setShowGold] = createSignal<boolean>(false);
      const [selectedGoldRequestId, setSelectedGoldRequestId] = createSignal<
        string | undefined
      >(undefined);
      const [goldInstances, setGoldInstances] = createSignal<
        {
          sessionId: string;
          email: string | undefined;
          requestId: string | undefined;
          dateAdded: Date;
          type: $Enums.CppGoldExampleType;
        }[]
      >([]);
    }
    ```

    ```ts
    """

    max_tokens = 100

    vanilla_output, speculative_output, vanilla_time, speculative_time = compare_methods(test_prompt, max_tokens)

    print(f"Vanilla Edit (Time: {vanilla_time:.4f}s):")
    print(vanilla_output)

    print(f"\nSpeculative Edit (Time: {speculative_time:.4f}s):")
    print(speculative_output)

    print(f"\nTime difference: {vanilla_time - speculative_time:.4f}s")

    # Close the TensorBoard writer
    writer.close()