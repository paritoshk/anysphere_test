import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from text_generation import vanilla_edit, speculative_edit, compare_methods
from next_action_prediction import predict_next_action
from perfect_edit import perfect_edit
from bug_detection import detect_bugs
from multi_hop_context import retrieve_multi_hop_context
from torch.utils.tensorboard import SummaryWriter
from typing import List, Tuple, Dict

def setup_tensorboard():
    return SummaryWriter()

def main():
    # Load model and tokenizer
    model_path = "/workspace/llama3finetune/model"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token  # Set pad token to eos token
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).cuda()
    model.eval()

    # Setup TensorBoard
    writer = setup_tensorboard()

    # Test prompt
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

    # Compare vanilla and speculative editing
    vanilla_output, speculative_output, vanilla_time, speculative_time = compare_methods(model, tokenizer, test_prompt, max_tokens, writer)

    print(f"Vanilla Edit (Time: {vanilla_time:.4f}s):")
    print(vanilla_output)

    print(f"\nSpeculative Edit (Time: {speculative_time:.4f}s):")
    print(speculative_output)

    print(f"\nTime difference: {vanilla_time - speculative_time:.4f}s")

    # Test next action prediction
    next_action = predict_next_action(model, tokenizer, test_prompt)
    print(f"\nPredicted next action: {next_action}")

    # Test perfect edit
    perfect_edit_result = perfect_edit(model, tokenizer, test_prompt, "Add a comment explaining the purpose of the Visualization function")
    print(f"\nPerfect Edit result:\n{perfect_edit_result}")

    # Test bug detection
    bugs = detect_bugs(model, tokenizer, test_prompt)
    print(f"\nDetected bugs:\n{bugs}")

    # Test multi-hop context retrieval
    context = retrieve_multi_hop_context(model, tokenizer, test_prompt, ["file1.ts", "file2.ts"])
    print(f"\nRetrieved multi-hop context:\n{context}")

    # Close the TensorBoard writer
    writer.close()

if __name__ == "__main__":
    main()