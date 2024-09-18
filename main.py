import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from text_generation import vanilla_edit, speculative_edit
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from typing import List, Tuple, Dict
import time
from difflib import SequenceMatcher

def setup_tensorboard():
    return SummaryWriter()

def compare_methods(model, tokenizer, prompt: str, max_tokens: int, writer, num_runs: int = 5) -> Dict[str, Dict[str, float]]:
    results = {
        "vanilla": {"times": [], "outputs": []},
        "speculative": {"times": [], "outputs": [], "correct_speculations": []}
    }

    for _ in tqdm(range(num_runs), desc="Comparing methods"):
        # Vanilla edit
        start_time = time.time()
        vanilla_output = vanilla_edit(model, tokenizer, prompt, max_tokens)
        vanilla_time = time.time() - start_time
        results["vanilla"]["times"].append(vanilla_time)
        results["vanilla"]["outputs"].append(vanilla_output)

        # Speculative edit
        start_time = time.time()
        speculative_output, correct_speculations = speculative_edit(model, tokenizer, prompt, max_tokens)
        speculative_time = time.time() - start_time
        results["speculative"]["times"].append(speculative_time)
        results["speculative"]["outputs"].append(speculative_output)
        results["speculative"]["correct_speculations"].append(correct_speculations)

    # Calculate averages and log to TensorBoard
    for method in results:
        avg_time = sum(results[method]["times"]) / num_runs
        writer.add_scalar(f'{method.capitalize()}_Avg_Time', avg_time, max_tokens)
        
        if method == "speculative":
            avg_correct_speculations = sum(results[method]["correct_speculations"]) / num_runs
            writer.add_scalar('Avg_Correct_Speculations', avg_correct_speculations, max_tokens)

    # Calculate similarity
    similarity = SequenceMatcher(None, results["vanilla"]["outputs"][-1], results["speculative"]["outputs"][-1]).ratio()
    writer.add_scalar('Output_Similarity', similarity, max_tokens)

    return results

def main():
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model_name = "meta-llama/Llama-2-7b-chat-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")

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

    print("Starting comparison...")
    results = compare_methods(model, tokenizer, test_prompt, max_tokens, writer)

    print("\nResults:")
    for method in results:
        avg_time = sum(results[method]["times"]) / len(results[method]["times"])
        print(f"\n{method.capitalize()} Edit:")
        print(f"  Average Time: {avg_time:.4f}s")
        print(f"  Last Output: {results[method]['outputs'][-1]}")
        
        if method == "speculative":
            avg_correct_speculations = sum(results[method]["correct_speculations"]) / len(results[method]["correct_speculations"])
            print(f"  Average Correct Speculations: {avg_correct_speculations:.2f}")

    similarity = SequenceMatcher(None, results["vanilla"]["outputs"][-1], results["speculative"]["outputs"][-1]).ratio()
    print(f"\nOutput Similarity: {similarity:.4f}")

    print("\nDetailed results have been logged to TensorBoard.")
    writer.close()

if __name__ == "__main__":
    main()