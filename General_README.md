

## Project Overview

This project implements speculative decoding for text generation using the Meta-Llama/Meta-Llama-3.1-8B-Instruct model. It includes functionality for both vanilla and speculative text generation, along with performance comparison and logging.

## Setup and Installation

1. Connect to your SSH environment (e.g., RunPod).

2. Navigate to the workspace and clone the repository:
   ```
   cd workspace
   git clone https://github.com/paritoshk/anysphere_test
   cd anysphere_test
   ```

3. Create a virtual environment:
   ```
   python3 -m venv anysphere_test
   source anysphere_test/bin/activate
   ```

4. Install the required packages:
   ```
   pip install torch accelerate transformers sentencepiece huggingface_hub tensorboard
   ```

5. Set your Hugging Face authentication token:
   ```
   export HF_AUTH_TOKEN="YOUR_TOKEN_HERE"
   ```

## Running the Project

1. Download the model:
   ```
   python download_model.py
   ```

2. Test the model:
   ```
   python test_model.py
   ```

3. Run the main script:
   ```
   python main.py
   ```

## Project Structure and Logic

The project consists of three main Python scripts:

1. `download_model.py`: Downloads the Meta-Llama/Meta-Llama-3.1-8B-Instruct model from Hugging Face.

2. `test_model.py`: Verifies that the model is correctly downloaded and can generate text.

3. `main.py`: Implements the core functionality, including:
   - Vanilla text generation
   - Speculative text generation
   - Performance comparison between the two methods
   - TensorBoard logging for analysis

### Key Components in `main.py`

1. `vanilla_edit(prompt: str, max_tokens: int) -> str`:
   - Performs standard text generation using the loaded model.
   - Uses greedy decoding (no sampling).

2. `speculative_edit(prompt: str, max_tokens: int, num_heads: int = 3, codebase_context: Dict[str, str] = None) -> str`:
   - Implements speculative decoding for faster text generation.
   - Uses multiple heads to predict several tokens at once.
   - Incorporates optional codebase context for improved generation.

3. `compare_methods(prompt: str, max_tokens: int) -> Tuple[str, str, float, float]`:
   - Compares the performance of vanilla and speculative editing.
   - Measures and logs the time difference between the two methods.

4. TensorBoard Logging:
   - Tracks the number of correct speculations and time differences between methods.

## Warnings and Notes

1. The current implementation may generate warnings related to `do_sample`, `temperature`, and `top_p` settings. These can be addressed by adjusting the generation parameters in the code.

2. An attention mask warning suggests that you may need to explicitly set the attention mask for more reliable results.

3. The `past_key_values` warning indicates that a deprecated method is being used. Consider updating to use the appropriate `Cache` class in future versions.

## Next Steps and Improvements

1. Implement a `next_action_prediction` function for predicting user actions in the editor.
2. Develop a `perfect_edit` function for larger, asynchronous edits.
3. Create a `bug_detection` function for code analysis and suggestions.
4. Implement a `multi_hop_context` retrieval system for improved codebase understanding.

## Troubleshooting

If you encounter issues with package installation or model downloading, ensure that:
1. Your Hugging Face token is correctly set and has the necessary permissions.
2. You have sufficient disk space for the model.
3. Your internet connection is stable.

For any persistent issues, please refer to the project's GitHub repository or contact the maintainers.