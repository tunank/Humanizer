<img width="1400" alt="Screenshot 2025-04-22 at 4 35 35 PM" src="https://github.com/user-attachments/assets/1cd40e53-b331-4903-9b20-437530e9def0" />


# Mistral Humanizer Fine-Tuning Project


This project fine-tunes the Mistral-7B-Instruct model to rewrite AI-generated text to sound more human-like, based on the HC3 dataset (reddit_eli5 subset).

## Project Structure

- `notebooks/`: Jupyter notebooks for exploratory data analysis (EDA) and visualization.
- `src/`: Python source code modules for data processing, model utilities, training, and inference.
- `scripts/`: Executable Python scripts to run the training and inference pipelines.
- `results/`: Output directory for saved figures and trained model adapters.
- `requirements.txt`: Python package dependencies.
- `README.md`: This file.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd mistral_humanizer_project
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Hugging Face Token:**
    - Obtain a Hugging Face access token with write permissions (if you plan to upload models).
    - Replace `"YOUR TOKEN"` in `scripts/run_training.py` and `scripts/run_inference.py` with your actual token, or set it as an environment variable (`HUGGING_FACE_HUB_TOKEN`).

## Usage

1.  **Exploratory Data Analysis (Optional):**
    - Open and run the `notebooks/exploration_and_filtering.ipynb` notebook to understand the data and visualize length distributions.

2.  **Run Training:**
    - Modify configuration parameters (model IDs, paths, hyperparameters) directly in `scripts/run_training.py` if needed.
    - Execute the training script:
      ```bash
      python scripts/run_training.py
      ```
    - This will load the data, filter it, format it, load the base model, apply LoRA, train the adapter, and save the final adapter and tokenizer to `results/models/mistral-7b-humanizer-v1-final/`. Training logs and intermediate checkpoints will be in `results/models/mistral-7b-humanizer-v1-training_output/`.

3.  **Run Inference:**
    - Ensure the adapter has been saved successfully by the training script.
    - Modify the `adapter_path` or test prompts in `scripts/run_inference.py` if needed.
    - Execute the inference script:
      ```bash
      python scripts/run_inference.py
      ```
    - This will load the fine-tuned adapter, generate a humanized response for the example prompt, and optionally compare it with the base model's output.

## Configuration

Key configuration parameters (model IDs, dataset names, paths, hyperparameters, filtering settings) are located at the top of the `scripts/run_training.py` and `scripts/run_inference.py` files.
