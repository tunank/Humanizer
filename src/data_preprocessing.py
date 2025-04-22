import numpy as np
from datasets import load_dataset

def load_hc3_dataset(dataset_name, subset, split="train", sample_size=None):
    """Loads the HC3 dataset, optionally sampling it."""
    print(f"Loading dataset: {dataset_name} ({subset})...")
    dataset = load_dataset(dataset_name, subset, split=split)
    print(f"Dataset loaded with {len(dataset)} examples")
    if sample_size:
        dataset = dataset.shuffle(seed=42).select(range(sample_size))
        print(f"Sampled down to {len(dataset)} examples")
    return dataset

def get_longest_human_length(example):
    """Calculates the word count of the longest human answer in an example."""
    if not example.get("human_answers"):
        return 0
    human_lengths = [
        len(answer.split()) for answer in example["human_answers"] if answer
    ]
    return max(human_lengths) if human_lengths else 0

def filter_dataset_by_length(dataset, min_length=75, iqr_multiplier=1.5):
    """Filters the dataset based on the length of the longest human answer."""
    print("\n--- Filtering Dataset by Longest Human Answer Length ---")
    if not dataset:
        print("Dataset is empty, cannot filter.")
        return dataset

    longest_human_lengths = dataset.map(
        lambda x: {"longest_len": get_longest_human_length(x)},
        batched=True,  # Process in batches for speed
        num_proc=4,  # Use multiple processes
    )["longest_len"]

    if not longest_human_lengths:
        print("Could not calculate longest human lengths. Skipping filtering.")
        return dataset

    q1 = np.percentile(longest_human_lengths, 25)
    q3 = np.percentile(longest_human_lengths, 75)
    iqr = q3 - q1
    upper_bound = q3 + iqr_multiplier * iqr
    lower_bound = min_length  # Use the user-defined minimum

    print(f"Q1: {q1:.2f}, Q3: {q3:.2f}, IQR: {iqr:.2f}")
    print(f"Effective Lower Bound (min_length): {lower_bound} words")
    print(f"Effective Upper Bound (Q3 + {iqr_multiplier}*IQR): {upper_bound:.2f} words")

    original_size = len(dataset)
    filtered_dataset = dataset.filter(
        lambda x: lower_bound <= get_longest_human_length(x) <= upper_bound
    )
    filtered_size = len(filtered_dataset)

    print(f"Original dataset size: {original_size}")
    print(f"Filtered dataset size: {filtered_size}")
    print(f"Number of examples removed: {original_size - filtered_size}")

    if filtered_size == 0:
        print("Warning: Filtered dataset is empty!")

    return filtered_dataset

def format_example_longest_detailed(example):
    """
    Formats the dataset example for Mistral Instruct fine-tuning.
    Uses the LONGEST human answer as the target response and includes
    an enhanced instruction prompt. Returns empty text on error or missing data.
    """
    try:
        question = example.get("question")
        human_answers = example.get("human_answers")
        chatgpt_answers = example.get("chatgpt_answers")

        if (
            not question
            or not human_answers
            or not chatgpt_answers
            or not chatgpt_answers[0]
        ):
            return {"text": ""}

        ai_response = chatgpt_answers[0]
        valid_human_answers = [
            ans for ans in human_answers if ans and ans.strip()
        ]
        if not valid_human_answers:
            return {"text": ""}

        longest_human_answer = max(
            valid_human_answers, key=lambda x: len(x.split())
        )

        prompt_instruction = (
            "Rewrite the following AI response to the question so it sounds more "
            "natural and human-like, while ensuring the key information and "
            "level of detail are preserved."
        )

        text = (
            f"<s>[INST] {prompt_instruction}\n\n"
            f'Question: "{question}"\n\n'
            f'AI Response: "{ai_response}" [/INST] '
            f"{longest_human_answer} </s>"
        )
        return {"text": text}

    except (KeyError, IndexError, TypeError) as e:
        print(f"Error formatting example: {e}. Skipping.") # Optional debug
        return {"text": ""}

def prepare_training_dataset(dataset):
    """Applies formatting and filters out empty examples."""
    print("\nFormatting dataset for training...")
    formatted_dataset = dataset.map(
        format_example_longest_detailed,
        batched=True, # Process in batches
        num_proc=4,   # Use multiple processes
        remove_columns=dataset.column_names # Remove original columns
    )

    # Filter out examples that resulted in empty text
    original_count = len(formatted_dataset)
    formatted_dataset = formatted_dataset.filter(lambda x: len(x["text"]) > 0)
    filtered_count = len(formatted_dataset)
    print(f"Removed {original_count - filtered_count} examples during formatting (due to missing data or errors).")
    print(f"Final formatted dataset size: {filtered_count}")

    if filtered_count > 0:
        print("\nExample formatted text:")
        print(formatted_dataset[0]["text"])
    else:
        print("Warning: Formatted dataset is empty after filtering.")

    return formatted_dataset
