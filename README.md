# LLaMA-LoRA-PubMed-FineTuning

This repository contains a Python script (`llama_lora_pubmed_finetuning.py`) that demonstrates how to fine-tune the LLaMA-3.2-1B model using Low-Rank Adaptation (LoRA) on a subset of PubMed abstracts for causal language modeling. Below is an overview of the script, its purpose, and how to use it.

## Overview

The script performs the following tasks:
1. **Environment Setup**: Configures the environment by checking for GPU availability and installing required libraries.
2. **Data Acquisition**: Downloads a random subset (1% by default) of PubMed XML files from the NCBI FTP server.
3. **Data Preprocessing**: Extracts abstracts from the downloaded XML files and creates a dataset using the `datasets` library.
4. **Tokenization**: Tokenizes the abstracts using the LLaMA tokenizer for training.
5. **Model Fine-Tuning**: Applies LoRA to fine-tune the LLaMA-3.2-1B model on the processed dataset.
6. **Text Generation**: Generates text using the fine-tuned model with a formatted output.

The script is designed to run in a Colab environment or a similar setup with GPU support, leveraging libraries like `transformers`, `peft`, `datasets`, and `accelerate`.

## Prerequisites

- Python 3.8+
- GPU (recommended for faster training, e.g., NVIDIA T4)
- Required libraries (installed automatically in the script):
  - `torch`
  - `transformers`
  - `peft`
  - `datasets`
  - `accelerate`
  - `ftplib`
  - `gzip`
  - `shutil`
  - `xml.etree.ElementTree`

Ensure you have a Hugging Face account and a valid access token for downloading the LLaMA model.

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/llama-lora-pubmed-finetuning.git
   cd llama-lora-pubmed-finetuning
   ```

2. Install dependencies (if not running in Colab):
   ```bash
   pip install torch transformers peft datasets accelerate
   ```

3. Log in to Hugging Face CLI to access the LLaMA model:
   ```bash
   huggingface-cli login
   ```

## Usage

1. **Run the Script**:
   Execute the script in a Python environment with GPU support:
   ```bash
   python llama_lora_pubmed_finetuning.py
   ```

2. **Key Steps in the Script**:
   - **GPU Check**: Detects if a GPU is available and sets the device accordingly (`cuda` or `cpu`).
   - **PubMed Data Download**: Connects to the NCBI FTP server, downloads a random 1% subset of `.xml.gz` files, and extracts them to XML format.
   - **Data Processing**: Parses XML files to extract abstracts and creates a `Dataset` object, split into 90% training and 10% validation sets.
   - **Tokenization**: Uses the LLaMA-3.2-1B tokenizer with padding and truncation to prepare the dataset for training.
   - **LoRA Fine-Tuning**: Applies LoRA with a reduced rank (`r=2`) and targets the `q_proj` module to fine-tune the model efficiently.
   - **Training**: Trains the model for 1 epoch with a batch size of 16, using mixed precision (FP16) for faster training.
   - **Model Saving**: Saves the fine-tuned model and tokenizer to `./llama-lora-pubmed-subset`.
   - **Text Generation**: Generates text from a prompt (`"Recent advances in cancer immunotherapy suggest"`) with formatted output.

3. **Output**:
   - The script downloads and processes PubMed abstracts, fine-tunes the model, and saves it to the specified directory.
   - It generates a formatted text response based on the input prompt, wrapped at 80 characters per line for readability.

## Configuration

You can modify the following parameters in the script:
- `SUBSET_RATIO`: Adjust the percentage of PubMed files to download (default: 1%).
- `RANDOM_SEED`: Set for reproducibility (default: 42).
- `max_length` in `tokenize_function`: Controls the maximum token length (default: 512).
- LoRA parameters:
  - `r`: Rank of LoRA adaptation (default: 2).
  - `lora_alpha`: Scaling factor (default: 4).
  - `lora_dropout`: Dropout rate (default: 0.05).
- Training arguments:
  - `per_device_train_batch_size`: Batch size (default: 16).
  - `num_train_epochs`: Number of epochs (default: 1).
  - `learning_rate`: Learning rate (default: 2e-5).
  - `max_steps`: Maximum training steps (default: 1000).

## Output Files

- `./pubmed_subset/`: Directory containing downloaded and extracted PubMed XML files.
- `./llama-lora-pubmed-subset/`: Directory with the fine-tuned model and tokenizer.
- Console output: Displays the number of files downloaded, dataset sizes, training progress, and a generated text sample.

## Example Generated Output

The script generates text based on the prompt `"Recent advances in cancer immunotherapy suggest"`. The output is formatted with 80-character line wrapping and indentation for readability.

Example:
```
Generated Response:
========================================
Recent advances in cancer immunotherapy suggest that novel approaches, such as
    checkpoint inhibitors and CAR-T cell therapies, are transforming treatment
    paradigms. These methods leverage the immune system to target cancer cells
    with high specificity, improving patient outcomes in various malignancies.
    Ongoing research focuses on optimizing these therapies to overcome resistance
    mechanisms and enhance efficacy.
========================================
```

## Notes

- The script assumes access to the LLaMA-3.2-1B model via Hugging Face. Ensure you have the necessary permissions.
- The PubMed dataset is large; the script uses a 1% subset to reduce resource demands, but you can adjust `SUBSET_RATIO` for larger datasets.
- Training is optimized for a single epoch with a small LoRA rank to fit on modest hardware (e.g., Colab T4 GPU).
- The generated text may vary due to the stochastic nature of the model (`temperature=0.7`, `top_p=0.9`).

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments

- Built using the `transformers` and `peft` libraries from Hugging Face.
- PubMed data sourced from the NCBI FTP server.
- Inspired by fine-tuning tutorials for large language models.
