import argparse
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Set up argument parser
parser = argparse.ArgumentParser(description="Load model and generate text with transformers.")
parser.add_argument(
    "--model_id", type=str, required=True, help="Path or model identifier for the pre-trained model"
)
parser.add_argument(
    "--prompt", type=str, required=True, help="Prompt text to generate text from"
)
args = parser.parse_args()

# Use the model_id and prompt from the command line arguments
model_id = args.model_id
prompt = args.prompt

# Load tokenizer and set pad_token_id explicitly
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

# Ensure `pad_token_id` is set
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# Define custom eos_token_ids (if needed)
terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

# Load the model and tokenizer using the specified model_id
pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16, "pad_token_id": tokenizer.pad_token_id},
    device_map="auto",
)

# Generate text from the prompt with explicit truncation
outputs = pipeline(
    prompt,
    max_length=100,
    truncation=True,  # Explicitly enable truncation
    eos_token_id=terminators,
)

# Print the generated text
print(outputs[0]["generated_text"])
