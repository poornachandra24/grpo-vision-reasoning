from datasets import load_dataset
import os

print("Loading dataset...")
dataset = load_dataset("AI4Math/MathVista", split="testmini")
print("Dataset loaded.")
example = dataset[0]
print("Image Type:", type(example['image']))
print("Decoded Image Type:", type(example['decoded_image']))
print("Decoded Image Value:", example['decoded_image'])
