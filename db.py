# db.py
from datasets import load_dataset

ds = load_dataset("Maluuba/newsqa", split="train")  # "validation" or "test" also available
print(ds)
