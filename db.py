# db.py
from datasets import load_dataset

ds = load_dataset("boyiwei/newsqa_filtered_sorted", split="train")  # "validation" or "test" also available
print(ds[:1])


