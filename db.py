from dataclasses import dataclass, field
from peft import LoraConfig
from datasets import load_dataset

ds = load_dataset("boyiwei/newsqa_filtered_sorted", split="train")  # "validation" or "test" also available
print(ds[:1])

def finetune():
    ds = ds.shuffle(seed=23)
    train_dataset = ds.select(range(50000))
    eval_dataset = sd.select(range(50000, 50500))

    
