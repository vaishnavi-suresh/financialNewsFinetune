from dataclasses import dataclass, field
from peft import LoraConfig
from datasets import load_dataset
import json

ds = load_dataset("boyiwei/newsqa_filtered_sorted", split="train")  # "validation" or "test" also available
print(ds[:1])

def create_convo(sample):
    system_message = (
        "You are a financial news analyst assistant. Answer questions accurately based on the provided news context."
    )
    return {
        #ADJUST BASED ON DBSCHEMA ON HUGGINGFACE
        "messages": [
            {
                "role": "system",
                "content": system_message,
            },
            {"role": "user", "content": f'question: {sample["question"]} news source:{sample["story_text"]}'},
            {"role": "assistant", "content": sample["answer"]},
        ]
    }

def finetune():
    ds = ds.shuffle(seed=23)
    train_dataset = ds.select(range(50000))
    eval_dataset = sd.select(range(50000, 50500))

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=32,
        lora_alpha=16,
        lora_dropout=0.1, 
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",  
            "gate_proj", "up_proj", "down_proj",      
            "embed_tokens", "lm_head"                   
        ]
    )


def main():
    """Test the create_convo function"""
    print("="*60)
    print("TESTING create_convo FUNCTION")
    print("="*60)
    
    # Load one sample
    ds = load_dataset("boyiwei/newsqa_filtered_sorted", split="train")
    sample = ds[0]
    
    # Apply formatting
    formatted = create_convo(sample)
    
    # Print as pretty JSON
    print("\n📄 FORMATTED OUTPUT:\n")
    print(json.dumps(formatted, indent=2))
    
    print("\n" + "="*60)
    print("Function is working correctly!")
    print("="*60)

if __name__ == "__main__":
    main()