from datasets import load_dataset
from peft import LoraConfig

ds = load_dataset("boyiwei/newsqa_filtered_sorted", split="train")  # "validation" or "test" also available
print(ds[:1])

def create_convo(sample):
    system_message = (
        #insert all of the context (i.e. you are a chatbot)
    )
    return {
        #ADJUST BASED ON DBSCHEMA ON HUGGINGFACE
        "messages": [
            {
                "role": "system",
                "content": system_message.format(schema=sample["context"]),
            },
            {"role": "user", "content": sample["question"]},
            {"role": "assistant", "content": sample["answer"] + ";"},
        ]
    }

def finetune():
    ds = ds.shuffle(seed=23)
    train_dataset = ds.select(range(50000))
    eval_dataset = sd.select(range(50000, 50500))

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=64,
        lora_alpha=16,
        lora_dropout=0.1, 
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",  
            "gate_proj", "up_proj", "down_proj",      
            "embed_tokens", "lm_head"                   
        ]
    )


