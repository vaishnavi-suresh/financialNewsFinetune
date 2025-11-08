from dataclasses import dataclass, field
from peft import LoraConfig
from dataclasses import dataclass
from optimum.neuron import NeuronHfArgumentParser as HfArgumentParser
from optimum.neuron import NeuronSFTConfig, NeuronSFTTrainer, NeuronTrainingArguments
from torch_xla.core.xla_model import is_master_ordinal
from optimum.neuron.models.training import NeuronModelForCausalLM
from datasets import load_dataset 
import torch
import json
from transformers import AutoTokenizer


@dataclass
class model_config:
    model_id: str = field(
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        metadata={
            "help": "The model that you want to train from the Hugging Face hub."
        },
    )
    tokenizer_id: str = field(
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        metadata={"help": "The tokenizer used to tokenize text for fine-tuning."},
    )
    lora_r: int = field(
        default=32,
        metadata={"help": "LoRA r value to be used during fine-tuning."},
    )
    lora_alpha: int = field(
        default=16,
        metadata={"help": "LoRA alpha value to be used during fine-tuning."},
    )
    lora_dropout: float = field(
        default=0.1,
        metadata={"help": "LoRA dropout value to be used during fine-tuning."},
    )
    secret_name: str = field(
        default="huggingface/token",
        metadata={"help": "AWS Secrets Manager secret name containing Hugging Face token."},
    )
    secret_region: str = field(
        default="us-west-2",
        metadata={"help": "AWS region where the secret is stored."},
    )

cfg = model_config()


def create_convo(sample):
    system_message = (
        f"You are a financial news analyst assistant. Answer questions accurately based on the provided news context. This is the news article to reference: {sample['story_text']}"
    )
    return {
        #ADJUST BASED ON DBSCHEMA ON HUGGINGFACE
        "messages": [
            {
                "role": "system",
                "content": system_message,
            },
            {"role": "user", "content": f'question: {sample["question"]}'},
            {"role": "assistant", "content": sample["answer"]},
        ]
    }

def finetune(data, training_args):
    data = data.shuffle(seed=23)
    train_dataset = data.select(range(50000))
    eval_dataset = data.select(range(50000, 50500))

    train_dataset = train_dataset.map(
        create_convo, remove_columns=train_dataset.features, batched=False
    )
    eval_dataset = eval_dataset.map(
        create_convo, remove_columns=eval_dataset.features, batched=False
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_id)



    model = NeuronModelForCausalLM.from_pretrained(
        cfg.model_id,
        training_args.trn_config, 
        torch_dtype= torch.bfloat16,
        use_flash_attention_2=False,
    )

    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout, 
        target_modules=[
            "q_proj",
            "gate_proj",
            "v_proj",
            "o_proj",
            "k_proj",
            "up_proj",
            "down_proj",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )

    args = training_args.to_dict()

    sft_config = NeuronSFTConfig(**training_args.to_dict())

    trainer = NeuronSFTTrainer(
        args=sft_config,
        model=model,
        peft_config=lora_config,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train()
    trainer.save_model("finetuned_model")







def main():
    """Test the create_convo function"""
    print("="*60)
    print("TESTING create_convo FUNCTION")
    print("="*60)
    cfg = model_config()

    
    training_args = NeuronTrainingArguments(
        output_dir="outputs",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=1,
        logging_steps=10,
        save_steps=1000,
        bf16=True,
    )
    
    # Load one sample
    ds = load_dataset("boyiwei/newsqa_filtered_sorted", split="train")
    sample = ds[0]

    finetune(ds, training_args)
    
    

if __name__ == "__main__":
    main()