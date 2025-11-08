from datasets import load_dataset
from peft import LoraConfig
from dataclasses import dataclass
from optimum.neuron import NeuronHfArgumentParser as HfArgumentParser
from optimum.neuron import NeuronSFTConfig, NeuronSFTTrainer, NeuronTrainingArguments
from torch_xla.core.xla_model import is_master_ordinal
from optimum.neuron.models.training import NeuronModelForCausalLM
import torch

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

    model = NeuronModelForCausalLM.from_pretrained(
        model_config.model_id,
        trn_config=NeuronSFTConfig(
            max_seq_length=6000,
            packing=True,
            dataset_kwargs={
                "add_special_tokens": False,
                "append_concat_token": True,
            },
        ),
        torch_dtype=torch.bfloat16,
        use_flash_attention_2=False,
    )

    lora_config = LoraConfig(
        r=model_config.lora_r,
        lora_alpha=model_config.lora_alpha,
        lora_dropout=model_config.lora_dropout, 
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",  
            "gate_proj", "up_proj", "down_proj",      
            "embed_tokens", "lm_head"                   
        ]
    )

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



