import torch
from transformers import AutoTokenizer
from optimum.neuron.models.training import NeuronModelForCausalLM


def build_messages(story_text: str, question: str):
    system_message = (
        f"You are a financial news analyst assistant. "
        f"Answer this question on current events"
    )
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": f"question: {question}"},
    ]


def main():
    model_dir = "finetuned_model"

    print(f"Loading model and tokenizer from `{model_dir}`...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = NeuronModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        use_flash_attention_2=False,
    )


    while True:
        story_text = input("Enter your question (or type 'quit' to exit):\n> ")
        if story_text.lower() in ["quit", "exit"]:
            break

        question = input("\nEnter your question:\n> ")
        if question.lower() in ["quit", "exit"]:
            break

        messages = build_messages(story_text, question)
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer(prompt, return_tensors="pt")

        print("Generating answer...")
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )

        generated_ids = outputs[0, inputs["input_ids"].shape[1]:]
        answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        print("\n===================== ANSWER =====================")
        print(answer)
        print("==================================================")


if __name__ == "__main__":
    main()
