from datasets import load_dataset

def load_financial_news_dataset():
    """Load and format NewsQA dataset for financial Q&A"""
    
    print("   → Loading NewsQA dataset from Hugging Face...")
    dataset = load_dataset("microsoft/newsqa", split="train")
    
    # Use subset for faster hackathon training
    print(f"   → Using subset of 1000 examples for faster training...")
    dataset = dataset.select(range(1000))
    
    def format_example(example):
        """Format each example for instruction-following"""
        context = example.get('story_text', '')
        question = example.get('question', '')
        
        # Extract answer from character positions
        try:
            if example.get('answer_char_ranges') and len(example['answer_char_ranges']) > 0:
                answer_start = example['answer_char_ranges'][0][0]
                answer_end = example['answer_char_ranges'][0][1]
                answer = context[answer_start:answer_end].strip()
            else:
                return None  # Skip examples without answers
        except:
            return None
        
        # Skip if answer is empty or too short
        if not answer or len(answer) < 5:
            return None
        
        # Create instruction-following format
        prompt = f"""### Financial News Context:
{context[:1000]}

### Question:
{question}

### Answer:
{answer}"""
        
        return {"text": prompt}
    
    # Apply formatting and filter out None values
    print(f"   → Formatting examples...")
    dataset = dataset.map(format_example)
    dataset = dataset.filter(lambda x: x['text'] is not None)
    
    print(f"   ✓ Dataset ready: {len(dataset)} valid examples")
    return dataset

if __name__ == "__main__":
    # Test loading
    print("\nTesting dataset loading...")
    dataset = load_financial_news_dataset()
    print("\n" + "="*60)
    print("SAMPLE EXAMPLE:")
    print("="*60)
    print(dataset[0]['text'][:500] + "...")