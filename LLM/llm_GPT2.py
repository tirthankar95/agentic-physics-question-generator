from transformers import  AutoTokenizer, AutoModelForCausalLM 

model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

class GPT2LLM():
    """Anthropic Claude LLM implementation."""
    
    def __init__(self, model_name: str):
        super().__init__()
        self.model_name = model_name
        self._client = None

    def get_response(self, prompt):
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        gen_tokens = model.generate(
            input_ids,
            do_sample=True,
            temperature=0.1,
            max_length=256,
        )
        return tokenizer.batch_decode(gen_tokens)[0]