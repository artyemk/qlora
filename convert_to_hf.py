import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, LlamaTokenizerFast


BASE_MODEL = "huggyllama/llama-7b"
LORA_WEIGHTS = "Kernel/qlora_guanaco"

tokenizer = LlamaTokenizerFast.from_pretrained(BASE_MODEL)


tokenizer.add_special_tokens(
    {
        "eos_token": tokenizer.convert_ids_to_tokens(tokenizer.eos_token_id),
        "bos_token": tokenizer.convert_ids_to_tokens(tokenizer.bos_token_id),
        "unk_token": tokenizer.convert_ids_to_tokens(tokenizer.pad_token_id),
    }
)
    
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    load_in_8bit=False,
    torch_dtype=torch.float16,
    device_map="auto",
    offload_folder="offload", 
)
    
model = PeftModel.from_pretrained(
    model, 
    LORA_WEIGHTS, 
    torch_dtype=torch.float16,
    device_map="auto",
    offload_folder="offload", 

)

model = model.merge_and_unload()
model.save_pretrained("Kernel/qlora_guanaco_merged")
