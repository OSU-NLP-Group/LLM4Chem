import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModelForCausalLM


BASE_MODELS = {
    'osunlp/LlaSMol-Mistral-7B': 'mistralai/Mistral-7B-v0.1',
}


def load_tokenizer_and_model(model_name, base_model=None):
    if base_model is None:
        if model_name in BASE_MODELS:
            base_model = BASE_MODELS[model_name]
    assert base_model is not None, "Please assign the corresponding base model to the argument 'base_model'."

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = '<pad>'
    tokenizer.sep_token = '<unk>'
    tokenizer.cls_token = '<unk>'
    tokenizer.mask_token = '<unk>'

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    try:
        if torch.backends.mps.is_available():
            device = "mps"
    except:  # noqa: E722
        pass
    
    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
            
        model = PeftModelForCausalLM.from_pretrained(
            model,
            model_name,
            torch_dtype=torch.bfloat16,
        )
    else:
        raise NotImplementedError("No implementation for loading model on CPU yet.")
    
    model = model.merge_and_unload()

    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    return tokenizer, model
