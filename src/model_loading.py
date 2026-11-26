import bitsandbytes as bnb
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig

from .prepare import LANGUAGES


MODELS = {
    "madlad": ("google/madlad400-3b-mt", "fa184c675da0b5c9e1c8694fccd4e12e2d422094"),
    "gemma": ("google/gemma-3-1b-it", "dcc83ea841ab6100d6b47a070329e1ba4cf78752"),
}
NAMES = {"en": "English", **LANGUAGES}


def load(args, quantized=False):
    repo, revision = MODELS[args.model]
    path = args.models / repo.split("/")[-1] if args.models else repo
    revision_args = {} if args.models else {"revision": revision}
    seq2seq = args.model == "madlad"
    tokenizer = AutoTokenizer.from_pretrained(path, **revision_args)
    tokenizer.padding_side = "right" if seq2seq else "left"
    if seq2seq:
        for code in NAMES:
            tag = f"<2{code}>"
            assert tokenizer.convert_tokens_to_ids(tag) in tokenizer.encode(tag)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                      bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
    cls = AutoModelForSeq2SeqLM if seq2seq else AutoModelForCausalLM
    model = cls.from_pretrained(path, device_map={"": 0}, dtype=dtype,
                                quantization_config=quantization if quantized else None,
                                **revision_args)
    if quantized:
        layers = [m for m in model.modules() if isinstance(m, bnb.nn.Linear4bit)]
        assert layers and all(m.weight.quant_state.quant_type == "nf4" for m in layers)
    return model, tokenizer


def encode_source(tokenizer, row, seq2seq):
    source, target = row["direction"].split("-")
    if seq2seq:
        return tokenizer.encode(f"<2{target}> {row['source']}")
    prompt = (f"Translate the following text from {NAMES[source]} to {NAMES[target]}. "
              f"Return only the translation.\n\n{row['source']}")
    return tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                         tokenize=True, add_generation_prompt=True)


def attach(model, name):
    seq2seq = name == "madlad"
    adapter = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
        target_modules=["q", "v"] if seq2seq else ["q_proj", "v_proj"],
        task_type="SEQ_2_SEQ_LM" if seq2seq else "CAUSAL_LM",
    ))
    config = adapter.peft_config["default"]
    config.base_model_name_or_path, config.revision = MODELS[name]
    return adapter
