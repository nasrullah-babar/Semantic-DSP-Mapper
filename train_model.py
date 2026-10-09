import torch
import json
from unsloth import FastLanguageModel
from datasets import load_dataset, Dataset
from trl import SFTTrainer
from transformers import TrainingArguments

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-1.5B-bnb-4bit",
    max_seq_length=1024,
    dtype=None,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

raw_dataset = load_dataset("seungheondoh/socialfx-original")
processed_entries = []

split_mapping = {"eq": "EQ", "reverb": "Reverb", "comp": "Compressor"}

for split_name, plugin_label in split_mapping.items():
    for entry in raw_dataset[split_name]:
        try:
            extra_str = entry["extra"].replace("'", '"').replace("nan", "null")
            extra_data = json.loads(extra_str)
            
            consistency = extra_data.get("ratings_consistency")

            if extra_data.get("lang") != "English" or consistency is None or float(consistency) < 0.70:
                continue
        except Exception:
            continue
            
        parameters = {str(feat).replace("RSC_", "").replace("_band", ""): round(val, 2) 
                         for feat, val in zip(entry["param_keys"], entry["param_values"])}
        
        prompt = f'Below is an instruction that describes a desired audio characteristic. Write the corresponding digital audio plugin parameter values to achieve this sound.\n\n### Instruction:\nMake the audio sound "{entry["text"]}"\n\n### VST Parameters:\n{plugin_label}: {parameters}'
        processed_entries.append({"text": prompt})

dataset = Dataset.from_list(processed_entries)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=1024,
    dataset_num_proc=2,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=60,
        learning_rate=2e-4,
        fp16=False,
        bf16=True,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
    ),
)

trainer.train()
model.save_pretrained("lora_safe_audio_model")
tokenizer.save_pretrained("lora_safe_audio_model")