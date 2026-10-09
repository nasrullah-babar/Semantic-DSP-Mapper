from unsloth import FastLanguageModel
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="lora_safe_audio_model",
    max_seq_length=1024,
    dtype=None,
    load_in_4bit=True,
)

FastLanguageModel.for_inference(model)

prompt_template = 'Below is an instruction that describes a desired audio characteristic. Write the corresponding digital audio plugin parameter values to achieve this sound.\n\n### Instruction:\n{semantic_text}\n\n### VST Parameters:\n'

inputs = tokenizer(
    [prompt_template.format(semantic_text="Make the bass guitar sound huge, wide, and subby")], 
    return_tensors="pt"
).to("cuda")

outputs = model.generate(**inputs, max_new_tokens=128, use_cache=True)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].split("### VST Parameters:\n")[-1])