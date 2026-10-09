from datasets import load_dataset
dataset = load_dataset("seungheondoh/socialfx-original")
print("COLUMNS:", dataset["eq"].column_names)
print("FIRST ROW:", dataset["eq"][0])