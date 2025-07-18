from transformers import AutoModel, AutoTokenizer

model_path = "../all-MiniLM-L6-v2"

tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModel.from_pretrained(model_path, local_files_only=True)