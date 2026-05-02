from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_path = "model/final_model"

model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

model.eval()

print("Model loaded")