import os
import torch
from tqdm import tqdm
import numpy as np
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer
from torch.nn import CrossEntropyLoss

# Define model paths
model_paths = [
    "/kaggle/working/DL2-2024/EquiNLG/nlg-bias-master/models/regard_v2.1/bert_regard_v2.1/checkpoint-90",
    "/kaggle/working/DL2-2024/EquiNLG/nlg-bias-master/models/regard_v2.1/bert_regard_v2.1_2/checkpoint-90",
    "/kaggle/working/DL2-2024/EquiNLG/nlg-bias-master/models/regard_v2.1/bert_regard_v2.1_3/checkpoint-90"
]

# Load models and tokenizers
models = []
tokenizers = []
for model_path in model_paths:
    config = BertConfig.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path, config=config)
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    models.append(model)
    tokenizers.append(tokenizer)

# labels = ["negative", "neutral", "positive", "other"]
labels = [-1, 0, 1, 2]
label_map = {i: label for i, label in enumerate(labels)}

def predict(text):
    input_ids = []
    attention_masks = []

    for tokenizer in tokenizers:
        inputs = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=128,
            pad_to_max_length=True,
            return_tensors="pt",
            return_attention_mask=True,
            truncation=True
        )
        input_ids.append(inputs["input_ids"].to('cuda' if torch.cuda.is_available() else 'cpu'))
        attention_masks.append(inputs["attention_mask"].to('cuda' if torch.cuda.is_available() else 'cpu'))

    preds = []

    for model, input_id, attention_mask in tqdm(zip(models, input_ids, attention_masks)):
        with torch.no_grad():
            outputs = model(input_ids=input_id, attention_mask=attention_mask)
            logits = outputs[0]
            pred = np.argmax(logits.detach().cpu().numpy(), axis=1).item()
            preds.append(pred)

    majority_label = max(set(preds), key=preds.count)
    return label_map[majority_label]

def process_file(input_file):
    output_file = f"{os.path.splitext(input_file)[0]}_preds.tsv"

    with open(input_file, "r") as reader, open(output_file, "w") as writer:
        for line in reader:
            text = line.strip()
            regard_label = predict(text)
            writer.write(f"{regard_label}\t{text}\n")

    print(f"Predictions written to {output_file}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Predict regard scores for input text file")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input text file")

    args = parser.parse_args()

    process_file(args.input_file)
