import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config, AdamW

# Define the paths to input and label files
label_file_path = '../data/pre-train-data.txt'
input_file_path = '../data/mask-pre-train-data.txt'
# Load the input and label data from the files
def load_data_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = f.readlines()
    return data

input_data = load_data_from_file(input_file_path)
label_data = load_data_from_file(label_file_path)

# Create a custom Dataset
class CustomDataset(Dataset):
    def __init__(self, tokenizer, input_data, label_data, max_length=128):
        self.tokenizer = tokenizer
        self.input_data = input_data
        self.label_data = label_data
        self.max_length = max_length

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        inputs = self.tokenizer.encode_plus(
            self.input_data[idx],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        labels = self.tokenizer.encode_plus(
            self.label_data[idx],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )

        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': labels['input_ids'].squeeze(),
            'labels_attention_mask': labels['attention_mask'].squeeze(),
        }

# Define the T5 model and tokenizer
model_name = 't5-base'
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Prepare data and split into train and validation sets
dataset = CustomDataset(tokenizer, input_data, label_data)
val_size = 100#int(0.1 * len(dataset))
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Define the DataLoader for train and validation sets
batch_size = 32
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

# Define the optimizer and training parameters
optimizer = AdamW(model.parameters(), lr=1e-4)

# Fine-tuning the model
num_epochs = 3
print_step = 100
for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0

    for step, batch in enumerate(train_dataloader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        labels_attention_mask = batch['labels_attention_mask'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % print_step == 0:
            print(f"Epoch: {epoch + 1}, Step: {step}, Loss: {loss.item()}")

        if step % print_step  == 0 and step > 2:
            model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for val_batch in val_dataloader:
                    input_ids = val_batch['input_ids'].to(device)
                    attention_mask = val_batch['attention_mask'].to(device)
                    labels = val_batch['labels'].to(device)
                    labels_attention_mask = val_batch['labels_attention_mask'].to(device)
                    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                    total_val_loss += loss.item()
                for i, val_batch in enumerate(val_dataloader):
                    if i >= 10:
                        break

                    input_ids = val_batch['input_ids'].to(device)
                    attention_mask = val_batch['attention_mask'].to(device)
                    labels = val_batch['labels'].to(device)

                    input_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
                    label_text = tokenizer.decode(labels[0], skip_special_tokens=True)

                    outputs = model.generate(input_ids, attention_mask=attention_mask, max_length=128)
                    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

                    print(f"Inpu: {input_text}")
                    print(f"Gold: {label_text}")
                    print(f"Pred: {generated_text}\n")

            avg_val_loss = total_val_loss / len(val_dataloader)
            print(f"Validation Loss: {avg_val_loss}")

            model.train()

    avg_train_loss = total_train_loss / len(train_dataloader)
    print(f"Epoch {epoch + 1} completed. Average Training Loss: {avg_train_loss}")
