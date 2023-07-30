import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Define the masked token prediction dataset
class MaskedTokenDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        inputs = self.tokenizer.encode_plus(
            self.texts[index],
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        input_ids = inputs['input_ids'].squeeze()
        mask_labels = torch.clone(input_ids)

        # Mask 15% of the tokens (similar to BERT)
        masked_indices = torch.rand(input_ids.size()).argsort()[:int(0.15 * input_ids.size()[0])]
        mask_labels[masked_indices] = -100  # -100 is used for masked token prediction
        return input_ids, mask_labels

# Fine-tuning function
def fine_tune_t5_on_masked_task(train_texts, epochs=3, batch_size=16, max_length=512):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the pre-trained T5 model and tokenizer
    model_name = 't5-base'  # You can also use 't5-base', 't5-large', etc.
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)

    # Create a dataset and DataLoader
    train_dataset = MaskedTokenDataset(train_texts, tokenizer, max_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Prepare the optimizer and loss function
    optimizer = AdamW(model.parameters(), lr=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)

    # Fine-tuning loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            input_ids, mask_labels = batch
            input_ids = input_ids.to(device)
            mask_labels = mask_labels.to(device)

            outputs = model(input_ids=input_ids, labels=mask_labels)
            loss = outputs.loss
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.4f}")

    # Save the fine-tuned model
    model.save_pretrained("fine_tuned_t5_masked_task")

# Example usage
if __name__ == "__main__":
    train_texts = open('../data/pre-train-data.txt').readlines()
    fine_tune_t5_on_masked_task(train_texts)

