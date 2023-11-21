import numpy as np
from tensorflow import Tensor
from torch import LongTensor
from transformers import AutoTokenizer, RobertaForSequenceClassification, get_linear_schedule_with_warmup, \
    RobertaTokenizer
import torch
import pandas as pd
import torch
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, TensorDataset, random_split
from transformers.modeling_outputs import SequenceClassifierOutput

tokenizer = RobertaTokenizer.from_pretrained('DeepChem/ChemBERTa-77M-MLM', from_pt=True)

print("Loading Datasets")
df = pd.concat(map(pd.read_csv, ['Testing_Smiles.csv', 'Training_Smiles.csv']))
smiles = df['Smiles'].tolist()
labels = df['Property'].tolist()

encodings = tokenizer(smiles, padding=True, truncation=True, return_tensors='pt')
labels = torch.tensor(labels)

print("Loading Pre-Trained Model")

# O=C(O)c1cc(N=Nc2ccc(S(=O)(=O)Nc3ccccn3)cc2)ccc1O


model: RobertaForSequenceClassification = RobertaForSequenceClassification.from_pretrained(
    'DeepChem/ChemBERTa-77M-MLM',
    num_labels=2, # The number of output labels--2 for binary classification.
    output_attentions=False,
    output_hidden_states=True,
)


optimizer = AdamW(model.parameters(), lr=2e-5)

dataset = TensorDataset(encodings['input_ids'], encodings['attention_mask'], labels)

train_size = int(0.7 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size


print("Len of training, val, test:", train_size, val_size, test_size)
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

batch_size = 8

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

num_epochs = 20

total_steps = len(train_dataloader) * num_epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

print("Training")


for epoch in range(num_epochs):
    model.train()
    for batch in train_dataloader:
        optimizer.zero_grad()
        inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[2]}
        outputs = model(**inputs)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()

    # Validation
    model.eval()
    accuracy = 0
    for batch in val_dataloader:
        with torch.no_grad():
            inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[2]}
            outputs: SequenceClassifierOutput = model(**inputs)

            # Evaluate your validation metrics here
            predictions = np.argmax(outputs.logits, axis=1)
            for pred, real in zip(predictions, batch[2]):
                if pred == real:
                    accuracy += 1
    print(epoch, accuracy/val_size)

# Save the fine-tuned model
model.save_pretrained("fine_tuned_model")
