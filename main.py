import numpy as np
from tensorflow import Tensor
from torch import LongTensor, FloatTensor
from transformers import AutoTokenizer, RobertaForSequenceClassification, get_linear_schedule_with_warmup, \
    RobertaTokenizer
import matplotlib.pyplot as plt
import torch
import pandas as pd
import torch
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, TensorDataset, random_split
from transformers.modeling_outputs import SequenceClassifierOutput

tokenizer = RobertaTokenizer.from_pretrained('DeepChem/ChemBERTa-77M-MLM', from_pt=True)

print("Loading Datasets")
df = pd.concat(map(pd.read_csv, ['Training_Smiles.csv']))
# df = pd.concat(map(pd.read_csv, ['Testing_Smiles.csv', 'Training_Smiles.csv']))
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

batch_size = 32

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

num_epochs = 20

total_steps = len(train_dataloader) * num_epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

print("Training")

accuracies = []
losses = []

for epoch in range(num_epochs):
    model.train()
    mean_loss = 0
    for batch in train_dataloader:
        optimizer.zero_grad()
        inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[2]}
        outputs = model(**inputs)
        loss: FloatTensor = outputs.loss
        mean_loss += loss.detach().numpy()
        loss.backward()
        optimizer.step()
        scheduler.step()
    losses.append(mean_loss / len(train_dataloader))

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
    print("VALIDATION ACCURACY:", epoch, accuracy/val_size)
    accuracies.append(accuracy/val_size)

    accuracy = 0
    for batch in test_dataloader:
        with torch.no_grad():
            inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[2]}
            outputs: SequenceClassifierOutput = model(**inputs)

            # Evaluate your validation metrics here
            predictions = np.argmax(outputs.logits, axis=1)
            for pred, real in zip(predictions, batch[2]):
                if pred == real:
                    accuracy += 1
    print("TEST ACCURACY:", epoch, accuracy/test_size)
    accuracies.append(accuracy/test_size)

# Save the fine-tuned model
model.save_pretrained("fine_tuned_model")

acc_plot = plt.figure()
plt.xlabel("Epochs")
plt.ylabel("Accuracies")
plt.plot(range(num_epochs), accuracies[0::2])
plt.plot(range(num_epochs), accuracies[1::2])
plt.plot(range(num_epochs), losses)
plt.savefig("Training")
plt.close()

output_rec = [[], []]

accuracy = 0
for batch in test_dataloader:
    with torch.no_grad():
        inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[2]}
        outputs: SequenceClassifierOutput = model(**inputs)

        # Evaluate your validation metrics here
        list_output = outputs.logits.detach().numpy().tolist()
        label_ = batch[2].detach().numpy()
        for label, out in zip(label_, list_output):
            output_rec[label].append(out)

        predictions = np.argmax(outputs.logits, axis=1)
        for pred, real in zip(predictions, batch[2]):
            if pred == real:
                accuracy += 1

outputs_plot = plt.figure()
plt.scatter([x[0] for x in output_rec[0]], [x[1] for x in output_rec[0]], marker='X')
plt.scatter([x[0] for x in output_rec[1]], [x[1] for x in output_rec[1]], marker='.')
plt.xlabel("X1")
plt.ylabel("X2")
plt.savefig("OUTPUTS")
print(accuracy/test_size)
