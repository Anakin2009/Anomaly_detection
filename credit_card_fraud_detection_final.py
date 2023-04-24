

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import random
import seaborn as sns
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve, auc

if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True

# ignore FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

if torch.cuda.is_available():
    DEVICE = "cuda:0"
else:
    DEVICE = "cpu"

#df = pd.read_csv('../Project/creditcard.csv')
#df.head()

# Google Colab

from google.colab import drive
drive.mount("/content/drive/")

proj_path = '/content/drive/MyDrive/Colab Notebooks/'
df = pd.read_csv(proj_path +'creditcard.csv')

df.head()

df1 = df[df['Class'] == 1]
sns.kdeplot(df1['Time'], shade=True)
sns.kdeplot(df['Time'], shade=True)
plt.title('Density plot of Time')

fig, axs = plt.subplots(ncols=2, figsize=(15, 5))

# plot the first density plot on the first subplot
sns.kdeplot(df1['Amount'], shade=True, ax=axs[0])

# plot the second density plot on the second subplot
sns.kdeplot(df['Amount'], shade=True, ax=axs[1])

# set the titles of the subplots
axs[0].set_title('Density plot of Fraud')
axs[1].set_title('Density plot of all data')

# Check the distribution of the target variable ('Class'):

print("Class Distribution:")
print(df['Class'].value_counts())

# Plot the class distribution
sns.countplot(x='Class', data=df)
plt.title("Class Distribution")
plt.show()

# Analyze the distribution of 'Time' and 'Amount' features:

fig, axes = plt.subplots(1, 2, figsize=(18, 4))

sns.distplot(df['Time'], ax=axes[0], color='blue')
axes[0].set_title("Time Distribution")

sns.distplot(df['Amount'], ax=axes[1], color='red')
axes[1].set_title("Amount Distribution")

plt.show()

# Check the correlation between features:

corr_matrix = df.corr()

# Plot the correlation heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, cmap='coolwarm', annot=True, annot_kws={'size':5})
plt.title("Correlation Heatmap")
plt.show()

# Normalize the 'Time' and 'Amount' features:

scaler = StandardScaler()

df['Time'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))
df['Amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))

# Split the dataset into train and test sets:

random.seed(586)
X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train set and test set shapes
print("Train set shape:", X_train.shape)
print("Test set shape:", X_test.shape)

# Convert data to PyTorch tensors

X_train = torch.tensor(X_train.values, dtype=torch.float)
X_test = torch.tensor(X_test.values, dtype=torch.float)
y_train = torch.tensor(y_train.values, dtype=torch.float).unsqueeze(1)
y_test = torch.tensor(y_test.values, dtype=torch.float).unsqueeze(1)

# Create a custom dataset class and data loaders:

class CreditCardDataset(Dataset):
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

train_dataset = CreditCardDataset(X_train, y_train)
test_dataset = CreditCardDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

device = torch.device(DEVICE)

torch.manual_seed(0)

for epoch in range(2):

    for batch_idx, (x, y) in enumerate(train_loader):

        print('Epoch:', epoch+1, end='')
        print(' | Batch index:', batch_idx, end='')
        print(' | Batch size:', y.size()[0])

        x = x.to(device)
        y = y.to(device)
        break

# Define the LSTM base model:

class LSTMModelBase(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModelBase, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()

        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])
        return out

# Define the LSTM experimental model:

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.5)  # Adding a dropout layer
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)  # Adding an additional fully connected layer
        self.fc2 = nn.Linear(hidden_size // 2, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_().to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_().to(x.device)

        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.dropout(out[:, -1, :])  # Apply dropout after the LSTM layer
        out = self.fc1(out)  # Pass the output through the first fully connected layer
        out = torch.relu(out)  # Apply ReLU activation function
        out = self.fc2(out)  # Pass the output through the second fully connected layer
        return out

# Instantiate the model, loss function, and optimizer:
model = LSTMModel(input_size = 30, hidden_size = 70, num_layers = 3, output_size = 1)

model.to(DEVICE)

pos_weight = torch.tensor([1])
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# Define the threshold to use for binary predictions:
threshold = 0.5

# Define the early stopping criteria:
best_f1_score = 0
patience = 5
counter = 0

# Train the model:
random.seed(586)
num_epochs = 30

# Record the start time for measuring the training duration
start_time = time.time()

model.train()

for epoch in range(num_epochs):
    for i, (features, targets) in enumerate(train_loader):
        features = features.unsqueeze(1).to(DEVICE)
        targets = targets.to(DEVICE)

        # Forward pass
        logits = model(features)
        loss = criterion(logits, targets)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Compute the F1 score on the training dataset:
    model.eval()
    with torch.no_grad():
        y_true = []
        y_pred = []
        for features, targets in train_loader:
            features = features.unsqueeze(1)
            targets = targets
            logits = model(features)
            preds = (torch.sigmoid(logits) >= threshold).int()
            y_true += targets.tolist()
            y_pred += preds.tolist()

        f1 = f1_score(y_true, y_pred)

    # Print the loss and F1 score for the current epoch:
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, F1 Score: {f1:.4f}')

    # Check if the current F1 score is better than the best one seen so far:
    if f1 > best_f1_score:
        best_f1_score = f1
        counter = 0
    else:
        counter += 1

    # Stop training if the F1 score has not improved for a certain number of epochs:
    if counter >= patience:
        print(f'Early stopping at epoch {epoch+1}')
        break

# Print the total training time
print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))


# Final evaluation on the training dataset:
model.eval()
with torch.no_grad():
    y_true = []
    y_pred = []
    for features, targets in train_loader:
        features = features.unsqueeze(1).to(DEVICE)
        targets = targets.to(DEVICE)
        logits = model(features)
        preds = (torch.sigmoid(logits) >= threshold).int()
        y_true += targets.tolist()
        y_pred += preds.tolist()

    f1 = f1_score(y_true, y_pred)

print(f'Training F1 Score: {f1:.4f}')

# Evaluate the model(Train):

model.eval()
targets_list = []
predicted_list = []

with torch.no_grad():
    for features, targets in train_loader:
        features = features.unsqueeze(1).to(DEVICE)
        targets = targets.to(DEVICE)
        logits = model(features)
        predicted = torch.sigmoid(logits).round()

        targets_list.extend(targets.cpu().numpy())
        predicted_list.extend(predicted.cpu().numpy())

# Calculate metrics
accuracy = accuracy_score(targets_list, predicted_list)
precision = precision_score(targets_list, predicted_list)
recall = recall_score(targets_list, predicted_list)
f1 = f1_score(targets_list, predicted_list)

print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')

# Evaluate the model(Train):
model.eval()
targets_list = []
predicted_list = []

with torch.no_grad():
    for features, targets in train_loader:
        features = features.unsqueeze(1).to(DEVICE)
        targets = targets.to(DEVICE)
        logits = model(features)
        predicted = torch.sigmoid(logits).cpu().numpy()

        targets_list.extend(targets.cpu().numpy())
        predicted_list.extend(predicted)

# Compute precision and recall values
precision, recall, thresholds = precision_recall_curve(targets_list, predicted_list)

# Compute AUPRC score
auprc = auc(recall, precision)

# Plot the AUPRC curve
plt.plot(recall, precision, label=f'AUPRC={auprc:.3f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()

# Find and plot the optimal threshold
f1_scores = 2 * (precision * recall) / (precision + recall)
optimal_threshold = thresholds[np.argmax(f1_scores)]
#plt.axvline(x=optimal_threshold, color='r', linestyle='--', label=f'Optimal Threshold={optimal_threshold:.3f}')
plt.legend()

plt.show()


# Find the index of the optimal threshold based on F1 score
optimal_idx = np.argmax(f1_scores)

# Find the optimal threshold and print the corresponding precision and recall
optimal_threshold = thresholds[optimal_idx]
optimal_precision = precision[optimal_idx]
optimal_recall = recall[optimal_idx]

print(f'Threshold: {optimal_threshold:.4f}')
print(f'Precision: {optimal_precision:.4f}')
print(f'Recall: {optimal_recall:.4f}')
print(f'F1_scores : {2 * (optimal_precision * optimal_recall) / (optimal_precision + optimal_recall) :.4f}')

# Evaluate the model(Testing):

model.eval()
targets_list = []
predicted_list = []

with torch.no_grad():
    for features, targets in test_loader:
        features = features.unsqueeze(1).to(DEVICE)
        targets = targets.to(DEVICE)
        logits = model(features)
        predicted = torch.sigmoid(logits).round()

        targets_list.extend(targets.cpu().numpy())
        predicted_list.extend(predicted.cpu().numpy())

# Calculate metrics
accuracy_experimental = accuracy_score(targets_list, predicted_list)
precision_experimental = precision_score(targets_list, predicted_list)
recall_experimental = recall_score(targets_list, predicted_list)
f1_experimental = f1_score(targets_list, predicted_list)

print(f'Accuracy: {accuracy_experimental:.4f}')
print(f'Precision: {precision_experimental:.4f}')
print(f'Recall: {recall_experimental:.4f}')
print(f'F1 Score: {f1_experimental:.4f}')

# Evaluate the model(Testing):

model.eval()
targets_list = []
predicted_list = []

with torch.no_grad():
    for features, targets in test_loader:
        features = features.unsqueeze(1).to(DEVICE)
        targets = targets.to(DEVICE)
        logits = model(features)
        predicted = torch.sigmoid(logits).cpu().numpy()

        targets_list.extend(targets.cpu().numpy())
        predicted_list.extend(predicted)

# Compute precision and recall values
precision, recall, thresholds = precision_recall_curve(targets_list, predicted_list)

# Compute AUPRC score
auprc = auc(recall, precision)

# Plot the AUPRC curve
import matplotlib.pyplot as plt
plt.plot(recall, precision, label=f'AUPRC={auprc:.3f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()

# Find and plot the optimal threshold
f1_scores = 2 * (precision * recall) / (precision + recall)
optimal_threshold = thresholds[np.argmax(f1_scores)]
#plt.axvline(x=optimal_threshold, color='r', linestyle='--', label=f'Optimal Threshold={optimal_threshold:.3f}')
plt.legend()

plt.show()


# Find the index of the optimal threshold based on F1 score
optimal_idx = np.argmax(f1_scores)

# Find the optimal threshold and print the corresponding precision and recall
optimal_threshold = thresholds[optimal_idx]
optimal_precision = precision[optimal_idx]
optimal_recall = recall[optimal_idx]

print(f'Threshold: {optimal_threshold:.4f}')
print(f'Precision: {optimal_precision:.4f}')
print(f'Recall: {optimal_recall:.4f}')
print(f'F1_scores : {2 * (optimal_precision * optimal_recall) / (optimal_precision + optimal_recall) :.4f}')

# LSTM Base Model

# Instantiate the model, loss function, and optimizer:
modelBase = LSTMModelBase(input_size = 30, hidden_size = 50, num_layers = 1, output_size = 1)

modelBase.to(DEVICE)

criterionBase = nn.BCEWithLogitsLoss()

optimizerBase = optim.Adam(modelBase.parameters(), lr=0.001)


# Train the model:

# Record the start time for measuring the training duration
start_timeBase = time.time()

num_epochsBase = 10

for epoch in range(num_epochsBase):
    for i, (features, targets) in enumerate(train_loader):
        featuresBase = features.unsqueeze(1).to(DEVICE)
        targetsBase = targets.to(DEVICE)

        # Forward pass
        logitsBase = modelBase(featuresBase)
        lossBase = criterion(logitsBase, targetsBase)

        # Backward and optimize
        optimizerBase.zero_grad()
        lossBase.backward()
        optimizerBase.step()

    print(f'Epoch [{epoch+1}/{num_epochsBase}], Loss: {lossBase.item():.4f}')


# Print the total training time
print('Total Training Time: %.2f min' % ((time.time() - start_timeBase)/60))


# Evaluate the model:

modelBase.eval()
targets_listBase = []
predicted_listBase = []

with torch.no_grad():
    for features, targets in test_loader:
        featuresBase = features.unsqueeze(1).to(DEVICE)
        targetsBase = targets.to(DEVICE)
        logitsBase = modelBase(featuresBase)
        predictedBase = torch.sigmoid(logitsBase).round()

        targets_listBase.extend(targetsBase.cpu().numpy())
        predicted_listBase.extend(predictedBase.cpu().numpy())

# Calculate metrics
accuracy_Base = accuracy_score(targets_listBase, predicted_listBase)
precision_Base = precision_score(targets_listBase, predicted_listBase)
recall_Base = recall_score(targets_listBase, predicted_listBase)
f1_Base = f1_score(targets_listBase, predicted_listBase)

print(f'Accuracy: {accuracy_Base:.4f}')
print(f'Precision: {precision_Base:.4f}')
print(f'Recall: {recall_Base:.4f}')
print(f'F1 Score: {f1_Base:.4f}')

# Compare the LSTM base model with the LSTM experimental model.

import pandas as pd

# Combine the results into a dictionary
results = {
    'Model': ['LSTM Base', 'LSTM Expe'],
    'Accuracy': [accuracy_Base, accuracy_experimental],
    'Precision': [precision_Base, precision_experimental],
    'Recall': [recall_Base, recall_experimental],
    'F1 Score': [f1_Base, f1_experimental]
}

# Create a pandas DataFrame from the dictionary
results_df = pd.DataFrame(results)

# Display the table
print(results_df)