import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ignore FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

df = pd.read_csv('../Project/creditcard.csv')
#df = pd.read_csv('creditcard.csv')
df.head()


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

X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# train set and test set shapes
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

# Define the LSTM model:

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
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()

        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.dropout(out[:, -1, :])  # Apply dropout after the LSTM layer
        out = self.fc1(out)  # Pass the output through the first fully connected layer
        out = torch.relu(out)  # Apply ReLU activation function
        out = self.fc2(out)  # Pass the output through the second fully connected layer
        return out


# Instantiate the model, loss function, and optimizer:

model = LSTMModel(input_size=30, hidden_size=50, num_layers=1, output_size=1)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Train the model:

num_epochs = 10

for epoch in range(num_epochs):
    for i, (features, targets) in enumerate(train_loader):
        features = features.unsqueeze(1)
        targets = targets

        # Forward pass
        logits = model(features)
        loss = criterion(logits, targets)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


# Evaluate the model:

model.eval()
targets_list = []
predicted_list = []

with torch.no_grad():
    for features, targets in test_loader:
        features = features.unsqueeze(1)
        targets = targets
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