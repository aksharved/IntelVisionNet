import torch
import torch.optim as optim
import torch.nn as nn
from src.model import IntelCNN
from src.data_loader import load_data

# Hyperparameters
batch_size = 32
learning_rate = 0.001
num_epochs = 10
data_dir = "pathtointeldata"  # Update with the correct path

# Load data
train_loader, test_loader = load_data(batch_size, data_dir)

# Model, Loss, Optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = IntelCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}, Accuracy: {100 * correct / total}%")
    
# Save model
torch.save(model.state_dict(), 'models\custom_cnn_model_weights_updated.pth')
