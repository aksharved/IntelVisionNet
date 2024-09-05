import torch
from src.model import IntelCNN
from src.data_loader import load_data

# Load model and test data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = IntelCNN().to(device)
model.load_state_dict(torch.load('models\custom_cnn_model_weights_updated.pth'))

batch_size = 32
data_dir = "pathtointeldata" 
_, test_loader = load_data(batch_size, data_dir)

# Test the model
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

print(f'Test Accuracy: {100 * correct / total}%')
