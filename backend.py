from flask import Flask, request, jsonify, render_template
import torch
from torchvision import transforms
from PIL import Image
import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 64 * 5 * 5)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x


app = Flask(__name__)
@app.route('/')
def home():
    return render_template('index.html')
# Load the model
model = SimpleCNN()  # Define your model architecture
model.load_state_dict(torch.load('mnist_cnn.pth'))
model.eval()

# Preprocessing for the image
preprocess = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    img = Image.open(file).convert('L')  # Convert to grayscale
    img = preprocess(img).unsqueeze(0)

    with torch.no_grad():
        output = model(img)
        _, predicted = torch.max(output, 1)
        prediction = predicted.item()

    return jsonify({'prediction': prediction})



if __name__ == '__main__':
    app.run()
