from flask import Flask, request, jsonify
import torch
import pickle

app = Flask(__name__)

# Load the trained model
with open('mnist_cnn_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Set the model to evaluation mode
model.eval()

# Define a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the request
    input_data = request.json['data']

    # Perform inference using the model
    with torch.no_grad():
        output = model(torch.tensor(input_data).view(1, 1, 28, 28).float())
        predicted_class = torch.argmax(output).item()

    # Return the prediction as JSON
    return jsonify({'predicted_class': predicted_class})

if __name__ == '__main__':
    app.run(debug=True)
