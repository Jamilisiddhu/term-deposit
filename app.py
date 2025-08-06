from flask import Flask, request, jsonify, render_template # Added render_template
from flask_cors import CORS
import pickle
from collections import Counter
import math
import csv
import numpy as np

# Your custom functions from the original project
def gini_index(groups, classes):
    """Calculates the Gini index for a list of groups."""
    total = sum(len(group) for group in groups)
    gini = 0.0
    for group in groups:
        size = len(group)
        if size == 0:
            continue
        score = 0.0
        counts = Counter(group)
        for c in classes:
            p = counts.get(c, 0) / size
            score += p * p
        gini += (1.0 - score) * (size / total)
    return gini

def get_best_split(X, y):
    """Finds the best split point for a dataset."""
    best_index, best_value, best_score, best_groups = None, None, float('inf'), None
    classes = list(set(y))
    
    for index in range(len(X[0])):
        values = set([row[index] for row in X])
        for value in values:
            left_y = [y[i] for i in range(len(X)) if X[i][index] == value]
            right_y = [y[i] for i in range(len(X)) if X[i][index] != value]
            
            gini = gini_index([left_y, right_y], classes)
            
            if gini < best_score:
                best_score = gini
                best_index = index
                best_value = value
                
                left_X = [X[i] for i in range(len(X)) if X[i][index] == value]
                right_X = [X[i] for i in range(len(X)) if X[i][index] != value]
                
                best_groups = (left_X, left_y, right_X, right_y)
                
    return {'index': best_index, 'value': best_value, 'groups': best_groups}

def build_cart(X, y, max_depth, depth=0):
    """Builds a CART decision tree recursively."""
    if len(set(y)) == 1 or depth >= max_depth:
        return {'label': Counter(y).most_common(1)[0][0]}
    
    node = get_best_split(X, y)
    
    left_X, left_y, right_X, right_y = node['groups']
    
    node['left'] = build_cart(left_X, left_y, max_depth, depth + 1)
    node['right'] = build_cart(right_X, right_y, max_depth, depth + 1)
    
    return node

def predict_cart(tree, x):
    """
    Predicts the class of a single data point using the CART decision tree.
    This function is based on your project's logic.
    """
    while 'label' not in tree:
        feature_index = tree['index']
        value = tree['value']
        
        if x[feature_index] == value:
            tree = tree['left']
        else:
            tree = tree['right']
    return tree['label']

# Load the trained CART model and encoders
try:
    with open('cart_model.pkl', 'rb') as file:
        cart_tree = pickle.load(file)
    with open('encoders.pkl', 'rb') as file:
        encoders = pickle.load(file)
    print("Model and encoders loaded successfully.")
except FileNotFoundError:
    print("Error: Model or encoders file not found. Please ensure they are in the same directory.")
    cart_tree = None
    encoders = None

# Initialize the Flask application and enable CORS
app = Flask(__name__)
CORS(app)

# New route to serve the index.html file
@app.route('/')
def serve_index():
    return render_template('index.html')

# Define the API endpoint for predictions
@app.route('/predict', methods=['POST'])
def predict():
    if cart_tree is None or encoders is None:
        return jsonify({'error': 'Model not loaded'}), 500

    data = request.json
    
    header = [
        'age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 
        'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 
        'previous', 'poutcome'
    ]
    encoded_input = []
    
    for feature_name in header:
        input_value = data.get(feature_name)
        
        # Handle categorical features
        if feature_name in encoders and encoders[feature_name] is not None:
            mapping = encoders[feature_name]
            encoded_value = mapping.get(input_value)
            if encoded_value is None:
                return jsonify({'error': f"Unknown value '{input_value}' for feature '{feature_name}'"}), 400
            encoded_input.append(encoded_value)
        # Handle numerical features
        else:
            try:
                encoded_input.append(float(input_value))
            except (ValueError, TypeError):
                return jsonify({'error': f"Invalid numerical value '{input_value}' for feature '{feature_name}'"}), 400

    # Make a prediction with the CART model
    prediction_result = predict_cart(cart_tree, encoded_input)
    
    # Decode the numerical prediction back to 'yes' or 'no'
    label_decoder = {v: k for k, v in encoders['y'].items()}
    decoded_prediction = label_decoder.get(prediction_result, 'Unknown')

    # Return the prediction to the web page
    return jsonify({'prediction': decoded_prediction})

if __name__ == '__main__':
    app.run(debug=True)