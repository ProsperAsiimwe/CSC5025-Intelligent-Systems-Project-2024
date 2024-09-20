import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, ParameterGrid
import networkx as nx
import time
from tqdm import tqdm

# Use GPU if available, otherwise default to CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load the dataset
data = pd.read_csv('JSE_clean_truncated.csv')

# Data Cleaning: Check for missing values
if data.isnull().values.any():
    data.fillna(method='ffill', inplace=True)  # Forward fill for simplicity

# Normalize the data
scaler = StandardScaler()
normalized_data = scaler.fit_transform(data)

# Create sequences for time series forecasting based on prediction horizons
def create_sequences(data, seq_length, horizon):
    xs, ys = [], []
    for i in range(len(data) - seq_length - horizon + 1):
        x = data[i:i + seq_length]
        y = data[i + seq_length + horizon - 1]  # Target is the value at horizon days later
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# Parameters for different input windows and prediction horizons
input_windows = [30, 60, 120]
prediction_horizons = [1, 2, 5, 10, 30]

# Split the data into training, validation, and test sets for each input window and prediction horizon
results = {}

for window in input_windows:
    results[window] = {}
    for horizon in prediction_horizons:
        X, y = create_sequences(normalized_data, window, horizon)
        
        # Split into train/validation/test sets
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

        # Convert to PyTorch tensors and move to device
        results[window][horizon] = {
            'X_train': torch.FloatTensor(X_train).to(device),
            'y_train': torch.FloatTensor(y_train).to(device),
            'X_val': torch.FloatTensor(X_val).to(device),
            'y_val': torch.FloatTensor(y_val).to(device),
            'X_test': torch.FloatTensor(X_test).to(device),
            'y_test': torch.FloatTensor(y_test).to(device)
        }

# Graph Construction (using correlation matrix)
correlation_matrix = np.corrcoef(data.T)
graph = nx.from_numpy_array(correlation_matrix)

# Create adjacency matrix from correlation matrix
adj_matrix = torch.FloatTensor(nx.to_numpy_matrix(graph)).to(device)

# Define the Graph Convolutional Layer
class GraphConvLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvLayer, self).__init__()
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        out = torch.matmul(adj, x)
        out = self.fc(out)
        return out

# Define the GraphWaveNet model
class GraphWaveNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_nodes, addaptadj=True):
        super(GraphWaveNet, self).__init__()
        self.num_nodes = num_nodes
        self.hidden_size = hidden_size
        
        # Adaptive adjacency matrix
        if addaptadj:
            self.adapt_adj = nn.Parameter(torch.eye(num_nodes))  # Learnable adjacency matrix

        # Graph convolutional layers
        self.gc1 = GraphConvLayer(input_size, hidden_size)
        self.gc2 = GraphConvLayer(hidden_size, hidden_size)

        # Temporal convolution layers
        self.temporal_conv = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=3, padding=1)

        # Fully connected layer for output
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, adj):
        # Adjust adjacency matrix if adaptive
        if hasattr(self, 'adapt_adj'):
            adj = self.adapt_adj
        
        # Graph convolution
        x = self.gc1(x, adj)
        x = torch.relu(x)
        x = self.gc2(x, adj)
        x = torch.relu(x)

        # Temporal convolution
        x = x.transpose(1, 2)  # Prepare for temporal convolution
        x = self.temporal_conv(x)
        x = torch.relu(x)
        x = x.transpose(1, 2)

        # Final output layer
        out = self.fc(x[:, -1, :])  # Last time step
        return out

# Hyperparameter grid
param_grid = {
    'hidden_size': [32, 64, 128],
    'num_epochs': [100],  # Increased to allow for better convergence
    'learning_rate': [0.001]
}

# Perform grid search across different input windows and prediction horizons
for window in input_windows:
    print(f"\nInput Window: {window}")
    
    for horizon in prediction_horizons:
        print(f"Prediction Horizon: {horizon} days")
        
        best_loss = float('inf')
        best_params = {}
        
        for params in ParameterGrid(param_grid):
            print(f"Testing parameters: {params}")
            
            # Initialize the model and optimizer
            model = GraphWaveNet(input_size=results[window][horizon]['X_train'].shape[2],
                                  hidden_size=params['hidden_size'],
                                  output_size=results[window][horizon]['y_train'].shape[1],
                                  num_nodes=results[window][horizon]['X_train'].shape[1]).to(device)
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
            
            # Training the model with timing and loss tracking using tqdm for progress reporting
            start_time = time.time()
            
            for epoch in tqdm(range(params['num_epochs']), desc="Training Epochs"):
                model.train()
                optimizer.zero_grad()
                outputs = model(results[window][horizon]['X_train'], adj_matrix)
                loss = criterion(outputs, results[window][horizon]['y_train'])
                loss.backward()
                optimizer.step()
            
            training_time = time.time() - start_time
            
            # Evaluation on the validation set
            model.eval()
            with torch.no_grad():
                val_outputs = model(results[window][horizon]['X_val'], adj_matrix)
                val_loss = criterion(val_outputs, results[window][horizon]['y_val'])
                print(f'Validation Loss: {val_loss.item():.4f}, Training Time: {training_time:.2f} seconds')

                # Check if this is the best loss
                if val_loss.item() < best_loss:
                    best_loss = val_loss.item()
                    best_params = params

        print(f'Best parameters for window {window} and horizon {horizon}: {best_params} with validation loss: {best_loss:.4f}')

        # Final evaluation on the test set with the best parameters
        model = GraphWaveNet(input_size=results[window][horizon]['X_train'].shape[2],
                              hidden_size=best_params['hidden_size'],
                              output_size=results[window][horizon]['y_train'].shape[1],
                              num_nodes=results[window][horizon]['X_train'].shape[1]).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=best_params['learning_rate'])

        # Train the model again with the best parameters
        for epoch in tqdm(range(best_params['num_epochs']), desc="Final Training"):
            model.train()
            optimizer.zero_grad()
            outputs = model(results[window][horizon]['X_train'], adj_matrix)
            loss = criterion(outputs, results[window][horizon]['y_train'])
            loss.backward()
            optimizer.step()

        # Evaluate on the test set
        model.eval()
        with torch.no_grad():
            test_outputs = model(results[window][horizon]['X_test'], adj_matrix)
            test_loss = criterion(test_outputs, results[window][horizon]['y_test'])
            
            print(f'Test Loss: {test_loss.item():.4f}')

# Inverse transform to get actual prices (for final evaluation)
final_outputs_inverse = scaler.inverse_transform(test_outputs.cpu().numpy())
final_y_test_inverse = scaler.inverse_transform(results[input_windows[-1]][prediction_horizons[-1]]['y_test'].cpu().numpy())
