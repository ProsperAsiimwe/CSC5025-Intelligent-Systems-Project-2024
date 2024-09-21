import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.metrics import mean_absolute_error, mean_squared_error
import networkx as nx
import time
from tqdm import tqdm
import random

# Set random seed for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

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

# Split the data into training/validation/test sets for each input window and prediction horizon
results = {}
for window in input_windows:
    results[window] = {}
    for horizon in prediction_horizons:
        X_seq, y_seq = create_sequences(normalized_data, window, horizon)

        # Split into train/validation/test sets
        X_train, X_temp, y_train, y_temp = train_test_split(X_seq, y_seq, test_size=0.4, random_state=42)
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
    def __init__(self, input_size, hidden_size, output_size, num_nodes):
        super(GraphWaveNet, self).__init__()
        self.num_nodes = num_nodes
        self.hidden_size = hidden_size

        # Adaptive adjacency matrix (optional)
        self.adapt_adj = nn.Parameter(torch.eye(num_nodes))

        # Graph convolutional layers
        self.gc1 = GraphConvLayer(input_size, hidden_size)
        self.gc2 = GraphConvLayer(hidden_size, hidden_size)

        # Temporal convolution layers
        self.temporal_conv = nn.Conv1d(in_channels=hidden_size,
                                        out_channels=hidden_size,
                                        kernel_size=3,
                                        padding=1)

        # Fully connected layer for output
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        adj = self.adapt_adj  # Use adaptive adjacency matrix

        # Graph convolution layers with ReLU activation
        x = self.gc1(x, adj)
        x = torch.relu(x)
        x = self.gc2(x, adj)
        x = torch.relu(x)

        # Temporal convolution layer processing
        x = x.transpose(1, 2)  # Prepare for temporal convolution
        x = self.temporal_conv(x)
        x = torch.relu(x)
        x = x.transpose(1, 2)

        # Final output layer using last time step's output
        out = self.fc(x[:, -1, :])
        return out

# Function to compute MAPE (Mean Absolute Percentage Error)
def MAPE(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Continual Learning Class Definition
class ContinualLearningModel:
    def __init__(self):
        self.model = None
        self.optimizer = None
        
    def initialize_model(self, input_size, hidden_size, output_size, num_nodes):
        """Initialize model and optimizer."""
        self.model = GraphWaveNet(input_size=input_size,
                                   hidden_size=hidden_size,
                                   output_size=output_size,
                                   num_nodes=num_nodes).to(device)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def update_model(self, new_data_tensor, new_labels_tensor):
        """Update the model with new data."""
        
        self.model.train()
        
        # Zero gradients and perform forward pass with new data
        self.optimizer.zero_grad()
        
        outputs = self.model(new_data_tensor)  # Forward pass through the model
        
        loss_fn = nn.MSELoss()
        
        loss = loss_fn(outputs.squeeze(), new_labels_tensor)  # Compute loss
        
        loss.backward()  # Backpropagation
        
        self.optimizer.step()  # Update weights

    def evaluate(self, test_data_tensor, test_labels_tensor):
        """Evaluate the model on test data."""
        
        self.model.eval()
        
        with torch.no_grad():
            test_outputs = self.model(test_data_tensor.to(device))
            test_loss_fn = nn.MSELoss()
            test_loss = test_loss_fn(test_outputs.squeeze(), test_labels_tensor.to(device))
            test_mae = mean_absolute_error(test_labels_tensor.cpu().numpy(), test_outputs.cpu().numpy())
            test_rmse = np.sqrt(mean_squared_error(test_labels_tensor.cpu().numpy(), test_outputs.cpu().numpy()))
            return test_loss.item(), test_mae, test_rmse

# Hyperparameter grid for tuning model parameters 
param_grid = {
    'hidden_size': [32],
    'num_epochs': [100], 
    'learning_rate': [0.001]
}

# Define the number of runs for stability analysis 
num_runs = 3

# Perform grid search across different input windows and prediction horizons 
for window in input_windows:
    print(f"\nInput Window: {window}")
    
    for horizon in prediction_horizons:
        
       print(f"Prediction Horizon: {horizon} days")
       
       best_loss=float('inf')
       best_params={}
       
       all_losses=[] 

       for params in ParameterGrid(param_grid):
           print(f"Testing parameters: {params}")
           run_losses=[] 

           continual_model=ContinualLearningModel()
           continual_model.initialize_model(input_size=results[window][horizon]['X_train'].shape[2],
                                            hidden_size=params['hidden_size'],
                                            output_size=results[window][horizon]['y_train'].shape[1],
                                            num_nodes=results[window][horizon]['X_train'].shape[1])

           for run in range(num_runs):
               start_time=time.time()

               # Training loop with tqdm progress bar 
               for epoch in tqdm(range(params['num_epochs']), desc="Training Epochs"):
                   continual_model.update_model(results[window][horizon]['X_train'], results[window][horizon]['y_train'])

               # Evaluate on the validation set 
               val_loss,val_mae,val_rmse=continual_model.evaluate(results[window][horizon]['X_val'], results[window][horizon]['y_val'])
               run_losses.append(val_loss) 

               training_time=time.time()-start_time 
               print(f'Run {run + 1} - Validation Loss: {val_loss:.4f}, Training Time: {training_time:.2f} seconds')

           avg_loss=np.mean(run_losses) 
           all_losses.append(avg_loss) 

           if avg_loss < best_loss:
               best_loss=avg_loss 
               best_params=params 

           print(f'Best parameters for window {window} and horizon {horizon}: {best_params} with average validation loss: {best_loss:.4f}')

# Final evaluation on the test set with the best parameters 
final_model=GraphWaveNet(input_size=results[window][horizon]['X_train'].shape[2],
                          hidden_size=best_params['hidden_size'],
                          output_size=results[window][horizon]['y_train'].shape[1],
                          num_nodes=results[window][horizon]['X_train'].shape[1]).to(device)

optimizer_final=torch.optim.Adam(final_model.parameters(), lr=best_params['learning_rate'])

criterion_final = nn.MSELoss()  # Define criterion here

for epoch in tqdm(range(best_params['num_epochs']), desc="Final Training"):
     final_model.train()
     optimizer_final.zero_grad()
     outputs=final_model(results[window][horizon]['X_train'])
     loss=criterion_final(outputs ,results[window][horizon]['y_train'])
     loss.backward()
     optimizer_final.step()

# Evaluate on the test set 
final_model.eval()

with torch.no_grad():
     test_outputs=final_model(results[window][horizon]['X_test'])
     test_loss=criterion_final(test_outputs ,results[window][horizon]['y_test'])

     test_outputs_inverse=scaler.inverse_transform(test_outputs.cpu().numpy())
     test_y_inverse=scaler.inverse_transform(results[window][horizon]['y_test'].cpu().numpy())

     test_mae=mean_absolute_error(test_y_inverse,test_outputs_inverse)
     test_rmse=np.sqrt(mean_squared_error(test_y_inverse,test_outputs_inverse))
     test_mape=MAPE(test_y_inverse,test_outputs_inverse)

print(f'Test Loss (MSE): {test_loss.item():.4f}')
print(f'Test MAE: {test_mae:.4f}')
print(f'Test RMSE: {test_rmse:.4f}')
print(f'Test MAPE: {test_mape:.2f}%')