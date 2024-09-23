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
import matplotlib.pyplot as plt
import os
import seaborn as sns

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

# Define the GraphWaveNet model, includes a dropout
class GraphWaveNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_nodes, dropout_rate=0.2):
        super(GraphWaveNet, self).__init__()
        self.num_nodes = num_nodes
        self.hidden_size = hidden_size

        # Adaptive adjacency matrix
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

        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        adj = self.adapt_adj

        # Graph convolution layers with ReLU activation and dropout
        x = self.gc1(x, adj)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.gc2(x, adj)
        x = torch.relu(x)
        x = self.dropout(x)

        # Temporal convolution layer processing
        x = x.transpose(1, 2)
        x = self.temporal_conv(x)
        x = torch.relu(x)
        x = x.transpose(1, 2)
        x = self.dropout(x)

        # Final output layer using last time step's output
        out = self.fc(x[:, -1, :])
        return out

# Function to compute MAPE (Mean Absolute Percentage Error)
def MAPE(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Continual Learning Class Definition, includes early stopping and adaptive learning rate
class ContinualLearningModel:
    def __init__(self, patience=10, min_delta=1e-4):
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        
    def initialize_model(self, input_size, hidden_size, output_size, num_nodes, learning_rate=0.001, dropout_rate=0.2):
        self.model = GraphWaveNet(input_size=input_size,
                                  hidden_size=hidden_size,
                                  output_size=output_size,
                                  num_nodes=num_nodes,
                                  dropout_rate=dropout_rate).to(device)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    def update_model(self, train_data_tensor, train_labels_tensor, val_data_tensor, val_labels_tensor):
        self.model.train()
        
        self.optimizer.zero_grad()
        
        outputs = self.model(train_data_tensor)
        
        loss_fn = nn.MSELoss()
        
        loss = loss_fn(outputs.squeeze(), train_labels_tensor)
        
        loss.backward()
        
        self.optimizer.step()

        # Validation step for early stopping and learning rate scheduling
        val_loss = self.validate(val_data_tensor, val_labels_tensor)
        self.scheduler.step(val_loss)

        # Early stopping check
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1

        return loss.item(), val_loss, self.counter >= self.patience

    def validate(self, val_data_tensor, val_labels_tensor):
        self.model.eval()
        with torch.no_grad():
            val_outputs = self.model(val_data_tensor)
            val_loss_fn = nn.MSELoss()
            val_loss = val_loss_fn(val_outputs.squeeze(), val_labels_tensor)
        return val_loss.item()

    def evaluate(self, test_data_tensor, test_labels_tensor):
        self.model.eval()
        
        with torch.no_grad():
            test_outputs = self.model(test_data_tensor.to(device))
            test_loss_fn = nn.MSELoss()
            test_loss = test_loss_fn(test_outputs.squeeze(), test_labels_tensor.to(device))
            test_mae = mean_absolute_error(test_labels_tensor.cpu().numpy(), test_outputs.cpu().numpy())
            test_rmse = np.sqrt(mean_squared_error(test_labels_tensor.cpu().numpy(), test_outputs.cpu().numpy()))
            return test_loss.item(), test_mae, test_rmse
        

# Hyperparameter grid for tuning model parameters 
# param_grid = {
#     'hidden_size': [64, 128],
#     'num_epochs': [100],
#     'learning_rate': [0.001, 0.0005],
#     'dropout_rate': [0.1, 0.2, 0.3]
# }

param_grid = {
    'hidden_size': [128],
    'num_epochs': [100],
    'learning_rate': [0.001],
    'dropout_rate': [0.2]
}

# Visualization function for performance metrics
def plot_metrics(metrics_dict, metric_name, input_windows, prediction_horizons, save_dir='plots'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)  # Create directory if it doesn't exist

    plt.figure(figsize=(12, 6))
    for window in input_windows:
        for horizon in prediction_horizons:
            values = metrics_dict[window][horizon]
            plt.plot(values, label=f'Window: {window}, Horizon: {horizon}')
    
    plt.xlabel('Run')
    plt.ylabel(metric_name)
    plt.title(f'{metric_name} across Runs for Different Windows and Horizons')
    plt.legend()
    
    # Save plot as a .png file
    plot_filename = f"{metric_name.replace(' ', '_').lower()}_windows_{window}_horizons_{horizon}.png"
    plt.savefig(os.path.join(save_dir, plot_filename))
    plt.close()

# Add a new function for box plot visualization
def plot_boxplots(metrics_dict, metric_name, input_windows, prediction_horizons, save_dir='box_plots'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    data = []
    labels = []
    for window in input_windows:
        for horizon in prediction_horizons:
            data.extend(metrics_dict[window][horizon])
            labels.extend([f'W{window}_H{horizon}'] * len(metrics_dict[window][horizon]))

    plt.figure(figsize=(12, 6))
    sns.boxplot(x=labels, y=data)
    plt.xlabel('Window_Horizon')
    plt.ylabel(metric_name)
    plt.title(f'{metric_name} Distribution for Different Windows and Horizons')
    plt.xticks(rotation=45)
    
    plot_filename = f"{metric_name.replace(' ', '_').lower()}_boxplot.png"
    plt.savefig(os.path.join(save_dir, plot_filename), bbox_inches='tight')
    plt.close()

# Initialize dictionaries to store losses and metrics for each run
train_losses = {window: {horizon: [] for horizon in prediction_horizons} for window in input_windows}
val_losses = {window: {horizon: [] for horizon in prediction_horizons} for window in input_windows}
test_maes = {window: {horizon: [] for horizon in prediction_horizons} for window in input_windows}
test_rmses = {window: {horizon: [] for horizon in prediction_horizons} for window in input_windows}

# Define the number of runs for stability analysis 
num_runs = 3

# Initialize a dictionary to store the best model for each window and horizon
best_models = {window: {horizon: None for horizon in prediction_horizons} for window in input_windows}
best_losses = {window: {horizon: float('inf') for horizon in prediction_horizons} for window in input_windows}

# Perform grid search across different input windows and prediction horizons, has incorporated early stopping
for window in input_windows:
    print(f"\nInput Window: {window}")
    
    for horizon in prediction_horizons:
        print(f"Prediction Horizon: {horizon} days")
        
        best_loss = float('inf')
        best_params = {}
        
        for params in ParameterGrid(param_grid):
            print(f"Testing parameters: {params}")
            run_losses = []

            for run in range(num_runs):
                start_time = time.time()

                continual_model = ContinualLearningModel()
                continual_model.initialize_model(input_size=results[window][horizon]['X_train'].shape[2],
                                                 hidden_size=params['hidden_size'],
                                                 output_size=results[window][horizon]['y_train'].shape[1],
                                                 num_nodes=results[window][horizon]['X_train'].shape[1],
                                                 learning_rate=params['learning_rate'],
                                                 dropout_rate=params['dropout_rate'])

                # Training loop with early stopping
                for epoch in tqdm(range(params['num_epochs']), desc="Training Epochs"):
                    train_loss, val_loss, should_stop = continual_model.update_model(
                        results[window][horizon]['X_train'],
                        results[window][horizon]['y_train'],
                        results[window][horizon]['X_val'],
                        results[window][horizon]['y_val']
                    )
                    
                    if should_stop:
                        print(f"Early stopping triggered at epoch {epoch}")
                        break

                training_time = time.time() - start_time 
                print(f'Run {run + 1} - Validation Loss: {val_loss:.4f}, Training Time: {training_time:.2f} seconds')

                # Testing phase
                test_loss, test_mae, test_rmse = continual_model.evaluate(results[window][horizon]['X_test'], results[window][horizon]['y_test'])
                
                print(f'Run {run + 1} - Test Loss (MSE): {test_loss:.4f}, Test MAE: {test_mae:.4f}, Test RMSE: {test_rmse:.4f}')

                # Store losses and metrics for later visualization
                train_losses[window][horizon].append(train_loss)
                val_losses[window][horizon].append(val_loss)
                test_maes[window][horizon].append(test_mae)
                test_rmses[window][horizon].append(test_rmse)

                run_losses.append(val_loss)

            avg_loss = np.mean(run_losses)

            if avg_loss < best_loss:
                best_loss = avg_loss
                best_params = params
                best_models[window][horizon] = continual_model.model
                best_losses[window][horizon] = best_loss

        print(f'Best parameters for window {window} and horizon {horizon}: {best_params} with average validation loss: {best_loss:.4f}')
        print('\n')

        # Save the best model for this window and horizon
        torch.save(best_models[window][horizon].state_dict(), f'best_model_w{window}_h{horizon}.pth')


# Visualization: Plot metrics across runs
plot_metrics(train_losses, 'Training Loss', input_windows, prediction_horizons)
plot_metrics(val_losses, 'Validation Loss', input_windows, prediction_horizons)
plot_metrics(test_maes, 'Test MAE', input_windows, prediction_horizons)
plot_metrics(test_rmses, 'Test RMSE', input_windows, prediction_horizons)

# Add box plot visualizations
plot_boxplots(train_losses, 'Training Loss', input_windows, prediction_horizons)
plot_boxplots(val_losses, 'Validation Loss', input_windows, prediction_horizons)
plot_boxplots(test_maes, 'Test MAE', input_windows, prediction_horizons)
plot_boxplots(test_rmses, 'Test RMSE', input_windows, prediction_horizons)

# Final evaluation on the test set with the best overall model
print("Final evaluation with the best models for each window and horizon")

for window in input_windows:
    for horizon in prediction_horizons:
        print(f"\nEvaluating Window: {window}, Horizon: {horizon}")
        
        best_model = best_models[window][horizon]
        best_model.eval()

        with torch.no_grad():
            test_outputs = best_model(results[window][horizon]['X_test'])
            test_loss = nn.MSELoss()(test_outputs, results[window][horizon]['y_test'])

            test_outputs_inverse = scaler.inverse_transform(test_outputs.cpu().numpy())
            test_y_inverse = scaler.inverse_transform(results[window][horizon]['y_test'].cpu().numpy())

            test_mae = mean_absolute_error(test_y_inverse, test_outputs_inverse)
            test_rmse = np.sqrt(mean_squared_error(test_y_inverse, test_outputs_inverse))
            test_mape = MAPE(test_y_inverse, test_outputs_inverse)

        print(f'Test Loss (MSE): {test_loss.item():.4f}')
        print(f'Test MAE: {test_mae:.4f}')
        print(f'Test RMSE: {test_rmse:.4f}')
        print(f'Test MAPE: {test_mape:.2f}%')