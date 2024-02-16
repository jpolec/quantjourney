"""
	Portfolio Optimization with DNN
 
	author: jpolec@gmail.com
	date: 15-02-2024
 
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns

class Portfolio(nn.Module):
	"""
	This class defines a simple Portfolio model using PyTorch.
	:param input_size: Number of input features
	:param output_size: Number of output features
	"""
	def __init__(self, input_size, output_size):
		super(Portfolio, self).__init__()
		# Define a simple MLP architecture
		self.network = nn.Sequential(
			nn.Linear(input_size, 128),
			nn.ReLU(),
			nn.Linear(128, 64),
			nn.ReLU(),
			nn.Linear(64, output_size),
			nn.Softmax(dim=-1)  # Use Softmax to ensure allocations sum up to 1
		)

	def forward(self, x):
		return self.network(x)

def loss_function(predictions, targets, lambda_reg=0.5, diversification_reg=0.1):
	"""
	Loss function: Mean return - lambda_reg * portfolio variance + diversification_reg * diversification penalty
	"""
	
	portfolio_return = torch.mean(torch.sum(predictions * targets, dim=1))
	portfolio_variance = torch.var(torch.sum(predictions * targets, dim=1))
	
	# New term to encourage diversification
	diversification_penalty = diversification_reg * torch.var(predictions)
	total_loss = -portfolio_return + lambda_reg * portfolio_variance + diversification_penalty
	
	return total_loss


def loss_function_first(predictions, targets, lambda_reg=0.5):
	"""
	Simplified loss function: Mean return - lambda_reg * portfolio variance
	"""

	portfolio_return = torch.mean(torch.sum(predictions * targets, dim=1))
	portfolio_variance = torch.var(torch.sum(predictions * targets, dim=1))
	total_loss = -portfolio_return + lambda_reg * portfolio_variance
	
	return total_loss

def train_network(model, data_loader, optimizer, epochs=1000):
	"""
	Train the model using the given data_loader and optimizer
	:param model: The model to train
	:param data_loader: DataLoader for the training data
	:param optimizer: Optimizer to use for training
	:param epochs: Number of epochs to train
	
	:return: None
	"""
	model.train()
	losses = []  # Initialize a list to store loss values
	for epoch in range(epochs):
		for inputs, targets in data_loader:
			optimizer.zero_grad()
			predictions = model(inputs)
			loss = loss_function(predictions, targets)
			loss.backward()
			optimizer.step()
			losses.append(loss.item())
		print(f'Epoch {epoch}, Loss: {loss.item()}')
		
	return losses

def prepare_data(tickers, start_date, end_date):
	"""
	Function to download stock data and prepare the dataset
	"""

	# Download historical data for given tickers
	data = yf.download(tickers, start=start_date, end=end_date)
	
	# Calculate daily returns
	daily_returns = data['Adj Close'].pct_change().dropna()
	
	# Calculate mean returns and covariance matrix for risk
	mean_returns = daily_returns.mean()
	cov_matrix = daily_returns.cov()
	
	# Prepare inputs and targets for the model
	# Note: This is a simplified example. You might want to use more sophisticated features for your inputs.
	inputs = torch.tensor(daily_returns.values[:-1], dtype=torch.float32)  # Using returns as inputs
	targets = torch.tensor(daily_returns.values[1:], dtype=torch.float32)  # Predicting next day's returns
	
	# Create a TensorDataset and DataLoader
	dataset = TensorDataset(inputs, targets)
	data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
	
	return data_loader, daily_returns

def dynamic_optimization(model, daily_returns, lookback=5):
	"""
	Perform dynamic optimization using the trained model
	"""
	# Store daily weights and returns
	daily_weights = []
	daily_returns_list = []

	# Start with an initial allocation (e.g., equal weights)
	weights = torch.ones(len(daily_returns.columns)) / len(daily_returns.columns)

	for i in range(lookback, len(daily_returns)):
		# Prepare input for the current day
		current_input = daily_returns.iloc[i-lookback:i].values.reshape(1, -1)
		current_input = torch.tensor(current_input, dtype=torch.float32)

		# Predict optimal weights
		predicted_weights = model(current_input)

		# Rebalance portfolio based on predictions (adjust weights if necessary)
		weights = predicted_weights.detach().numpy().flatten()

		# Store weights and returns for plotting
		daily_weights.append(weights.copy())
		daily_returns_list.append(daily_returns.iloc[i]['Adj Close'].values)

	return daily_weights, daily_returns_list


def plot_loss_over_epochs(losses):
	"""
	Plot the training loss over epochs
	"""
	plt.figure(figsize=(10, 6))
	plt.plot(losses, label='Training Loss')
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.title('Training Loss Over Epochs')
	plt.legend()
	plt.show()
	
def plot_allocations(allocations, tickers):
	"""
	Plot the optimized portfolio allocations
	"""
	plt.figure(figsize=(10, 6))
	plt.bar(tickers, allocations, color='blue')
	plt.xlabel('Assets')
	plt.ylabel('Allocation Proportion')
	plt.title('Optimized Portfolio Allocations')
	plt.xticks(rotation=45)
	plt.show()
	
def plot_correlation_matrix(daily_returns):
	corr = daily_returns.corr()
	plt.figure(figsize=(10, 8))
	sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
	plt.title('Correlation Matrix of Asset Returns')
	plt.show()
	
def plot_portfolio_return_distribution(optimized_allocations, daily_returns):
	# Convert daily_returns to numpy array if it's a DataFrame
	if isinstance(daily_returns, pd.DataFrame):
		daily_returns_np = daily_returns.values
	else:
		daily_returns_np = daily_returns
	
	# Ensure optimized_allocations is a numpy array for dot product
	portfolio_returns = np.dot(daily_returns_np, optimized_allocations)
	
	plt.figure(figsize=(10, 6))
	plt.hist(portfolio_returns, bins=50, alpha=0.75)
	plt.xlabel('Return')
	plt.ylabel('Frequency')
	plt.title('Portfolio Return Distribution')
	plt.show()

def find_nearest_date_index(dates, target_date):
	"""
	Find the index of the nearest date to the target_date in the given dates series.
	"""
	# Convert target_date to datetime if it's a string
	if isinstance(target_date, str):
		target_date = pd.to_datetime(target_date)
	
	# Find the index of the nearest date
	nearest_date_index = np.abs(dates - target_date).argmin()
	return nearest_date_index

def backtest(model, daily_returns, start_date, end_date):
	"""
	Backtest the model using the given daily_returns and date range
	"""
	dates = daily_returns.index  # Assuming this is a DatetimeIndex
	
	# Find indices for start and end dates
	start_index = find_nearest_date_index(dates, start_date)
	end_index = find_nearest_date_index(dates, end_date)

	test_returns = []

	# Assuming 'daily_returns' DataFrame has one row per day and columns for each asset
	for i in range(start_index, end_index + 1):
		# Adjust based on the structure of your inputs to the model
		current_input = daily_returns.iloc[i - 1].values.reshape(1, -1)  # Example adjustment
		current_input = torch.tensor(current_input, dtype=torch.float32)
		
		predicted_weights = model(current_input).detach().numpy().flatten()
		# Calculate the return for the day based on the predicted weights
		day_return = np.dot(predicted_weights, daily_returns.iloc[i].values)
		test_returns.append(day_return)

	avg_return = np.mean(test_returns)
	std_dev = np.std(test_returns)
	return avg_return, std_dev

def rebalance_portfolio(model, daily_returns, rebalance_interval):
    """
	Rebalance the portfolio using the given model and daily returns.
	:param model: Trained model for portfolio optimization.
	:param daily_returns: DataFrame with daily returns for each asset.
	:param rebalance_interval: Number of days between rebalancing.
	:return: List of optimized asset allocations over time, List of rebalance dates.
	
    """
    optimized_allocations_over_time = []
    rebalance_dates = []
    
    for i in range(rebalance_interval, len(daily_returns)+1, rebalance_interval):
        # Ensure there's enough data to create a valid input tensor
        if i-rebalance_interval >= 0:
            input_data = daily_returns.iloc[i-rebalance_interval:i]
            current_input = torch.tensor(input_data.values, dtype=torch.float32)
            
            with torch.no_grad():
                # Use the model to predict the new allocations
                # Ensure current_input is not empty
                if current_input.size(0) > 0:
                    new_allocations = model(current_input[-1].unsqueeze(0)).numpy()
                    optimized_allocations_over_time.append(new_allocations[0])
                    
                    # Capture the rebalance date
                    rebalance_dates.append(daily_returns.index[i-1].strftime('%Y-%m-%d'))
    
    return optimized_allocations_over_time, rebalance_dates


def simulate_portfolio(daily_returns, optimized_allocations):
	"""
	Simulate the portfolio performance over the entire period using optimized allocations.
	
	:param daily_returns: DataFrame with daily returns for each asset.
	:param optimized_allocations: Numpy array with the optimized asset allocations.
	:return: Cumulative portfolio return over the entire period.
	"""
	# Calculate daily portfolio returns
	portfolio_daily_returns = (daily_returns * optimized_allocations).sum(axis=1)
	
	# Calculate cumulative return of the portfolio
	cumulative_return = (1 + portfolio_daily_returns).cumprod() - 1
	
	return cumulative_return

def plot_rebalanced_allocations_stacked(rebalanced_allocations, tickers, rebalance_dates):
	"""
	Plot a stacked bar chart of rebalanced portfolio allocations over time.
	
	:param rebalanced_allocations: List of numpy arrays with the rebalanced asset allocations.
	:param tickers: List of asset tickers.
	:param rebalance_dates: List of dates when the rebalancing occurred.
	"""
	num_rebalances = len(rebalance_dates)
	
	plt.figure(figsize=(15, 8))
	
	# Bottom of the bar, starting at 0
	bar_bottom = np.zeros(num_rebalances)
	
	# Plot a stacked bar for each asset
	for i, ticker in enumerate(tickers):
		allocations = [allocation[i] for allocation in rebalanced_allocations]
		plt.bar(rebalance_dates, allocations, bottom=bar_bottom, label=ticker)
		# Update the bottom of the bar for the next asset stack
		bar_bottom += allocations
	
	plt.xlabel('Rebalance Date')
	plt.ylabel('Allocation Proportion')
	plt.title('Rebalanced Portfolio Allocations Over Time (Stacked)')
	plt.xticks(rotation=90)
	plt.legend(loc='best')
	plt.tight_layout()  # Adjust layout to fit labels
	plt.show()

	
def main():
    
    # Define the list of tickers and the date range
	tickers = ['GOOG', 'AMZN', 'META', 'ORCL', 'INTC', 'IBM']  
 
	start_date = '2018-01-01'
	end_date = '2018-06-01'
 
	input_size = len(tickers)
	output_size = len(tickers)  # Assuming output size matches number of tickers

	# Prepare the data
	data_loader, daily_returns = prepare_data(tickers, start_date, end_date)

	# Initialize the model and optimizer
	model = Portfolio(input_size=input_size, output_size=output_size)
	optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

	# Train the model
	losses = train_network(model, data_loader, optimizer)
 
 	# Plot the training loss over epochs
	plot_loss_over_epochs(losses)
	
	# Define rebalancing interval (e.g., monthly)
	rebalance_interval = 20  # Approximate number of trading days in a month
	
	# Simulate dynamic rebalancing of the portfolio
	rebalanced_allocations, rebalance_dates = rebalance_portfolio(model, daily_returns, rebalance_interval)

	# Plot the rebalanced portfolio allocations over time
	plot_rebalanced_allocations_stacked(rebalanced_allocations, tickers, rebalance_dates)

	
 	# Adjusted to use 'daily_returns' DataFrame
	avg_return, std_dev = backtest(model, daily_returns, '2018-05-01', '2018-05-31')
	print(f"Backtest Avg Return: {avg_return}, Std Dev: {std_dev}")

	
	current_input = daily_returns.values[-1:].astype('float32')  # Get the last day returns
	current_input = torch.tensor(current_input).unsqueeze(0)  # Add an extra dimension to match (N, input_size)
	
	# Calculate optimized allocations
	optimized_allocations = model(current_input).detach().numpy().flatten()
	print("Optimized Allocations:", optimized_allocations)
	
	 # In your main function, after obtaining the optimized allocations:
	cumulative_return = simulate_portfolio(daily_returns, optimized_allocations)
	print("Cumulative Return Over the Entire Period:", cumulative_return.iloc[-1])

	plot_portfolio_return_distribution(optimized_allocations, daily_returns)
	plot_correlation_matrix(daily_returns)
	plot_allocations(optimized_allocations, tickers)

if __name__ == '__main__':
	main()