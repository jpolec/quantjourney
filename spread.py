"""
    Plotting the spread between two assets and the assets themselves
    Additionally, polynomial regression and trendlines are included
    
    author: Jakub Polec
    date: 2024-02-18
"""

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import warnings
warnings.filterwarnings("ignore")

class SpreadAnalyzer:
    def __init__(self, tickers, start_date, end_date):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.data, self.volume = self.download_prices()

    def download_prices(self) -> dict:
        """
        Download historical prices and volume for the specified tickers.
        :param tickers: List of tickers to download data for.
        :param start_date: Start date for the historical data.
        :param end_date: End date for the historical data.
        :return: Tuple of two dictionaries with ticker as key and DataFrame of historical prices and volumes as values.
        """
        data_dict = {}
        volume_dict = {}
        for ticker in self.tickers:
            try:
                data = yf.download(ticker, start=self.start_date, end=self.end_date, progress=False)
                if not data.empty:
                    data_dict[ticker] = data['Adj Close']
                    volume_dict[ticker] = data['Volume']
                else:
                    print(f"No data for ticker: {ticker}")
            except Exception as e:
                print(f"Error downloading data for ticker: {ticker}. Error: {e}")
        
        return data_dict, volume_dict

    def calculate_spread(self, ticker1, ticker2) -> pd.Series:
        """
        Calculate the spread between two assets.
        :param ticker1: First asset ticker.
        :param ticker2: Second asset ticker.
        :return: Series with the spread between the two assets.
        """
        spread_series = self.data[ticker1] - self.data[ticker2]
        return spread_series
    
    def calculate_volume_weighted_spread(self, ticker1, ticker2) -> pd.Series:
        """
        Calculate the volume-weighted spread between two assets.
        :return: Series with the volume-weighted spread between the two assets.
        """
        spread_series = self.data[ticker1] - self.data[ticker2]
        average_volume = (self.volume[ticker1] + self.volume[ticker2]) / 2
        volume_weighted_spread = spread_series * average_volume
        return volume_weighted_spread
    
    def calculate_reversals(self, spread_series, threshold=0.10, min_gap_days=30) -> tuple:
        """
        Calculate the spread reversals based on a threshold and minimum time gap.
        :param spread_series: Series with the spread between two assets.
        :param threshold: Threshold for the percentage change in the spread.
        :param min_gap_days: Minimum time gap between reversals.
        :return: Tuple with the up and down reversals.
        """
        spread_pct_change = spread_series.pct_change().fillna(0)
        reversals_up = spread_pct_change[(spread_pct_change > threshold)].index
        reversals_down = spread_pct_change[(spread_pct_change < -threshold)].index
        return reversals_up, reversals_down


    def select_significant_days(self, spread_series, num_points=5, min_gap=pd.Timedelta(days=30)) -> tuple:
        """
        Select the top and bottom spread levels with a time gap.
        :param spread_series: Series with the spread between two assets.
        :param num_points: Number of top and bottom spread levels to select.
        :param min_gap: Minimum time gap between selected points.
        :return: Tuple with the top and bottom spread levels.
        """
        # This function can be enhanced based on specific requirements for selecting days
        return spread_series.nlargest(num_points), spread_series.nsmallest(num_points)
    
    def select_extreme_spread_points(self, spread_series, num_points=4, min_gap_days=30) -> tuple:
        """
        Select top max and min spread levels with a time gap.
        :param spread_series: Series with the spread between two assets.
        :param num_points: Number of top and bottom spread levels to select.
        :param min_gap_days: Minimum time gap between selected points.
        :return: Tuple with the top and bottom spread levels.
        """

        sorted_spread = spread_series.sort_values()
        min_points, max_points = sorted_spread[:num_points], sorted_spread[-num_points:]

        def filter_for_time_gap(points) -> list:
            """
            Filter the spread levels for a minimum time gap.
            :param points: List of spread levels.
            :return: List of spread levels with a minimum time gap.
            
            """
            filtered_dates = []
            last_date = None
            for date in points.index:
                if last_date is None or (date - last_date).days >= min_gap_days:
                    filtered_dates.append(date)
                    last_date = date
            return filtered_dates

        max_dates = filter_for_time_gap(max_points)
        min_dates = filter_for_time_gap(min_points)
        return max_dates, min_dates

    def plot_spread_and_assets(self, ticker1, ticker2) -> None:
        """
        Plot the spread between two assets and the assets themselves.
        :param ticker1: First asset ticker.
        :param ticker2: Second asset ticker.
        :return: None
        """
        spread_series = self.calculate_spread(ticker1, ticker2)
        
        # Calculate percentage change and identify reversals
        spread_pct_change = spread_series.pct_change().fillna(0)
        reversals_up = spread_pct_change[(spread_pct_change > 0.10)]
        reversals_down = spread_pct_change[(spread_pct_change < -0.10)]

        def filter_reversals(reversals) -> list:
            """
            Filter reversals with a minimum time delta of 1 month between them
            """
            filtered = []
            last_date = None
            for date in reversals.index:
                if last_date is None or (date - last_date) > pd.Timedelta(days=30):
                    filtered.append(date)
                    last_date = date
            return filtered

        reversals_up_filtered = filter_reversals(reversals_up)
        reversals_down_filtered = filter_reversals(reversals_down)
        
        plt.style.use('seaborn-whitegrid')  # A clean style with subtle gridlines
        plt.rcParams['font.family'] = 'Arial'
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['legend.fontsize'] = 10
        plt.rcParams['xtick.labelsize'] = 8
        plt.rcParams['ytick.labelsize'] = 8
        plt.rcParams['axes.labelsize'] = 10  # Slightly larger font size for axis labels
        plt.rcParams['axes.labelweight'] = 'bold'
        plt.rcParams['axes.titlesize'] = 12  # Even larger font size for the title
        plt.rcParams['axes.titleweight'] = 'bold'  # Bold font for the title
        plt.rcParams['legend.frameon'] = True
        plt.rcParams['legend.framealpha'] = 0.8
        plt.rcParams['legend.edgecolor'] = 'gray'
        # Updated color scheme
        ticker1_color = '#5b9bd5'  # A muted blue
        ticker2_color = '#ed7d31'  # A muted orange
        spread_color = '#2050C0'  # Lighter navy
        vertical_color = '#FFD700'
        poly_fit_color = '#4472c4'    # A darker blue
        reversal_up_color = '#f0ad47'  # Similar to min_spread_color for consistency
        reversal_down_color = '#f00000'  # A dark red

        fig, ax1 = plt.subplots(figsize=(14, 7))

        ax1.plot(self.data[ticker1], label=f"{ticker1} Close Price", color=ticker1_color, alpha=0.9, linewidth=1.3)
        ax1.plot(self.data[ticker2], label=f"{ticker2} Close Price", color=ticker2_color, alpha=0.9, linewidth=1.3)
        
        ax2 = ax1.twinx()
        ax2.plot(spread_series, color=spread_color, label='Spread', alpha=0.9, linewidth=1.3)

        # Calculate and plot reversals with a minimum time gap
        reversals_up, reversals_down = self.calculate_reversals(spread_series, threshold=0.10, min_gap_days=30)

        # Select and mark top 4 max/min spread levels
        max_dates, min_dates = self.select_extreme_spread_points(spread_series, num_points=4, min_gap_days=30)
        for date in max_dates + min_dates:
            ax2.axvline(x=date, color=vertical_color, linestyle='-', alpha=0.8, linewidth=1)

        # Create an array of sequential numbers from 0 to the length of spread_series minus 1
        # Reshape the array into a 2D array with one column
        X = np.arange(len(spread_series)).reshape(-1, 1)
        
        # Create a PolynomialFeatures object with a degree of 5
        poly = PolynomialFeatures(degree=5)
        
        # Transform the input array X into a new array where each column represents the original values raised to the power up to the specified degree
        X_poly = poly.fit_transform(X)
        
        # Create a LinearRegression object and fit it to the transformed data X_poly and the original spread_series values
        poly_regressor = LinearRegression().fit(X_poly, spread_series.values)
        
        # Use the fitted model to predict the spread_series values based on the polynomial features of X
        y_poly_pred = poly_regressor.predict(X_poly)
        
        # Plot the predicted values against the original indices of spread_series
        ax2.plot(spread_series.index, y_poly_pred, color=spread_color, label='Polynomial Fit', alpha=0.8, linewidth=1)

        # Before plotting reversals, check if the indices are not empty
        if not reversals_up.empty:
            ax2.scatter(reversals_up, spread_series.loc[reversals_up], color=reversal_up_color, label='Reversal Up > 10%', s=10, marker='o')
        if not reversals_down.empty:
            ax2.scatter(reversals_down, spread_series.loc[reversals_down], color=reversal_down_color, label='Reversal Down > 10%', s=10, marker='o')

        # Set labels and title
        ax1.set_xlabel('Date')
        ax1.set_ylabel(f'{ticker1} and {ticker2} Close Price', color='black')
        ax2.set_ylabel('Spread Value', color=spread_color)

        # Handling legends
        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        # Combine handles and labels from both axes
        combined_handles = handles1 + handles2
        combined_labels = labels1 + labels2
        # Remove duplicates by converting to a dictionary and back to lists
        legend_dict = dict(zip(combined_labels, combined_handles))
        ax1.legend(legend_dict.values(), legend_dict.keys(), loc='upper left')

        # Title and layout adjustments
        plt.title(f"Spread Analysis: {ticker1} vs. {ticker2} with Reversals", fontsize=14)
        plt.tight_layout()

        # Saving the figure in high-resolution
        file_name = f"./spread_analysis_{ticker1}_{ticker2}.png"
        plt.savefig(file_name, format='png', dpi=300)

    def plot_spread_and_assets_with_volume(self, ticker1, ticker2) -> None:
        """
        Plot the spread between two assets, the assets themselves, and the volume-weighted spread.
        :param ticker1: First asset ticker.
        :param ticker2: Second asset ticker.
        :return: None
        """
        spread_series = self.calculate_spread(ticker1, ticker2)
        volume_weighted_spread = self.calculate_volume_weighted_spread(ticker1, ticker2)
        
        plt.style.use('seaborn-whitegrid')  # A clean style with subtle gridlines
        plt.rcParams['font.family'] = 'Arial'
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['legend.fontsize'] = 10
        plt.rcParams['xtick.labelsize'] = 8
        plt.rcParams['ytick.labelsize'] = 8
        plt.rcParams['axes.labelsize'] = 10  # Slightly larger font size for axis labels
        plt.rcParams['axes.labelweight'] = 'bold'
        plt.rcParams['axes.titlesize'] = 12  # Even larger font size for the title
        plt.rcParams['axes.titleweight'] = 'bold'  # Bold font for the title
        plt.rcParams['legend.frameon'] = True
        plt.rcParams['legend.framealpha'] = 0.8
        plt.rcParams['legend.edgecolor'] = 'gray'
        # Updated color scheme
        ticker1_color = '#5b9bd5'  # A muted blue
        ticker2_color = '#ed7d31'  # A muted orange
        spread_color = '#2050C0'  # Lighter navy
        vertical_color = '#FFD700'
        poly_fit_color = '#4472c4'    # A darker blue
        reversal_up_color = '#f0ad47'  # Similar to min_spread_color for consistency
        reversal_down_color = '#f00000'  # A dark red
        
        fig, ax1 = plt.subplots(figsize=(14, 7))

        # Plotting the original assets' prices
        ax1.plot(self.data[ticker1], label=f"{ticker1} Price", color='blue')
        ax1.plot(self.data[ticker2], label=f"{ticker2} Price", color='orange')
        ax1.set_ylabel('Asset Prices')
        ax1.legend(loc='upper left')

        # Creating a secondary y-axis for spread
        ax2 = ax1.twinx()
        ax2.plot(spread_series, label='Spread', color='green', linestyle='--')
        ax2.plot(volume_weighted_spread, label='Volume Weighted Spread', color='red', alpha=0.5)
        ax2.set_ylabel('Spread and Volume Weighted Spread')
        ax2.legend(loc='upper right')

        plt.title(f"Spread and Volume Weighted Spread Analysis: {ticker1} vs. {ticker2}")
        plt.show()
        
    def plot_spread_vol_2d(self, ticker1, ticker2) -> None:
        """
        Plot the spread and volume in a 2D scatter plot.
        :param ticker1: First asset ticker.
        :param ticker2: Second asset ticker.
        :return: None
        """

        spread_series = self.calculate_spread(ticker1, ticker2)
        volume_series = (self.volume[ticker1] + self.volume[ticker2]) / 2
        
        # Feature engineering: Create a DataFrame with normalized features
        features = pd.DataFrame({
            'Spread': spread_series,
            'Volume': volume_series
        })
        features = StandardScaler().fit_transform(features)
        
        # Step 2: Apply DBSCAN for clustering
        db = DBSCAN(eps=0.5, min_samples=10).fit(features)
        labels = db.labels_
        
        # Step 3: Plotting
        plt.figure(figsize=(10, 6))
        plt.scatter(features[:, 0], features[:, 1], c=labels, cmap='viridis', marker='o', edgecolor='k', s=50)
        plt.title(f"Volume vs. Spread Clustering for {ticker1} and {ticker2}")
        plt.xlabel("Normalized Spread")
        plt.ylabel("Normalized Volume")
        plt.colorbar(label='Regime')
        plt.show()
        
    def cluster_and_plot_regimes_with_stocks(self, ticker1, ticker2):
        # Assuming self.data, self.volume already contain the required historical data for the tickers
        spread_series = self.calculate_spread(ticker1, ticker2)
        volume_series = (self.volume[ticker1] + self.volume[ticker2]) / 2
        df = pd.DataFrame({'Date': spread_series.index, 'Spread': spread_series.values, 'Volume': volume_series.values})
        df['Normalized_Spread'] = StandardScaler().fit_transform(df[['Spread']])
        df['Normalized_Volume'] = StandardScaler().fit_transform(df[['Volume']])
        
        # Perform DBSCAN clustering on normalized features
        clustering = DBSCAN(eps=0.3, min_samples=10).fit(df[['Normalized_Spread', 'Normalized_Volume']])
        df['Regime'] = clustering.labels_

        # Set up the figure and subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 7))

        # Left plot: Prices and Spread
        color1 = 'tab:blue'
        color2 = 'tab:orange'
        color3 = 'tab:green'
        
        ax1.plot(self.data[ticker1].index, self.data[ticker1], label=f"{ticker1} Price", color=color1)
        ax1.plot(self.data[ticker2].index, self.data[ticker2], label=f"{ticker2} Price", color=color2)
        
        ax1b = ax1.twinx()
        ax1b.plot(spread_series.index, spread_series, label='Spread', color=color3, linestyle='--')
        
        # Mark regimes on the left plot
        unique_regimes = df['Regime'].unique()
        colors = plt.cm.jet(np.linspace(0, 1, len(unique_regimes)))  # Using jet for clear distinction
        for regime, col in zip(unique_regimes, colors):
            regime_dates = df[df['Regime'] == regime]['Date']
            ax1b.plot(regime_dates, spread_series[regime_dates], marker='o', linestyle='', markersize=5, color=col, label=f'Regime {regime}')

        ax1.set_xlabel('Date')
        ax1.set_ylabel(f'{ticker1} and {ticker2} Price', color=color1)
        ax1b.set_ylabel('Spread', color=color3)
        ax1.legend(loc='upper left')
        ax1b.legend(loc='upper right')

        # Right plot: 2D Clustering Outcome
        scatter = ax2.scatter(df['Normalized_Spread'], df['Normalized_Volume'], c=df['Regime'], cmap='jet', label=df['Regime'])
        legend1 = ax2.legend(*scatter.legend_elements(), title="Regimes")
        ax2.add_artist(legend1)
        ax2.set_xlabel('Normalized Spread')
        ax2.set_ylabel('Normalized Volume')
        ax2.set_title('Clustering Outcome: Regimes')

        plt.tight_layout()
        plt.show()


def main():
    
    # Download tickers data
    tickers = [
        'GLD',  # SPDR Gold Shares, a fund that reflects the performance of the price of gold bullion
        'TLT',  # iShares 20+ Year Treasury Bond ETF, reflecting performance of long-term U.S. Treasury bonds
        'SPY',  # SPDR S&P 500 ETF Trust, designed to track the S&P 500 stock market index
        'ORCL',  # Oracle Corporation, multinational computer technology corporation stock
        'MSFT',  # Microsoft Corporation, multinational technology company stock
        'QQQ',  # Invesco QQQ Trust, tracks the NASDAQ-100 index, representing technology and non-financial stocks
        'AMZN',  # Amazon.com, Inc., multinational technology and e-commerce company stock
        'NVDA',  # NVIDIA Corporation, technology company and leading graphics processing units (GPU) manufacturer stock
        'AAPL',  # Apple Inc., multinational technology company known for consumer electronics like iPhone and Mac
        'GOOGL',  # Alphabet Inc., multinational conglomerate and parent company of Google
        'TSLA',  # Tesla, Inc., electric vehicle and clean energy company stock
        'META',  # Meta Platforms, Inc., formerly known as Facebook, a social media conglomerate stock
        'NFLX',  # Netflix, Inc., streaming service and production company stock
        'VIXY',  # ProShares VIX Short-Term Futures ETF, reflects the performance of the S&P 500 VIX Short-Term Futures Index
    ]

    # Major Indices
    tickers += [
        '^GSPC',  # S&P 500 Index, a market-capitalization-weighted index of the 500 largest U.S. publicly traded companies
        '^DJI',   # Dow Jones Industrial Average, a stock market index that measures the stock performance of 30 large companies listed on stock exchanges in the United States
        '^IXIC',  # NASDAQ Composite Index, a stock market index of the common stocks and similar securities listed on the NASDAQ stock market
        '^RUT'    # Russell 2000 Index, a small-cap stock market index of the bottom 2,000 stocks in the Russell 3000 Index
    ]

    # iShares Russell ETFs
    tickers += [
        'IWV',  # iShares Russell 3000 ETF, reflecting the performance of the Russell 3000 index
        'IWF',  # iShares Russell 1000 Growth ETF, tracks an index of US large- and mid-cap stocks selected from the Russell 1000 Index with the highest growth characteristics
        'IWD',  # iShares Russell 1000 Value ETF, tracks an index of US large- and mid-cap value stocks selected from the Russell 1000 Index based on three style factors
        'IWM',  # iShares Russell 2000 ETF, corresponds to the performance of the small-cap Russell 2000 index
        'IWN',  # iShares Russell 2000 Value ETF, measures the performance of the small-cap value sector of the U.S. equity market
        'IWO',  # iShares Russell 2000 Growth ETF, measures the performance of the small-cap growth sector of the U.S. equity market
        'IWP',  # iShares Russell Mid-Cap Growth ETF, tracks the Russell Midcap Growth Index, representing mid-cap growth stocks in the United States
        'IWR',  # iShares Russell Mid-Cap ETF, reflects the performance of the Russell Midcap Index which measures the mid-cap segment of the US equity universe
        'IWS',  # iShares Russell Mid-Cap Value ETF, measures the performance of the mid-cap value sector of the U.S. equity market
        'IWB',  # iShares Russell 1000 ETF, ETF tracks the Russell 1000 index, representing 1000 large-cap U.S. stocks
        'IWC'   # iShares Micro-Cap ETF, ETF tracks the Russell Microcap Index, representing micro-cap stocks in the U.S.
    ]
              
    # X
    tickers += [
        'XLF',  # Financial Select Sector SPDR Fund, reflects the performance of the financial sector of the S&P 500 Index
        'XLE',  # Energy Select Sector SPDR Fund, reflects the performance of the energy sector of the S&P 500 Index
        'XLI',  # Industrial Select Sector SPDR Fund, reflects the performance of the industrial sector of the S&P 500 Index
        'XLY',  # Consumer Discretionary Select Sector SPDR Fund, reflects the performance of the consumer discretionary sector of the S&P 500 Index
        "XLP",  # Consumer Staples Select Sector SPDR Fund, reflects the performance of the consumer staples sector of the S&P 500 Index
        'XLV',  # Health Care Select Sector SPDR Fund, reflects the performance of the health care sector of the S&P 500 Index
        'XLP',  # Consumer Staples Select Sector SPDR Fund, reflects the performance of the consumer staples sector of the S&P 500 Index
        'XLU',  # Utilities Select Sector SPDR Fund, reflects the performance of the utilities sector of the S&P 500 Index
        'XLRE',  # Real Estate Select Sector SPDR Fund, reflects the performance of the real estate sector of the S&P 500 Index
        'XLK',  # Technology Select Sector SPDR Fund, reflects the performance of the technology sector of the S&P 500 Index
        'XLC'   # Communication Services Select Sector SPDR Fund, reflects the performance of the communication services sector of the S&P 500 Index
        'XLB'   # Materials Select Sector SPDR Fund, reflects the performance of the materials sector of the S&P 500 Index
        'XBI'   # SPDR S&P Biotech ETF, reflects the performance of the biotechnology sector of the S&P 500 Index
    ]  
              
    #tickers = ['AAPL', 'MSFT', 'SPY', 'QQQ', 'TLT','GLD','VIXY']  # Example tickers
    #analyzer = SpreadAnalyzer(tickers, "2020-01-01", "2021-01-01")
    #analyzer.cluster_and_plot_regimes_with_stocks('SPY', 'QQQ')
    #analyzer.cluster_and_plot_regimes_with_stocks('GLD', 'TLT')
    # analyzer.cluster_and_plot_regimes_with_stocks('QQQ', 'VIXY')

    tickers = ['CAT','DE', 'SPY', 'QQQ', 'TLT','MSFT','AAPL']
    analyzer = SpreadAnalyzer(tickers, "2020-01-01", "2024-02-10")
    analyzer.cluster_and_plot_regimes_with_stocks('SPY', 'QQQ')
    analyzer.cluster_and_plot_regimes_with_stocks('MSFT', 'AAPL')
    analyzer.cluster_and_plot_regimes_with_stocks('MSFT', 'QQQ')
    #analyzer.plot_spread_and_assets_with_volume('CAT','DE')

    
    # Create an instance of SpreadAnalyzer
    analyzer = SpreadAnalyzer(tickers, "2018-01-01", "2024-02-18")

    # Example
    #analyzer.plot_spread_and_assets("GLD", "TLT")

    # Set of spread pairs to analyze
    spread_pairs = [("SPY", "QQQ"), 
                    ("GLD", "TLT"), 
                    ("GLD", "SPY"), 
                    ("MSFT", "QQQ"), 
                    ("AAPL", "MSFT"), 
                    ("TSLA", "AAPL"), 
                    ("SPY", "VIXY"), 
                    ("QQQ", "VIXY"),
                    ("AMZN", "GOOGL"),
                    ("NVDA", "META"),
                    ("NFLX", "META"),
                    ("IWF", "IWD"),
                    ("IWO", "IWM"),
                    ("IWP", "IWR"),
                    ("IWS", "IWB"),
                    ("IWC", "IWN"),
                    ("XLF", "XLE"),
                    ("XLI", "XLY"),
                    ("XLV", "XLP"),
                    ("XLY", "XLP"),                    
    ]

    # Plot spread analysis for each pair
    for spread in spread_pairs:
        print(f"Spread Analysis for {spread[0]} vs. {spread[1]}")
        analyzer.plot_spread_and_assets(spread[0], spread[1])
        
if __name__ == "__main__":
    main()  