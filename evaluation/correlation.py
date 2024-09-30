import json
from scipy.stats import pearsonr, spearmanr
import seaborn as sns
import numpy as np
import pandas as pd
from influxdb_client import InfluxDBClient
from scipy.stats import pearsonr, spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import datetime as dt
# InfluxDB details
with open("../config.json", "r") as file:
    config = json.load(file)

influxdb_config = config['influxdb']
client = InfluxDBClient(**influxdb_config)
query_api = client.query_api()
bucket_trades = influxdb_config['bucket']
bucket_ohlcv = influxdb_config['bucket_ohlcv']
org = influxdb_config['org']
symbol = config['symbol']
url = influxdb_config['url']

start = dt.datetime.fromisoformat('2024-07-21T12:00:00')
end = dt.datetime.fromisoformat('2024-08-31T12:30:00')
start_time = start.strftime('%Y-%m-%dT%H:%M:%SZ')
stop_time = end.strftime('%Y-%m-%dT%H:%M:%SZ')

window = '1h'


# Function to fetch number of trades from InfluxDB
def fetch_number_of_trades(start_time, stop_time):
        """
        Fetch the number of trades in specified windows from InfluxDB.

        Args:
            client (InfluxDBClient): InfluxDB client instance.
            start_time (str): Start time in RFC3339 format.
            stop_time (str): Stop time in RFC3339 format.
            symbol (str): Trading symbol to filter.
            window (str): Aggregation window (e.g., '4h').

        Returns:
            pd.DataFrame: DataFrame with timestamps and trade counts.
        """
        query = f'''
        from(bucket: "{bucket_trades}")
          |> range(start: {start_time}, stop: {stop_time})
          |> filter(fn: (r) => r["_measurement"] == "trades")
          |> filter(fn: (r) => r["_field"] == "id")
          |> filter(fn: (r) => r["symbol"] == "{symbol}")
          |> aggregateWindow(every: {window}, fn: count, createEmpty: false)
          |> yield(name: "count")
        '''
        result = client.query_api().query(org=org, query=query)

        # Parse the result into a list of records
        records = []
        for table in result:
            for record in table.records:
                records.append((record.get_time(), record.get_value()))

        # Create DataFrame
        df = pd.DataFrame(records, columns=['time', 'trade_count'])
        df.set_index('time', inplace=True)
        return df


# Function to fetch OHLC data from InfluxDB and calculate volatility
def fetch_volatility_data(start_time, stop_time):
    """
    Fetch OHLCV data and compute volatility measures from InfluxDB.

    Args:
        client (InfluxDBClient): InfluxDB client instance.
        start_time (str): Start time in RFC3339 format.
        stop_time (str): Stop time in RFC3339 format.
        symbol (str): Trading symbol to filter.
        window (str): Aggregation window (e.g., '4h').

    Returns:
        pd.DataFrame: DataFrame with timestamps and volatility measures.
    """
    # ------------------ Fetch Standard Deviation ------------------
    query_stddev = f'''
    from(bucket: "{bucket_ohlcv}")
      |> range(start: {start_time}, stop: {stop_time})
      |> filter(fn: (r) => r["_measurement"] == "ohlcvBucket")
      |> filter(fn: (r) => r["symbol"] == "{symbol}")
      |> filter(fn: (r) => r["_field"] == "close")
      |> aggregateWindow(every: {window}, fn: stddev, createEmpty: false)
      |> yield(name: "stddev")
    '''
    result_stddev = client.query_api().query(org=org, query=query_stddev)
    stddev_records = []
    for table in result_stddev:
        for record in table.records:
            stddev_records.append((record.get_time(), record.get_value()))
    df_stddev = pd.DataFrame(stddev_records, columns=['time', 'std_dev'])
    df_stddev.set_index('time', inplace=True)

    # ------------------ Fetch Max Close Price ---------------------
    query_max = f'''
    from(bucket: "{bucket_ohlcv}")
      |> range(start: {start_time}, stop: {stop_time})
      |> filter(fn: (r) => r["_measurement"] == "ohlcvBucket")
      |> filter(fn: (r) => r["symbol"] == "{symbol}")
      |> filter(fn: (r) => r["_field"] == "close")
      |> aggregateWindow(every: {window}, fn: count, createEmpty: false)
      |> yield(name: "count")
    '''
    result_max = client.query_api().query(org=org, query=query_max)
    max_records = []
    for table in result_max:
        for record in table.records:
            max_records.append((record.get_time(), record.get_value()))
    df_max = pd.DataFrame(max_records, columns=['time', 'max_close'])
    df_max.set_index('time', inplace=True)

    # ------------------ Fetch Min Close Price ---------------------
    query_min = f'''
    from(bucket: "{bucket_ohlcv}")
      |> range(start: {start_time}, stop: {stop_time})
      |> filter(fn: (r) => r["_measurement"] == "ohlcvBucket")
      |> filter(fn: (r) => r["symbol"] == "{symbol}")
      |> filter(fn: (r) => r["_field"] == "close")
      |> aggregateWindow(every: {window}, fn: min, createEmpty: false)
      |> yield(name: "min")
    '''
    result_min = client.query_api().query(org=org, query=query_min)
    min_records = []
    for table in result_min:
        for record in table.records:
            min_records.append((record.get_time(), record.get_value()))
    df_min = pd.DataFrame(min_records, columns=['time', 'min_close'])
    df_min.set_index('time', inplace=True)

    # ------------------ Compute Max-Min Range ---------------------
    # Merge max and min DataFrames on time
    df_range = df_max.join(df_min, how='inner')
    df_range['max_min'] = df_range['max_close'] - df_range['min_close']

    # ------------------ Combine with Standard Deviation ------------
    df_volatility = df_stddev.join(df_range, how='inner')

    # Select relevant columns
    df_volatility = df_volatility[['std_dev', 'max_min']]

    return df_volatility


def compute_correlations(df, x_col, y_cols):
    """
    Compute Pearson and Spearman correlations between x_col and each y_col.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        x_col (str): Column name for the independent variable.
        y_cols (list): List of column names for dependent variables.

    Returns:
        dict: Nested dictionary with correlation coefficients and p-values.
    """
    correlations = {}
    for y_col in y_cols:
        pearson_corr, pearson_p = pearsonr(df[x_col], df[y_col])
        spearman_corr, spearman_p = spearmanr(df[x_col], df[y_col])
        correlations[y_col] = {
            'pearson_corr': pearson_corr,
            'pearson_p': pearson_p,
            'spearman_corr': spearman_corr,
            'spearman_p': spearman_p
        }
    return correlations


def plot_correlations(df, x_col, y_cols, correlations):
    """
    Plot scatter plots with regression lines for each y_col against x_col.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        x_col (str): Column name for the independent variable.
        y_cols (list): List of column names for dependent variables.
        correlations (dict): Dictionary containing correlation coefficients.
    """
    num_plots = len(y_cols)
    plt.figure(figsize=(10, 5 * num_plots))

    for i, y_col in enumerate(y_cols, 1):
        plt.subplot(num_plots, 1, i)
        sns.regplot(x=x_col, y=y_col, data=df, scatter_kws={'alpha': 0.5})
        plt.title(f'{y_col} vs {x_col}')
        plt.xlabel(x_col)
        plt.ylabel(y_col)

        # Annotate with correlation coefficients
        pearson = correlations[y_col]['pearson_corr']
        spearman = correlations[y_col]['spearman_corr']
        plt.annotate(f"Pearson r = {pearson:.2f}\nSpearman rho = {spearman:.2f}",
                     xy=(0.05, 0.95), xycoords='axes fraction',
                     ha='left', va='top', fontsize=12,
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1))

    plt.tight_layout()
    plt.savefig('plots/correlation_1h.png')

# Fetch Number of Trades
print("Fetching number of trades...")
df_trades = fetch_number_of_trades(start_time, stop_time)
print("Number of Trades (Sample):")
print(df_trades.head(), "\n")

# Fetch Volatility Data
print("Fetching volatility data...")
df_volatility = fetch_volatility_data(start_time, stop_time)
print("Volatility Data (Sample):")
print(df_volatility.head(), "\n")

# Merge Datasets on Time
print("Merging datasets...")
df_merged = df_trades.join(df_volatility, how='inner')
print("Merged Data (Sample):")
print(df_merged.head(), "\n")

# Drop any NaN values
df_merged.dropna(inplace=True)

# Compute Correlations
print("Computing correlations...")
correlations = compute_correlations(df_merged, 'trade_count', ['std_dev', 'max_min'])
for y, corr in correlations.items():
    print(f"{y}:")
    print(f"  Pearson Correlation: {corr['pearson_corr']:.4f} (p-value: {corr['pearson_p']:.4e})")
    print(f"  Spearman Correlation: {corr['spearman_corr']:.4f} (p-value: {corr['spearman_p']:.4e})\n")

# Plot Correlations
print("Plotting correlations...")
plot_correlations(df_merged, 'trade_count', ['std_dev', 'max_min'], correlations)