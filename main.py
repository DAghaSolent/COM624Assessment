import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Nasdaq 100 companies stored in a list below called tickers
tickers = ['AAPL', 'MSFT', 'AMZN', 'NVDA', 'META', 'AVGO', 'GOOGL', 'GOOG', 'TSLA', 'ADBE', 'COST', 'PEP', 'NFLX', 'AMD'
           , 'CSCO', 'INTC', 'TMUS', 'CMCSA', 'INTU', 'QCOM', 'AMGN', 'TXN', 'HON', 'AMAT', 'SBUX', 'ISRG', 'BKNG',
           'MDLZ', 'LRCX', 'ADP', 'GILD', 'ADI', 'VRTX', 'REGN', 'MU', 'SNPS', 'PANW', 'PDD', 'MELI', 'KLAC', 'CDNS',
           'CSX', 'MAR', 'PYPL', 'CHTR', 'ASML', 'ORLY', 'MNST', 'CTAS', 'ABNB', 'LULU', 'NXPI', 'WDAY', 'CPRT', 'MRVL',
           'PCAR', 'CRWD', 'KDP', 'MCHP', 'ROST', 'ODFL', 'DXCM', 'ADSK', 'KHC', 'PAYX', 'FTNT', 'AEP', 'SGEN', 'CEG',
           'IDXX', 'EXC', 'AZN', 'EA', 'CTSH', 'FAST', 'VRSK', 'CSGP', 'BKR', 'DDOG', 'BIIB', 'XEL', 'GFS',
           'TTD', 'ON', 'MRNA', 'ZS', 'TEAM', 'FANG', 'WBD', 'ANSS', 'DLTR', 'EBAY', 'SIRI', 'WBA', 'ALGN', 'ZM', 'ILMN'
           , 'ENPH', 'JD', 'LCID']

# Creating variables to be used to set dates to download data within a 1-year timeframe.
end_date = datetime(2023, 12, 18) # Hardcoding the date as I am getting null errors from a specific stock after 18th Dec
start_date = end_date - timedelta(365)

# Empty dataframe which will be used to store the Adjusted close values for each Nasdaq 100 company that is stored in
# the tickers list.
adjClose_data = pd.DataFrame()

# Loop through each ticker within the tickers list and download 1 year adjusted close prices for that individual ticker,
# once obtained put that information in the new data frame that I created above
for ticker in tickers:
    data = yf.download(ticker, start=start_date, end=end_date)
    adjClose_data[ticker] = data['Adj Close']

# Utilising Pandas to display all rows to show all stocks
pd.set_option('display.max_rows', None)

# Transposing the data to get the right number of rows and columns for the assessment requirements
transposed_adjClose_data = adjClose_data.T

print(f"Before PCA reduction the shape of the data frame is{transposed_adjClose_data.shape}")

scaler = StandardScaler()
scaled_data = scaler.fit_transform(transposed_adjClose_data)

pca = PCA(n_components=10)
pca_reduced_data = pca.fit_transform(scaled_data)

explained_variance = pca.explained_variance_ratio_
print(f"After PCA reduction the shape of the data frame is{pca_reduced_data.shape}")
print("_______________________________________________________________________________________________________________")
pca_reduced_data_dataFrame = pd.DataFrame(data=pca_reduced_data, columns=[f"PC{i}" for i in range(1, pca_reduced_data.shape[1] + 1)])
pca_reduced_data_dataFrame["Tickers"] = tickers
print(pca_reduced_data_dataFrame.head())
print("_______________________________________________________________________________________________________________")

csvfilepath = r'C:\Users\Danny\Documents\Uni Solent Work\Year 3\COM624 Machine Learning\COM624 Assessment\pca_reduced_data.csv'
pca_reduced_data_dataFrame.to_csv(csvfilepath, index=False)

# Data preprocessing to only use the 10 PCA reduced columns and ignore the ticker names
pca_reduced_data_numeric_values = pca_reduced_data_dataFrame.iloc[:, 0:10]

# Kmeans Clustering the stocks into 4 clusters
kmeans = KMeans(n_clusters=4, init='k-means++', random_state=42)
cluster_labels = kmeans.fit_predict(pca_reduced_data_numeric_values)

# Creating a new dataframe to visualise which tickers represent in which cluster
kmeans_clustering_results_df = pd.DataFrame({'Ticker': tickers, 'Assigned Cluster': cluster_labels})

# Exporting the tickers and their Assigned Cluster label to better visualise the clusters and which ticker is assigned
# to which cluster.
print(kmeans_clustering_results_df)
cluster_csvfilepath = r'C:\Users\Danny\Documents\Uni Solent Work\Year 3\COM624 Machine Learning\COM624 Assessment\kmeans_clustering_results.csv'
kmeans_clustering_results_df.to_csv(cluster_csvfilepath, index=False)

print("_______________________________________________________________________________________________________________")

# Empty dataframe to store the Adjusted Close values for my selected stocks.
# My selected stocks are [NVDA, AMD, BKNG, ORLY]
selected_stocks = pd.DataFrame()

for ticker in tickers:
    if ticker in ('NVDA', 'AMD', 'BKNG', 'ORLY'):
        selected_stock_data = yf.download(ticker, start=start_date, end=end_date)
        selected_stocks[ticker] = selected_stock_data['Adj Close']

# Obtain the correlation info for my selected stocks [NVDA, AMD, BKNG, ORLY].
selected_stocks_correlated = selected_stocks.corr()

# Obtain the correlation info for the whole dataset stocks to compare and correlate against my selected stocks
adjClose_data_correlated = adjClose_data.corr()

# Looping through each stock from my selected stocks and displaying the stock and their top 10 positive/negative
# correlations from the entire dataset.
for stock in selected_stocks:
    print(f"Top 10 Positive Correlations with {stock}:")
    top10_positive_correlations_with_stock = adjClose_data_correlated[stock].sort_values(ascending=False).head(11)[1:]
    print(top10_positive_correlations_with_stock)

    # Converting the top10_positive_correlations_with_stock to a DataFrame, so that I can plot the positive correlation
    # between my selected stocks that are positively correlated against stocks from the whole dataset.
    top10_positive_correlations_with_stock_df = pd.DataFrame(top10_positive_correlations_with_stock, columns=[stock])

    # Creating and displaying the heatmap off the positive correlations for my selected stocks against the stocks from
    # the whole dataset.
    plt.figure(figsize=(10, 8))
    sns.heatmap(top10_positive_correlations_with_stock_df, annot=True, cmap='coolwarm')
    plt.title(f"Top 10 Positive Correlations with {stock}")
    plt.show()

    print("___________________________________________________________________________________________________________")

    print(f"Top 10 Negative Correlations with {stock}:")
    top10_negative_correlations_with_stock = adjClose_data_correlated[stock].sort_values().head(10)
    print(top10_negative_correlations_with_stock)

    # Converting the top10_negative_correlations_with_stock to a DataFrame, so that I can plot the negative correlation
    # between my selected stocks that are negatively correlated against stocks from the whole dataset.
    top10_negative_correlations_with_stock_df = pd.DataFrame(top10_negative_correlations_with_stock, columns=[stock])

    # Creating and displaying the heatmap off the negative correlations for my selected stocks against the stocks from
    # the whole dataset.
    plt.figure(figsize=(10, 8))
    sns.heatmap(top10_negative_correlations_with_stock_df, annot=True, cmap='coolwarm')
    plt.title(f"Top 10 Negative Correlations with {stock}")
    print("___________________________________________________________________________________________________________")

print("_______________________________________________________________________________________________________________")
print("Correlation Info between my selected stocks")
# Heatmap that I created to visualise the correlation matrix between my selected stocks [NVDA, AMD, BKNG, ORLY].
plt.figure(figsize=(10, 8))
print(selected_stocks_correlated)
sns.heatmap(selected_stocks_correlated, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix Heatmap for my selected stocks")
plt.show()