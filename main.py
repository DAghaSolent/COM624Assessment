import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

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
end_date = datetime(2023, 12, 17) # Hardcoding the date as I am getting null errors from a specific stock after 18th Dec
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