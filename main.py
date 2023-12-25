import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
from prophet.plot import plot_plotly
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import math

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
end_date = datetime.today() # Hardcoding the date as I am getting null errors from a specific stock after 19th Dec
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

csvfilepath = r'C:\Users\Danny\Documents\Uni Solent Work\Year 3\COM624 Machine Learning\COM624 Assessment\pca_reduced_data.csv'
pca_reduced_data_dataFrame.to_csv(csvfilepath, index=False)
print("_______________________________________________________________________________________________________________")

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
    # plt.show()

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
# plt.show()

# Creating and displaying a chart with a historical view of Adjusted Close prices for all my selected stocks.
plt.figure(figsize=(10, 8))

for stock in selected_stocks:
    plt.plot(selected_stocks.index, selected_stocks[stock], label=stock)

plt.title("Time Series Plot of Adjusted Close Prices for my selected stocks")
plt.xlabel("Date")
plt.ylabel("Adjusted Close Prices")
plt.legend()
# plt.show()

print("_______________________________________________________________________________________________________________")

# Facebook Prophet Method prediction
def fb_prophet():
    # Resetting and clearing the data to be processed for the Facebook Prophet Method
    selected_stocks.reset_index(inplace=True)
    selected_stocks_Date = selected_stocks['Date']

    for stock in selected_stocks:
        prophet = Prophet(
            daily_seasonality=True,
            yearly_seasonality=True,
            weekly_seasonality=True,
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10.0
        )

        # Creating a new DataFrame with the required columns for Facebook Prophet Prediction
        new_prophetDF = pd.DataFrame({'ds': selected_stocks_Date, 'y': selected_stocks[stock]})
        prophet.fit(new_prophetDF)

        # Creating a DataFrame to be used for the prediction
        future = prophet.make_future_dataframe(periods=365)

        # Passing the future DataFrame to generate a forecast prediction for my selected stocks.
        forecast = prophet.predict(future)

        # Plot the predictions that were made by Facebook Prophet Market prediction
        fig = plot_plotly(prophet, forecast)
        fig.update_layout(title_text=f"Facebook Prophet Prediction for {stock}")
        fig.show()
print("_______________________________________________________________________________________________________________")

# LSTM Model Prediction.
def lstm():
    for stock in selected_stocks:
        # Normalizing the data using MinMax Scaling
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(selected_stocks[stock].values.reshape(-1, 1))

        # Splitting the test and train data of the scaled data.
        train_size = int(len(scaled_data) * 0.8)
        test_size = int(len(scaled_data)) - train_size
        train_data, test_data = scaled_data[0: train_size], scaled_data[train_size:len(scaled_data), :1]
        print(len(train_data), len(test_data))

        # Function created to create the dataset for LSTM prediction
        def create_dataset(data, time_steps):
            X, y = [], []
            for i in range(len(data) - time_steps):
                X.append(data[i:(i + time_steps)])
                y.append(data[i + time_steps])
            return np.array(X), np.array(y)

        time_steps = 10

        # Using the create dataset function to create the dataset that will be used for LSTM Prediction
        X_train, y_train = create_dataset(train_data, time_steps)
        X_test, y_test = create_dataset(test_data, time_steps)
        print(X_train[0], y_train[0])

        # Creating the LSTM model
        model = Sequential()
        model.add(LSTM(units=50, activation='relu', input_shape=(time_steps, 1)))
        model.add(Dense(units=1))
        model.compile(optimizer='adam', loss='mse')

        # Training the model
        history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1, verbose=1, shuffle=False)

        # Visualising the training and testing process validation during training the LSTM model that I have created above.
        plt.figure(figsize=(10, 8))
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='test')
        plt.title(f"Validation loss for Stock:{stock}")
        plt.legend()
        plt.show()

        y_prediction = model.predict(X_test)

        # Evaluation of the predicted results made by the LSTM model, the visualisation shows the historic data and then
        # presents the future prediction made by the LSTM model for each stock. Finally plot the predicted results
        plt.figure(figsize=(10, 8))
        plt.plot(np.arange(0, len(y_train)), y_train, 'g', label="history")
        plt.plot(np.arange(len(y_train), len(y_train) + len(y_test)), y_test, marker='.', label="true")
        plt.plot(np.arange(len(y_train), len(y_train) + len(y_test)), y_prediction, 'r', label="prediction")
        plt.ylabel('Value')
        plt.xlabel('Time Step')
        plt.title(f"Stock:{stock}")
        plt.legend()
        plt.show()

        plt.figure(figsize=(10, 8))
        plt.plot(y_test, marker='.', label="true")
        plt.plot(y_prediction, 'r', label="prediction")
        plt.ylabel('Value')
        plt.xlabel('Time Step')
        plt.title(f"Stock:{stock}")
        plt.legend()
        plt.show()


print("_______________________________________________________________________________________________________________")
# ARIMA Model Prediction
def arima():
    for stock in selected_stocks:
        stock_prices = selected_stocks[stock]

        # Split the data into training data and testing data
        train_size = int(len(stock_prices) * 0.9)
        train_data = stock_prices[:train_size]
        test_data = stock_prices[train_size:]

        # Building the train and test data for the model
        history = [x for x in train_data]

        # Storing the stock prices predictions
        predictions = list()

        # Creating the Arima Model and fitting the Arima model ready for training.
        arima_model = ARIMA(history, order=(1, 1, 0))
        fitted_arima_model = arima_model.fit()
        forcasted_values = fitted_arima_model.forecast()[0]
        predictions.append(forcasted_values)
        history.append(test_data[0])

        # Rolling multiple forecasts
        for i in range(1, len(test_data)):
            # Prediction
            arima_model = ARIMA(history, order=(1, 1, 0))
            fitted_arima_model = arima_model.fit()
            forcasted_values = fitted_arima_model.forecast()[0]
            predictions.append(forcasted_values)
            observations = test_data[i]
            history.append(observations)

        # Plotting the results
        plt.figure(figsize=(12, 8))
        plt.plot(stock_prices, color='green', label='Train Stock Price')
        plt.plot(test_data.index, test_data, color='red', label='Real Stock Price')
        plt.plot(test_data.index, predictions, color='blue', label='Predicted Stock Price')
        plt.title(f'Stock Price Prediction for : {stock}')
        plt.legend()
        plt.show()

        # Reporting Performance for the ARIMA model
        mse = mean_squared_error(test_data, predictions)
        print('MSE: ' + str(mse))
        mae = mean_absolute_error(test_data, predictions)
        print('MAE: ' + str(mae))
        rmse = math.sqrt(mean_squared_error(test_data, predictions))
        print('RMSE: ' + str(rmse))

arima()