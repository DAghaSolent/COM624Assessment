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
import warnings

warnings.filterwarnings("ignore")

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
end_date = datetime(2023, 12, 26) # Hardcoding the date as I am getting null errors from a specific stock after 26th Dec
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

def pca_reduction_and_kmeans_clustering(): # Task 2 PCA reduction and Clustering Task
    # I display the shape off the dataframe before performing PCA reduction to showcase how many columns and rows are
    # in the dataframe before PCA operation.
    print(f"Before PCA reduction the shape of the data frame is{transposed_adjClose_data.shape}")

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(transposed_adjClose_data)

    # Reducing the data
    pca = PCA(n_components=10)
    pca_reduced_data = pca.fit_transform(scaled_data)
    explained_variance = pca.explained_variance_ratio_

    # I display the results to the terminal to confirm that PCA reduction has been successful in reducing the columns
    # from 260 to 10 columns.
    print(f"After PCA reduction the shape of the data frame is{pca_reduced_data.shape}")

    # Adding a column called Tickers which shows the names off the stocks within the pca_reduced_data_dataFrame
    pca_reduced_data_dataFrame = pd.DataFrame(data=pca_reduced_data, columns=[f"PC{i}" for i in range(1, pca_reduced_data.shape[1] + 1)])
    pca_reduced_data_dataFrame["Tickers"] = tickers

    # Terminal visualization of the reduced columns done by the PCA reduction operation
    print(pca_reduced_data_dataFrame.head())

    # Exporting to a CSV for better visualisation off the reduced columns done by the PCA reduction operation
    csvfilepath = r'C:\Users\Danny\Documents\Uni Solent Work\Year 3\COM624 Machine Learning\COM624 Assessment\pca_reduced_data.csv'
    pca_reduced_data_dataFrame.to_csv(csvfilepath, index=False)

    # Data preprocessing to only use the 10 PCA reduced columns and ignore the ticker names
    pca_reduced_data_numeric_values = pca_reduced_data_dataFrame.iloc[:, 0:10]

    # Kmeans Clustering the stocks into 4 clusters
    kmeans = KMeans(n_clusters=4, init='k-means++', random_state=42)
    cluster_labels = kmeans.fit_predict(pca_reduced_data_numeric_values)

    # Creating a new dataframe to visualise which stock tickers represent in which cluster group number.
    kmeans_clustering_results_df = pd.DataFrame({'Ticker': tickers, 'Assigned Cluster': cluster_labels})

    # Exporting the tickers and their Assigned Cluster label to better visualise the clusters and which ticker is assigned
    # to which cluster.
    cluster_csvfilepath = r'C:\Users\Danny\Documents\Uni Solent Work\Year 3\COM624 Machine Learning\COM624 Assessment\kmeans_clustering_results.csv'
    kmeans_clustering_results_df.to_csv(cluster_csvfilepath, index=False)

    # Cluster Lists for better visualisation in terminal and front end GUI solution.
    cluster0 = []
    cluster1 = []
    cluster2 = []
    cluster3 = []

    # Appending to specific list depending on the Assigned Cluster Number they have been assigned by KMeans Clustering.
    for index, row in kmeans_clustering_results_df.iterrows():
        if row['Assigned Cluster'] == 0:
            cluster0.append(row['Ticker'])
        elif row['Assigned Cluster'] == 1:
            cluster1.append(row['Ticker'])
        elif row['Assigned Cluster'] == 2:
            cluster2.append(row['Ticker'])
        elif row['Assigned Cluster'] == 3:
            cluster3.append(row['Ticker'])

    # Displaying Cluster Lists
    print(f"Cluster Group 0:\n{cluster0}")
    print(f"Cluster Group 1:\n{cluster1}")
    print(f"Cluster Group 2:\n{cluster2}")
    print(f"Cluster Group 3:\n{cluster3}")

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

def top10_positive_negative_correlation():
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

        print("_______________________________________________________________________________________________________")

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
        plt.show()
        print("_______________________________________________________________________________________________________")

def correlation_matrix_between_my_selected_stocks():
    print("Correlation Info between my selected stocks")
    # Heatmap that I created to visualise the correlation matrix between my selected stocks [NVDA, AMD, BKNG, ORLY].
    plt.figure(figsize=(10, 8))
    print(selected_stocks_correlated)
    sns.heatmap(selected_stocks_correlated, annot=True, cmap='coolwarm')
    plt.title("Correlation Matrix Heatmap for my selected stocks")
    plt.show()

def time_series_plots_for_my_selected_stocks():
    # Creating and displaying a chart with a historical view of Adjusted Close prices for all my selected stocks.
    plt.figure(figsize=(10, 8))

    for stock in selected_stocks:
        plt.plot(selected_stocks.index, selected_stocks[stock], label=stock)

    plt.title("Time Series Plot of Adjusted Close Prices for my selected stocks")
    plt.xlabel("Date")
    plt.ylabel("Adjusted Close Prices")
    plt.legend()
    plt.show()

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
        fig.update_layout(xaxis_title="Dates", yaxis_title="Stock Prices", title_text=f"Facebook Prophet Prediction for {stock}")
        fig.show()

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

        # Utilising my current Arima model to predict stock prices for the next 7 days
        future_arima_model = ARIMA(stock_prices, order=(1, 1, 0))
        fitted_future_arima_model = future_arima_model.fit()
        next7_forecasted_values = fitted_future_arima_model.forecast(steps=7)

        # Plotting the forecasted predicted prices for each stock in the next 7 days
        plt.figure(figsize=(12, 8))
        plt.plot(stock_prices.index, stock_prices, label='Original Prices')
        forecast_dates = pd.date_range(start=stock_prices.index[-1], periods=8, freq='D')[1:]
        plt.plot(forecast_dates, next7_forecasted_values, color='red', label="Forecasted Prices")
        plt.title(f"7 Day forecasted prediction for: {stock}")
        plt.legend()
        plt.tight_layout()
        plt.show()

def user_selected_stock_forecast_analysis_with_fbProphet(user_selected_stock, future_days=365):
    # This function has been created to allow the user to select a stock within the tickers list to be able to get an
    # analysis forecasting prediction provided by fb_prophet for their selected stock depending on the choice of time.

    user_selected_stock_date = user_selected_stock.index
    user_selected_stock_prices = user_selected_stock[user_selected_stock.columns[0]]

    # Fetch today's latest stock data and then adding it to the user_selected_stock DataFrame
    latest_data = yf.download(user_selected_stock.columns[0], start=end_date, end=end_date)
    user_selected_stock = pd.concat([user_selected_stock, latest_data['Adj Close']])

    prophet = Prophet(
        daily_seasonality=True,
        yearly_seasonality=True,
        weekly_seasonality=True,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0
    )

    # Creating a new DataFrame with the required columns for Facebook Prophet Prediction
    newProphetDF = pd.DataFrame({'ds': user_selected_stock_date, 'y': user_selected_stock_prices})
    prophet.fit(newProphetDF)

    # Creating a new DataFrame to be will be used for the prediction, but for this prediction we will pass the period
    # as a variable so that we can utilise this function for forecast stock prices for the users inputted stock
    # against different time periods which are 7, 14 and 30 days.
    future = prophet.make_future_dataframe(periods=future_days)

    # Passing the future DataFrame with the future days variable that will be passed and changed depending on the
    # forecasting analysis on the users selected stock
    forecast = prophet.predict(future)

    # Plot the predictions that were made by Facebook Prophet Market prediction
    fig = plot_plotly(prophet, forecast)
    fig.update_layout(xaxis_title="Dates", yaxis_title="Stock Prices", title_text=f"Facebook Prophet Prediction for {user_selected_stock.columns[0]}")
    fig.show()

    # Accessing the last/latest 'Adj Close' price for the user-selected stock
    last_stock_date_price = user_selected_stock.iloc[-1]
    last_stock_date = user_selected_stock.index[-1]
    print(f"Latest Stock Information for: {user_selected_stock.columns[0]}\nLatest Updated Stock Date: {last_stock_date}\n"
          f"Latest Stock Price: {last_stock_date_price.iloc[0]}\n")

    # Print the forecasted prices for the users selected stock depending on the future days that have been entered by
    # the user with the user_selected_stock_forecast_analysis function.
    forecast.rename(columns={'ds': 'Date', 'yhat': 'Adj Close Price'}, inplace=True)
    print(f"Forecasting Prices for Stock({user_selected_stock.columns[0]}) for the Next {future_days} Days:")
    print(forecast[['Date', 'Adj Close Price']].tail(future_days))

    # I am retrieving data for the different time variance depending on the future days variable which will be 7, 14, 30
    # days. I access the data within the forecast dataframe and then offset the data depending on the time variance from
    # the future_days variable, I then save this as a variable to be used for comparison for stock analysis.
    future_days_stock_price = forecast.loc[forecast['Date'] == (last_stock_date + pd.DateOffset(days=future_days)), 'Adj Close Price'].iloc[0]

    if last_stock_date_price.iloc[0] > future_days_stock_price:
        print("\nI'd advise that you sell this stock or don't invest in this stock at all.\nLatest stock price for "
              f"({user_selected_stock.columns[0]}) is: {last_stock_date_price.iloc[0]}.\nWhich is higher than the "
              f"predicted stock price valued at: {future_days_stock_price} in {future_days} days time.")
    elif last_stock_date_price.iloc[0] < future_days_stock_price:
        print(f"\nI'd advise you to invest in this stock.\nLatest stock price for ({user_selected_stock.columns[0]}) "
              f"is: {last_stock_date_price.iloc[0]}.\nWhich is lower than the predicted stock price valued at: {future_days_stock_price}"
              f" in {future_days} days time.")

def user_selected_stock_forecast_analysis():
    stock_user_selection = input("Please enter the stock you would like to analyse: ").upper()
    user_selected_stock_DF = pd.DataFrame()
    if stock_user_selection in tickers:
        user_selected_stock_data = yf.download(stock_user_selection, start=start_date, end=end_date)
        user_selected_stock_DF[stock_user_selection] = user_selected_stock_data['Adj Close']

        user_input = int(input("Stock Found\nEnter [1] for 7 day analysis\nEnter [2] for 14 day analysis\n"
                               "Enter [3] for 30 day analysis\nPlease Enter your Selection now: "))
        if user_input == 1:
            print(f"Analysing and forecasting stock prices for {stock_user_selection} for the next 7 days")
            user_selected_stock_forecast_analysis_with_fbProphet(user_selected_stock_DF, future_days=7)
        elif user_input == 2:
            print(f"Analysing and forecasting stock prices for {stock_user_selection} for the next 14 days")
            user_selected_stock_forecast_analysis_with_fbProphet(user_selected_stock_DF, future_days=14)
        elif user_input == 3:
            print(f"Analysing and forecasting stock prices for {stock_user_selection} for the next 30 days")
            user_selected_stock_forecast_analysis_with_fbProphet(user_selected_stock_DF, future_days=30)
        else:
            print("Invalid Input")
    else:
        print("Unable to find Stock information for that inputted Stock Code")

time_series_plots_for_my_selected_stocks()