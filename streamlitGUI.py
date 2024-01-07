from main import *

def home():
    st.title("Stock Price Analysis Tool - Home Page")

def pca_and_kmeans_page():
    st.title("Task 2: PCA and KMeans Clustering")
    pca_reduction_and_kmeans_clustering()

def correlation_analysis_page():
    st.title("Task 3: Correlation Analysis")
    top10_positive_negative_correlation()

def eda_visual_analysis():
    st.title("Task 4: EDA Visual Analysis of my selected stocks")
    st.write("In this page I created Exploratory Data Analysis to visualise, observe and compare the differences "
             "between my selected stocks. Click on the tabs below to see the Visual EDAs that I created for my selected"
             " stocks.")
    eda_tab1, eda_tab2 = st.tabs(["Time Series Analysis", "Correlation Matrix Heatmap"])
    with eda_tab1:
        time_series_plots_for_my_selected_stocks()
    with eda_tab2:
        correlation_matrix_between_my_selected_stocks()

def ml_models_prediction_forecasting():
    st.title("Task 5: Machine Learning Models for Prediction and Forecasting")
    st.write("In this page I utilised Machine Learning Models to train over my selected stock data to predict and "
             "forecast future adjusted close prices for each of my selected stock. In this page I was also able to "
             "critically analyse all the Machine Learning models to determine which would be the best one to provide "
             "users off my application the ability to predict and forecast adjusted close prices for any user selected "
             "stock.")
    st.write("Click on the tabs below to see the visual representation of the prediction and forecasting off my selected"
             " stocks utilising the machine learning models FB Prophet, LSTM, ARIMA.")
    st.write("**My Selected Stocks were [AMD, NVDA, BKNG, ORLY]**")
    ml_tab1, ml_tab2, ml_tab3, ml_tab4 = st.tabs(["LSTM", "FB Prophet", "ARIMA", "Linear Regression"])
    with ml_tab1:
        st.subheader("LSTM Model")
        run_lstm = st.button("Run LSTM")
        if run_lstm:
            lstm()
    with ml_tab2:
        st.subheader("Facebook Prophet Model")
        run_fb_prophet = st.button("Run FB Prophet Model")
        if run_fb_prophet:
            fb_prophet()
    with ml_tab3:
        st.subheader("ARIMA Model")
        run_arima = st.button("Run Arima")
        if run_arima:
            arima()
    with ml_tab4:
        st.subheader("Linear Regression")
        run_linear_regression = st.button("Run Linear Regression")
        if run_linear_regression:
            linear_regression()

def general_trading_analysis():
    st.title("Task 6: General Trading Signals and Analysis")
    user_selected_stock_forecast_analysis()

def main():
    st.sidebar.title("Navigation")
    pages = ["Home", "PCA and KMeans", "Correlation Analysis", "EDA Visual Analysis",
             "Machine Learning Models for Prediction and Forecasting", "General Trading Signals and Analysis"]
    choice = st.sidebar.selectbox("Go to", pages)

    if choice == "Home":
        home()
    elif choice == "PCA and KMeans":
        pca_and_kmeans_page()
    elif choice == "Correlation Analysis":
        correlation_analysis_page()
    elif choice == "EDA Visual Analysis":
        eda_visual_analysis()
    elif choice == "Machine Learning Models for Prediction and Forecasting":
        ml_models_prediction_forecasting()
    elif choice == "General Trading Signals and Analysis":
        general_trading_analysis()

if __name__ == "__main__":
    main()
