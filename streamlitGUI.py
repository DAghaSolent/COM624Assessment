from main import *

def home():
    st.title("COM624 AE1 - Danial Agha Financial Stock Analysis Software Artefact Solution")
    st.write("This is my Software Artefact Solution which has been built to complete the tasks proposed within the "
             "assessment brief for this assessment module. To keep the data consistent for my solution and due to stock"
             " de-listing errors as well I decided to stop the data retrieval on the 26th of December 2024. ")
    st.write("Bear in mind though I am still retrieving a 1 year worth of stock data (26/12/22- 26/12-23) that I am "
             "analysing with my Software Artefact Solution. Even though I am not providing real up to date data, this "
             "is a great opportunity to showcase my solution in analysing data with real world data. ")
    st.markdown("My Software Artefact Solution is also hosted on a GitHub repository that can be accessed with this "
                "link: [COM624-AE1 GitHub Repository Software Artefact Solution](https://github.com/DAghaSolent/COM624Assessment)")
    st.markdown("To view a demonstration off my Software Artefact in action click on the following YouTube link: "
                "[YouTube Video Demonstration of Software Solution Artefact](https://youtube.com)")
    st.subheader("My Selected Stocks are below")

    col1, col2 = st.columns([0.2, 0.8])
    col1.image("https://1000logos.net/wp-content/uploads/2020/05/Amd-logo.jpg", width=100)
    col2.subheader("AMD - Advanced Micro Devices Inc")

    col3, col4 = st.columns([0.2, 0.8])
    col3.image(
        "https://www.nvidia.com/content/dam/en-zz/Solutions/about-nvidia/logo-and-brand/01-nvidia-logo-horiz-500x200-2c50-l.png",
        width=120)
    col4.subheader("NVDA - NVIDIA Corp")

    col5, col6 = st.columns([0.2, 0.8])
    col5.markdown(
        f'<img src="https://upload.wikimedia.org/wikipedia/commons/3/3d/Booking_Holdings_Inc._Logo.svg" width="120" height="80">',
        unsafe_allow_html=True)
    col6.subheader("BKNG - Booking Holdings Inc")

    col7, col8 = st.columns([0.2, 0.8])
    col7.image("https://www.logo.wine/a/logo/O'Reilly_Auto_Parts/O'Reilly_Auto_Parts-Logo.wine.svg")
    col8.subheader("ORLY - O'Reilly Automotive Inc")

    st.write("To view my Software Solution Artefact handle the tasks that were outlined in the Assessment Brief click "
             "on the navigation dropdown on the left side of the screen and chose the solution implementation. ")

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
    ml_tab1, ml_tab2, ml_tab3, ml_tab4 = st.tabs(["ARIMA", "LSTM", "Linear Regression", "FB Prophet" ])
    with ml_tab1:
        st.subheader("ARIMA Model")
        run_arima = st.button("Run Arima")
        if run_arima:
            arima()
    with ml_tab2:
        st.subheader("LSTM Model")
        run_lstm = st.button("Run LSTM")
        if run_lstm:
            lstm()
    with ml_tab3:
        st.subheader("Linear Regression")
        run_linear_regression = st.button("Run Linear Regression")
        if run_linear_regression:
            linear_regression()
    with ml_tab4:
        st.subheader("Facebook Prophet Model")
        run_fb_prophet = st.button("Run FB Prophet Model")
        if run_fb_prophet:
            fb_prophet()

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
