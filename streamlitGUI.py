from main import *

def home():
    st.title("Stock Price Analysis Tool - Home Page")

def pca_and_kmeans_page():
    st.title("Task 2: PCA and KMeans Clustering")
    st.write(pca_reduction_and_kmeans_clustering())

def correlation_analysis_page():
    st.title("Correlation Analysis")
    correlation_matrix_between_my_selected_stocks()
    top10_positive_negative_correlation()

def time_series_plots_page():
    st.title("Time Series Plots")
    time_series_plots_for_my_selected_stocks()

def stock_forecast_analysis_page():
    st.title("Stock Forecast Analysis")
    user_selected_stock_forecast_analysis()

def main():
    st.sidebar.title("Navigation")
    pages = ["Home", "PCA and KMeans", "Correlation Analysis", "Time Series Plots", "Stock Forecast Analysis"]
    choice = st.sidebar.selectbox("Go to", pages)

    if choice == "Home":
        home()
    elif choice == "PCA and KMeans":
        pca_and_kmeans_page()
    elif choice == "Correlation Analysis":
        correlation_analysis_page()
    elif choice == "Time Series Plots":
        time_series_plots_page()
    elif choice == "Stock Forecast Analysis":
        stock_forecast_analysis_page()

if __name__ == "__main__":
    main()
