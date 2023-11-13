import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd

# Creating variables to be used to set dates to download data within a 1 year timeframe.
end_date = datetime.today()
start_date = end_date - timedelta(365)

# Download 1 year timeframe financial data for Nasdaq 100 index
data = yf.download("^NDX", start=start_date, end=end_date)

# Utilising Pandas to display more rows
pd.set_option('display.max_rows', None)

print(data)



