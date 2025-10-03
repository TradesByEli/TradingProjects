import yfinance as yf

tickers= ['MSTR','CRCL','COIN', 'BTC-USD']

data = yf.download(tickers, period="1mo", interval="1d")

volume= data["volume"]
print(volume)