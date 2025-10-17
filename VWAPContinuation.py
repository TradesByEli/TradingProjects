
import backtrader as bt
import yfinance as yf
import pandas as pd

# ----------------------
# 1. Define a simple strategy
# ----------------------
class TestStrategy(bt.Strategy):
    def next(self):
        if not self.position:
            if self.data.close[0] > self.data.close[-1]:
                self.buy(size=10)
        else:
            if self.data.close[0] < self.data.close[-1]:
                self.sell(size=10)

# ----------------------
# 2. Load historical data
# ----------------------
data_df = yf.download('AAPL', start='2025-01-01', end='2025-03-31')
data_df = data_df[['Open', 'High', 'Low', 'Close', 'Volume']]  # ensure correct columns
data_df.index.name = 'Date'  # set index name for Backtrader
data_df.index = pd.to_datetime(data_df.index)  # ensure datetime index

# Convert to Backtrader feed
data = bt.feeds.PandasData(
    dataname=data_df,
    open='Open',
    high='High',
    low='Low',
    close='Close',
    volume='Volume',
    datetime=None  # use index as datetime
)

# ----------------------
# 3. Set up Backtrader engine
# ----------------------
cerebro = bt.Cerebro()
cerebro.addstrategy(TestStrategy)
cerebro.adddata(data)
cerebro.broker.set_cash(10000)

# ----------------------
# 4. Run backtest
# ----------------------
print("Starting Portfolio Value:", cerebro.broker.getvalue())
cerebro.run()
print("Ending Portfolio Value:", cerebro.broker.getvalue())

# ----------------------
# 5. Plot results
# ----------------------
cerebro.plot()
