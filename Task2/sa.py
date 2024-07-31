import pandas as pd

# 检查电力价格数据的列名
electricity_price=pd.read_csv('datas/electricity_price_parsed.csv',parse_dates=["timestamp"])
print(electricity_price.columns)


