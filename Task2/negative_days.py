import pandas as pd

# 假设文件路径为 'Task1/data/electricity price.csv'
file_path = 'Task1/data/electricity price.csv'
data = pd.read_csv(file_path)

# 转换日期列为 datetime 格式
data['day'] = pd.to_datetime(data['day'], format='%Y/%m/%d')

# 过滤出负出清价格
negative_prices = data[data['clearing price (CNY/MWh)'] < 0]

# 将负出清价格的数据保存到新的 CSV 文件
negative_prices.to_csv('negative_clearing_prices.csv', index=False)

# # 获取负出清价格的日期
# negative_days = negative_prices['day'].unique()

# # 输出负出清价格的日期
# print("Negative clearing prices occurred on the following days:")
# for day in negative_days:
#     print(day.strftime('%Y-%m-%d'))


# # 过滤出负出清价格
# negative_prices = data[data['price'] < 0]


