import pandas as pd
import matplotlib.pyplot as plt

# 假设文件路径为 'Task1/data/electricity price.csv'
file_path = 'Task1/data/electricity price.csv'
data = pd.read_csv(file_path)

# 标准化时间格式
def standardize_time_format(time_str):
    if ':' in time_str and time_str.count(':') == 1:
        # 格式为 H:M 或 HH:MM
        time_str += ':00'
    elif time_str == '24:00:00':
        # 替换 24:00:00 为 00:00:00
        time_str = '00:00:00'
    return time_str

data['time'] = data['time'].apply(standardize_time_format)
data['time'] = pd.to_datetime(data['time'], format='%H:%M:%S').dt.time

# 转换日期列为datetime格式
data['day'] = pd.to_datetime(data['day'], format='%Y/%m/%d')

# 添加月份列
data['month'] = data['day'].dt.month

# 过滤出负出清价格
negative_prices = data[data['clearing price (CNY/MWh)'] < 0]

# 统计各个月份负出清价格的出现次数
negative_counts_by_month = negative_prices['month'].value_counts().sort_index()

# 绘制柱状图
plt.figure(figsize=(12, 6))
negative_counts_by_month.plot(kind='bar', color='skyblue', width=0.8)
plt.xlabel('Month')
plt.ylabel('Count of Negative Clearing Prices')
plt.title('Count of Negative Clearing Prices by Month')
plt.xticks(rotation=0)
plt.show()


