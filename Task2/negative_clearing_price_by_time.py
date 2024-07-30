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

# 添加小时列
data['hour'] = pd.to_datetime(data['time'], format='%H:%M:%S').dt.hour

# 过滤出负出清价格
negative_prices = data[data['clearing price (CNY/MWh)'] < 0]

# 统计各个小时负出清价格的出现次数
negative_counts = negative_prices['hour'].value_counts().sort_index()

# 绘制柱状图
plt.figure(figsize=(12, 6))
negative_counts.plot(kind='bar', color='skyblue', width=0.8)
plt.xlabel('Hour of Day')
plt.ylabel('Count of Negative Clearing Prices')
plt.title('Count of Negative Clearing Prices by Hour of Day')
plt.xticks(rotation=0)
plt.show()



