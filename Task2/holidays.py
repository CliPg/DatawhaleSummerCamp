import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 读取数据
file_path = 'Task1/data/electricity price.csv'
data = pd.read_csv(file_path)

# 处理时间格式
data['time'] = data['time'].str.strip()
data['time'] = data['time'].apply(lambda x: '00:00:00' if x == '24:00:00' else x)
data['time'] = data['time'].apply(lambda x: x if len(x) == 8 else (x + ':00'))

# 创建新的时间戳列
data['timestamp'] = pd.to_datetime(data['day'] + ' ' + data['time'], format='%Y/%m/%d %H:%M:%S')

# 修正日期
data.loc[data['time'] == '00:00:00', 'timestamp'] += pd.Timedelta(days=1)

# 选择你要绘制的日期范围
start_date = '2022-04-01'
end_date = '2022-04-11'
mask = (data['timestamp'] >= start_date) & (data['timestamp'] <= end_date)
filtered_data = data[mask]

# 创建绘图
plt.figure(figsize=(16, 6))

# 绘制demand折线图
#sns.lineplot(x='timestamp', y='demand', data=filtered_data, color='blue', label='Demand')

# 绘制clearingprice折线图
sns.lineplot(x='timestamp', y='clearing price (CNY/MWh)', data=filtered_data, color='green', label='Clearing Price')

# 添加负出清价格的红色虚线
plt.axhline(y=0, color='red', linestyle='--', linewidth=2)

# 设置图表标题和标签
plt.title('Demand and Clearing Price Trends: {} - {}'.format(start_date, end_date))
plt.xlabel('Timestamp')
plt.ylabel('Values')

# 添加背景色条
neg_price_periods = filtered_data[filtered_data['clearing price (CNY/MWh)'] < 0]
for _, row in neg_price_periods.iterrows():
    plt.axvspan(row['timestamp'], row['timestamp'] + pd.Timedelta(minutes=15), color='yellow', alpha=0.3)


# 显示图例
plt.legend()

# 显示图表
plt.show()



