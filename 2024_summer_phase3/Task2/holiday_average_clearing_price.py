import pandas as pd
import matplotlib.pyplot as plt

# 读取 CSV 文件
file_path = 'Task1/data/electricity price.csv'
data = pd.read_csv(file_path)
df = pd.DataFrame(data)

# 确保 day 和 time 列的类型正确
df['day'] = pd.to_datetime(df['day'], format='%Y/%m/%d')

# 检查 time 列并添加缺失的秒数部分
df['time'] = df['time'].apply(lambda x: x if len(x) == 8 else x + ':00')

# 将 '24:00:00' 转换为 '00:00:00' 并将日期加一天
df['time'] = df['time'].replace('24:00:00', '00:00:00')
df.loc[df['time'] == '00:00:00', 'day'] += pd.Timedelta(days=1)

# 将 time 列转换为时间差类型
df['time'] = pd.to_timedelta(df['time'])

# 创建一个新的列，将 day 和 time 合并为一个时间戳
df['datetime'] = df['day'] + df['time']

# 过滤出指定日期范围内的数据
start_date = '2023-01-18'
end_date = '2023-02-01'
mask = (df['datetime'] >= start_date) & (df['datetime'] <= end_date)
filtered_df = df[mask]

# 提取年份和时段（小时和分钟）
filtered_df['year'] = filtered_df['day'].dt.year
filtered_df['hour_minute'] = filtered_df['time'].dt.total_seconds() // 60

# 按年份和时段计算平均 clearing price
demand_by_time_year = filtered_df.groupby(['year', 'hour_minute'])['clearing price (CNY/MWh)'].mean().unstack(level=0)

# 绘制不同年份的折线图
plt.figure(figsize=(20, 10))

for year in demand_by_time_year.columns:
    plt.plot(demand_by_time_year.index, demand_by_time_year[year], marker='o', label=f'{year}')

# 设置横坐标标签为时段
time_labels = [f'{int(t // 60):02}:{int(t % 60):02}' for t in demand_by_time_year.index]
plt.xticks(demand_by_time_year.index[::4], time_labels[::4], rotation=90)  # 每隔4个时段显示一个标签

plt.xlabel('Time of Day')
plt.ylabel('Average clearing price (CNY/MWh)')
plt.title('Average clearing price by Time of Day (2023-01-18 to 2023-02-01)')
plt.legend()
plt.grid(True)
plt.show()
