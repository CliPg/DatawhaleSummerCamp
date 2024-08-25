import pandas as pd
import matplotlib.pyplot as plt

#按照月份划分需电量

# 读取 CSV 文件
file_path = 'Task1/data/electricity price.csv'
data = pd.read_csv(file_path)
df = pd.DataFrame(data)

# 确保 day 列的类型正确
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

# 提取年份和月份
df['year'] = df['day'].dt.year
df['month'] = df['day'].dt.month

# 按年份和月份计算平均 demand
demand_by_month_year = df.groupby(['year', 'month'])['demand'].mean().unstack(level=0)

# 绘制不同年份的折线图
plt.figure(figsize=(15, 8))

for year in demand_by_month_year.columns:
    plt.plot(demand_by_month_year.index, demand_by_month_year[year], marker='o', label=f'{year}')

# 设置横坐标标签为月份
months = range(1, 13)
month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
plt.xticks(months, month_labels)

plt.xlabel('Month')
plt.ylabel('Average Demand')
plt.title('Average Demand by Month for Different Years')
plt.legend()
plt.grid(True)
plt.show()
