import pandas as pd
import matplotlib.pyplot as plt

#需电量区间占比

# 读取 CSV 文件
file_path = 'Task1/data/electricity price.csv'
data = pd.read_csv(file_path)
df = pd.DataFrame(data)

# 找到 demand 的最大值和最小值
demand_min = df['clearing price (CNY/MWh)'].min()
demand_max = df['clearing price (CNY/MWh)'].max()

print(f'clearing price 最小值: {demand_min}')
print(f'clearing price 最大值: {demand_max}')

# 创建更多区间（bins）
bins = list(range(-100, 1200, 50))  # 从7000到75500，每50一个区间

# 将 demand 列分成区间，并获取区间中位数
df['clearing_price_bin'] = pd.cut(df['clearing price (CNY/MWh)'], bins=bins, labels=False, right=False)
df['clearing_price_mid'] = df['clearing_price_bin'] * 50 - 75  # 区间中位数

# 计算每个区间的数量
demand_counts = df['clearing_price_mid'].value_counts().sort_index()

# 创建图形和轴对象
fig, ax = plt.subplots(figsize=(20, 10))

# 绘制柱状图
demand_counts.plot(kind='bar', width=1, ax=ax, color='skyblue', label='Count')
ax.set_xlabel('clearing_price')
ax.set_ylabel('Count')

# 在同一个轴上绘制折线图
ax.plot(range(len(demand_counts)), demand_counts.values, color='red', marker='o', label='Line Plot')

# 设置横坐标标签显示间隔，例如每隔2个显示一个
# labels = [int(label) for i, label in enumerate(demand_counts.index) if i % 2 == 0]
# ax.set_xticks(range(0, len(demand_counts), 2))
# ax.set_xticklabels(labels, rotation=90)

# 添加图例
ax.legend(loc='upper left')

plt.show()