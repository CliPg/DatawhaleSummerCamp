import pandas as pd
import matplotlib.pyplot as plt

#需电量区间占比

# 读取 CSV 文件
file_path = 'Task1/data/electricity price.csv'
data = pd.read_csv(file_path)
df = pd.DataFrame(data)

# 找到 demand 的最大值和最小值
demand_min = df['demand'].min()
demand_max = df['demand'].max()

print(f'Demand 最小值: {demand_min}')
print(f'Demand 最大值: {demand_max}')

# 创建更多区间（bins）
bins = list(range(7000, 76000, 2000))  # 从7000到75500，每2000一个区间

# 将 demand 列分成区间，并获取区间中位数
df['demand_bin'] = pd.cut(df['demand'], bins=bins, labels=False, right=False)
df['demand_mid'] = df['demand_bin'] * 2000 + 8000  # 区间中位数

# 计算每个区间的数量
demand_counts = df['demand_mid'].value_counts().sort_index()

# 创建图形和轴对象
fig, ax = plt.subplots(figsize=(20, 10))

# 绘制柱状图
demand_counts.plot(kind='bar', width=1, ax=ax, color='skyblue', label='Count')
ax.set_xlabel('Demand')
ax.set_ylabel('Count')

# 在同一个轴上绘制折线图
ax.plot(range(len(demand_counts)), demand_counts.values, color='red', marker='o', label='Line Plot')

# 设置横坐标标签显示间隔，例如每隔2个显示一个
labels = [int(label) for i, label in enumerate(demand_counts.index) if i % 2 == 0]
ax.set_xticks(range(0, len(demand_counts), 2))
ax.set_xticklabels(labels, rotation=90)

# 添加图例
ax.legend(loc='upper left')

plt.show()



