import pandas as pd

# 读取CSV文件
# 读取 CSV 文件
file_path = 'Task1/data/electricity price.csv'
data = pd.read_csv(file_path)
df = pd.DataFrame(data)

# 处理日期列
df['day'] = pd.to_datetime(df['day'], format='%Y/%m/%d')

# 处理时间为24:00:00的行
mask = df['time'] == '24:00:00'
df.loc[mask, 'time'] = '00:00:00'
df.loc[mask, 'day'] += pd.Timedelta(days=1)

# 重新输出数据表
output_path = 'Task1/data/processed_data.csv'
df.to_csv(output_path, index=False)

