import pandas as pd

# 读取CSV文件
# 读取 CSV 文件
electricity_price = pd.read_csv(file_path, encoding='latin1')

train_data["hour"] = electricity_price.index.hour
train_data["day"] = electricity_price.index.day
train_data["month"] = electricity_price.index.month
train_data["year"] = electricity_price.index.year
train_data["weekday"] = electricity_price.index.weekday
# 根据月份信息，判断是否为风季（1-5月和9-12月），创建布尔型 "is_windy_season" 列
train_data["is_windy_season"] = electricity_price.index.month.isin([1, 2, 3, 4, 5, 9, 10, 11, 12])
# 根据小时信息，判断是否为低谷时段（10-15点），创建布尔型 "is_valley" 列
train_data["is_valley"] = electricity_price.index.hour.isin([10, 11, 12, 13, 14, 15])
train_data["quarter"] = electricity_price.index.quarter
# 对时间特征进行独热编码（One-Hot Encoding），删除第一列以避免多重共线性
train_data = pd.get_dummies(
    data=train_data,        # 需要进行独热编码的 DataFrame
    columns=["hour", "day", "month", "year", "weekday"],  # 需要独热编码的列
    drop_first=True         # 删除第一列以避免多重共线性
)

