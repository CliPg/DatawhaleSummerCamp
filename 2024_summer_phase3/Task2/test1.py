import numpy as np
import pandas as pd
import seaborn as sns
import os
from tqdm import tqdm
import matplotlib.pylab as plt
from pathlib import Path
import warnings
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector

warnings.filterwarnings('ignore')
plt.style.use('ggplot')
plt.rcParams['font.sans-serif'] = ["WenQuanYi Micro Hei",'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取电力价格数据
electricity_price=pd.read_csv('datas/electricity_price_parsed.csv',parse_dates=["timestamp"],index_col=0)
electricity_price.columns=["demand","price"]
sample_submit=pd.read_csv('Task1/data/sample_submit.csv')

# 添加时间特征
train_data = electricity_price.copy(deep=True)
# 提取时间索引的小时信息，并添加到训练数据中，创建 "hour" 列
train_data["hour"] = electricity_price.index.hour

# 提取时间索引的日期信息，并添加到训练数据中，创建 "day" 列
train_data["day"] = electricity_price.index.day

# 提取时间索引的月份信息，并添加到训练数据中，创建 "month" 列
train_data["month"] = electricity_price.index.month

# 提取时间索引的年份信息，并添加到训练数据中，创建 "year" 列
train_data["year"] = electricity_price.index.year

# 提取时间索引的星期信息，并添加到训练数据中，创建 "weekday" 列
train_data["weekday"] = electricity_price.index.weekday

# 根据月份信息，判断是否为风季（1-5月和9-12月），创建布尔型 "is_windy_season" 列
train_data["is_windy_season"] = electricity_price.index.month.isin([1, 2, 3, 4, 5, 9, 10, 11, 12])

# 根据小时信息，判断是否为低谷时段（10-15点），创建布尔型 "is_valley" 列
train_data["is_valley"] = electricity_price.index.hour.isin([10, 11, 12, 13, 14, 15])

# 提取时间索引的季度信息，并添加到训练数据中，创建 "quarter" 列
train_data["quarter"] = electricity_price.index.quarter

# 对时间特征进行独热编码（One-Hot Encoding），删除第一列以避免多重共线性
train_data = pd.get_dummies(
    data=train_data,        # 需要进行独热编码的 DataFrame
    columns=["hour", "day", "month", "year", "weekday"],  # 需要独热编码的列
    drop_first=True         # 删除第一列以避免多重共线性
)

def generate_holiday_dates(start_dates, duration):
    """
    生成一系列节假日日期列表。

    参数：
    start_dates (list): 节假日开始日期的列表，格式为字符串，例如 ["2022-01-31", "2023-01-21"]。
    duration (int): 每个节假日的持续天数。

    返回：
    list: 包含所有节假日日期的列表。
    """
    holidays = []  # 初始化一个空列表，用于存储节假日日期
    for start_date in start_dates:  # 遍历每个节假日的开始日期
        # 生成从 start_date 开始的日期范围，持续时间为 duration 天
        holidays.extend(pd.date_range(start=start_date, periods=duration).tolist())
    return holidays  # 返回所有节假日日期的列表

# 春节的开始日期列表
spring_festival_start_dates = ["2022-01-31", "2023-01-21", "2024-02-10"]

# 劳动节的开始日期列表
labor_start_dates = ["2022-04-30", "2023-04-29"]

# 生成春节的所有日期，持续时间为 7 天
spring_festivals = generate_holiday_dates(spring_festival_start_dates, 7)

# 生成劳动节的所有日期，持续时间为 5 天
labor = generate_holiday_dates(labor_start_dates, 5)

# 判断训练数据的索引是否在春节日期列表中，生成布尔型列 "is_spring_festival"
train_data["is_spring_festival"] = train_data.index.isin(spring_festivals)

# 判断训练数据的索引是否在劳动节日期列表中，生成布尔型列 "is_labor"
train_data["is_labor"] = train_data.index.isin(labor)

# 滚动窗口特征
def cal_range(x):
    return x.max() - x.min()

def increase_num(x):
    return (x.diff() > 0).sum()

def decrease_num(x):
    return (x.diff() < 0).sum()

def increase_mean(x):
    diff = x.diff()
    return diff[diff > 0].mean()

def decrease_mean(x):
    diff = x.diff()
    return diff[diff < 0].abs().mean()

def increase_std(x):
    diff = x.diff()
    return diff[diff > 0].std()

def decrease_std(x):
    diff = x.diff()
    return diff[diff < 0].std()

window_sizes = [4, 12, 24]
for window_size in tqdm(window_sizes):
    functions = ["mean", "std", "min", "max", cal_range, increase_num, decrease_num, increase_mean, decrease_mean, increase_std, decrease_std]
    for func in functions:
        func_name = func if type(func) == str else func.__name__
        column_name = f"demand_rolling_{window_size}_{func_name}"
        train_data[column_name] = train_data["demand"].rolling(window=window_size, min_periods=window_size//2, closed="left").agg(func)

# 移动和差分特征
train_data["demand_shift_1"] = train_data["demand"].shift(1)
train_data["demand_diff_1"] = train_data["demand"].diff(1)
train_data["demand_pct_1"] = train_data["demand"].pct_change(1)

# 确定训练和测试数据
train_length = 55392
X_train = train_data.iloc[:train_length].drop(columns=["price"]).bfill().ffill()
X_test = train_data.iloc[train_length:].drop(columns=["price"]).bfill().ffill()
y_train = train_data.iloc[:train_length][["price"]]

# 定义ABM模型的消费者和发电厂代理
class ConsumerAgent(Agent):
    def __init__(self, unique_id, model, demand):
        super().__init__(unique_id, model)
        self.demand = demand

    def step(self):
        self.model.total_demand += self.demand

class PowerPlantAgent(Agent):
    def __init__(self, unique_id, model, capacity, cost):
        super().__init__(unique_id, model)
        self.capacity = capacity
        self.cost = cost

    def step(self):
        if self.model.total_demand > 0:
            supply = min(self.capacity, self.model.total_demand)
            self.model.total_supply += supply
            self.model.total_demand -= supply

# 定义ABM电力市场模型
class ElectricityMarketModel(Model):
    def __init__(self, electricity_price_df, unit_df):
        self.electricity_price_df = electricity_price_df
        self.unit_df = unit_df
        self.total_demand = 0
        self.total_supply = 0
        self.schedule = RandomActivation(self)

        for index, row in self.electricity_price_df.iterrows():
            consumer = ConsumerAgent(index, self, row['demand'])
            self.schedule.add(consumer)

        for index, row in self.unit_df.iterrows():
            plant = PowerPlantAgent(index + len(self.electricity_price_df), self, capacity=row['Capacity(MW)'], cost=row['coal consumption (g coal/KWh)'])
            self.schedule.add(plant)

        self.datacollector = DataCollector(model_reporters={"Total Demand": "total_demand", "Total Supply": "total_supply"})

    def step(self):
        self.total_demand = 0
        self.total_supply = 0
        self.schedule.step()
        self.datacollector.collect(self)

# 读取单位数据
unit_df = pd.read_csv('Task1/data/unit.csv')

# 创建模型实例
model = ElectricityMarketModel(electricity_price, unit_df)

# 运行模型
for i in range(100):
    model.step()

# 收集数据
data = model.datacollector.get_model_vars_dataframe()




# 将时间戳和预测数据结合
result_df = pd.DataFrame({
    'timestamp': time['timestamp'],
    'predicted_demand': data['Total Demand'],
    'predicted_supply': data['Total Supply']
})

# 保存预测结果到CSV文件
result_df.to_csv('predicted_electricity.csv', index=False)
print("预测结果已保存到predicted_electricity.csv")

