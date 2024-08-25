import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 读取数据
electricity_price = pd.read_csv('datas/electricity_price_parsed.csv', parse_dates=["timestamp"], index_col=0)
unit_df = pd.read_csv('Task1/data/new_unit.csv')

# 提取时间特征
electricity_price["hour"] = electricity_price.index.hour
electricity_price["day"] = electricity_price.index.day
electricity_price["month"] = electricity_price.index.month
electricity_price["year"] = electricity_price.index.year
electricity_price["weekday"] = electricity_price.index.weekday

# 添加特定时间段的布尔特征
electricity_price["is_windy_season"] = electricity_price.index.month.isin([1, 2, 3, 4, 5, 9, 10, 11, 12])
electricity_price["is_valley"] = electricity_price.index.hour.isin([3, 4, 10, 11, 12, 13, 14, 15])

# 独热编码时间特征
train_data = pd.get_dummies(electricity_price, columns=["hour", "day", "month", "year", "weekday"], drop_first=True)

def generate_holiday_dates(start_dates, duration):
    holidays = []
    for start_date in start_dates:
        holidays.extend(pd.date_range(start=start_date, periods=duration).tolist())
    return holidays

# 定义假期日期
spring_festival_start_dates = ["2022-01-31", "2023-01-18", "2024-02-10"]
qingming_festival_start_dates = ["2022-04-01", "2023-04-01", "2024-04-01"]
labor_start_dates = ["2022-04-30", "2023-04-29"]

# 生成假期日期
spring_festivals = generate_holiday_dates(spring_festival_start_dates, 14)
labor = generate_holiday_dates(labor_start_dates, 10)
qingming = generate_holiday_dates(qingming_festival_start_dates, 10)

# 添加假期布尔特征
train_data["is_spring_festival"] = train_data.index.isin(spring_festivals)
train_data["is_labor"] = train_data.index.isin(labor)
train_data["is_qingming"] = train_data.index.isin(qingming)


class PowerPlantAgent:
    def __init__(self, agent_id, capacity, actual_cost):
        self.agent_id = agent_id
        self.capacity = capacity
        self.actual_cost = actual_cost
        self.bid_price = actual_cost
        self.participation_status = False
        self.compete_success = False

    def update_bid(self, environment):
        discount_factor = 1.0

        # 计算折扣因素
        if environment.is_special_day():
            discount_factor *= 0.1

        if environment.is_special_month():
            discount_factor *= 0.5

        if environment.is_festival():
            discount_factor *= -1

        self.bid_price = self.actual_cost * discount_factor

        # 判断是否参与竞争和是否竞争成功
        if not self.participation_status:
            if environment.demand > 500:
                self.participation_status = True
                self.compete_success = True
                environment.demand -= self.capacity
            elif environment.demand > 0:
                if self.capacity > environment.demand:
                    self.participation_status = True
                    self.compete_success = True
                    self.bid_price = self.actual_cost * discount_factor
                    environment.demand -= self.capacity
                else:
                    self.participation_status = True
                    self.compete_success = True
                    environment.demand -= self.capacity

class Environment:
    def __init__(self, demand, is_windy_season, is_valley, is_spring_festival, is_labor, is_qingming, date):
        self.demand = demand
        self.is_windy_season = is_windy_season
        self.is_valley = is_valley
        self.is_spring_festival = is_spring_festival
        self.is_labor = is_labor
        self.is_qingming = is_qingming
        self.date = date
        self.special_condition = any([is_windy_season, is_valley, is_spring_festival, is_labor, is_qingming])
        self.total_capacity = 0

    def is_special_day(self):
        return self.date.day in range(2, 6) or self.date.day in range(10, 16)

    def is_special_month(self):
        return self.date.month in [1, 2, 3, 4, 5, 9, 10, 11, 12]

    def is_festival(self):
        return self.is_spring_festival or self.is_qingming or self.is_labor

class ElectricityMarketModel:
    def __init__(self, electricity_price_df, unit_df):
        self.electricity_price_df = electricity_price_df
        self.unit_df = unit_df
        self.agents = []
        self.create_agents()
    
    def create_agents(self):
        for index, row in self.unit_df.iterrows():
            agent = PowerPlantAgent(index, row['Capacity(MW)'], row['actual'])
            self.agents.append(agent)
    
    def run_simulation(self):
        results = []
        # 用于保存每次竞争的详细信息
        competition_details = []
        
        for index, row in self.electricity_price_df.iterrows():
            environment = Environment(
                row['demand'],
                row['is_windy_season'],
                row['is_valley'],
                row['is_spring_festival'],
                row['is_labor'],
                row['is_qingming'],
                pd.to_datetime(row.name)  # 使用索引名称作为日期
            )

            for agent in self.agents:
                agent.update_bid(environment)

            # 获取所有成功竞争的代理，按报价排序
            successful_agents = sorted([agent for agent in self.agents if agent.compete_success], key=lambda x: x.bid_price)
            clearing_price = successful_agents[-1].bid_price if successful_agents else np.nan
            results.append(clearing_price)

            # 记录每个发电厂的竞标价格
            competition_details.append({
                'time': row.name,
                'demand': row['demand'],
                'clearing_price': clearing_price,
                'agents': [(agent.agent_id, agent.bid_price, agent.compete_success) for agent in self.agents]
            })

        self.electricity_price_df['predicted_price'] = results
        self.competition_details = competition_details  # 保存竞争细节供后续可视化使用

# 运行模型
model = ElectricityMarketModel(train_data, unit_df)
model.run_simulation()

# 可视化某一次竞争的图
def visualize_competition(competition_details, timestamp):
    # 查找特定时间点的竞争详情
    details = next((item for item in competition_details if item['time'] == timestamp), None)
    if not details:
        print(f"No competition details found for timestamp: {timestamp}")
        return

    agents = details['agents']
    agent_ids, bid_prices, compete_success = zip(*agents)

    plt.figure(figsize=(12, 6))
    plt.bar(agent_ids, bid_prices, color=['green' if success else 'red' for success in compete_success])
    plt.xlabel('Power Plant ID')
    plt.ylabel('Bid Price')
    plt.title(f'Competition at {timestamp}')
    plt.xticks(rotation=90)
    plt.show()

# 可视化某个时间点的竞争情况
visualize_competition(model.competition_details, pd.Timestamp('2024-01-01 10:00:00'))
