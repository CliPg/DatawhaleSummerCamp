import numpy as np
import pandas as pd
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
        self.compete_success = False
    
    def update_bid(self, environment):
        if environment.demand > 0:
            discount_factor = 1.0

            # 计算满足条件的数量
            if environment.special_condition:
                if environment.is_windy_season:
                    discount_factor *= 0.95  # 满足条件一
                if environment.is_valley:
                    discount_factor *= 0.95  # 满足条件二
                if environment.is_spring_festival:
                    discount_factor *= 0.9   # 满足条件三
                if environment.is_labor:
                    discount_factor *= 0.9   # 满足条件四
                if environment.is_qingming:
                    discount_factor *= 0.9   # 满足条件五

            # 根据条件的数量调整报价
            if environment.total_capacity < environment.demand / 2:
                if not self.compete_success:
                    self.bid_price = self.actual_cost * discount_factor
                    self.compete_success = True
                    environment.demand -= self.capacity
                    environment.total_capacity += self.capacity
            elif self.capacity > environment.demand:
                if not self.compete_success:
                    self.bid_price = self.actual_cost * discount_factor
                    self.compete_success = True
                    environment.demand -= self.capacity
                    environment.total_capacity += self.capacity
            else:
                if not self.compete_success:
                    self.bid_price = self.actual_cost * discount_factor
                    self.compete_success = True
                    environment.demand -= self.capacity
                    environment.total_capacity += self.capacity

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
    
    def is_special_condition(self, row):
        return row['is_windy_season'] or row['is_valley'] or row['is_spring_festival'] or row['is_labor'] or row['is_qingming']
    
    def run_simulation(self):
        results = []
        for index, row in self.electricity_price_df.iterrows():
            if 'is_spring_festival' not in row or 'is_labor' not in row or 'is_qingming' not in row:
                print(f"Missing column in row {index}")
                continue
            environment = Environment(
                row['demand'],
                row['is_windy_season'],
                row['is_valley'],
                row['is_spring_festival'],
                row['is_labor'],
                row['is_qingming'],
            )
            for agent in self.agents:
                agent.update_bid(environment)
            successful_bids = sorted([agent.bid_price for agent in self.agents if agent.compete_success])
            clearing_price = successful_bids[-1] if successful_bids else np.nan
            results.append(clearing_price)
        self.electricity_price_df['predicted_price'] = results

class Environment:
    def __init__(self, demand, is_windy_season, is_valley, is_spring_festival, is_labor, is_qingming):
        self.demand = demand
        self.is_windy_season = is_windy_season
        self.is_valley = is_valley
        self.is_spring_festival = is_spring_festival
        self.is_labor = is_labor
        self.is_qingming = is_qingming
        self.special_condition = any([is_windy_season, is_valley, is_spring_festival, is_labor, is_qingming])
        self.total_capacity = 0

    def is_special_condition(self):
        return self.special_condition

# 运行模型
model = ElectricityMarketModel(train_data, unit_df)
model.run_simulation()

# 保存预测结果
train_data['predicted_price'] = model.electricity_price_df['predicted_price']

# 读取保存的ABM模型预测结果和实际价格
actual_prices = pd.read_csv('datas/electricity_price_parsed.csv', parse_dates=["timestamp"], index_col=0)

# 合并数据
data = train_data.merge(actual_prices[['clearing price (CNY/MWh)']], left_index=True, right_index=True, suffixes=('_pred', '_actual'))


# 分离特征和目标变量
X = data[['predicted_price']]
y = data['clearing price (CNY/MWh)_actual']


predicted_price_non_null_count = data['predicted_price'].notnull().sum()
clearing_price_actual_non_null_count = data['clearing price (CNY/MWh)_actual'].notnull().sum()

print(f"Number of non-null rows in 'predicted_price': {predicted_price_non_null_count}")
print(f"Number of non-null rows in 'clearing price (CNY/MWh)_actual': {clearing_price_actual_non_null_count}")


predicted_price_total_count = len(data['predicted_price'])
clearing_price_actual_total_count = len(data['clearing price (CNY/MWh)_actual'])

print(f"Total number of rows in 'predicted_price': {predicted_price_total_count}")
print(f"Total number of rows in 'clearing price (CNY/MWh)_actual': {clearing_price_actual_total_count}")


# 拆分训练集和测试集

train_size = 55392
X_train = data.iloc[:train_size][['predicted_price']]
y_train = data.iloc[:train_size]['clearing price (CNY/MWh)_actual']
X_test = data.iloc[train_size:][['predicted_price']]
#y_test = data.iloc[train_size:]['clearing price (CNY/MWh)_actual']

# 线性回归模型
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
#y_pred_lin = lin_reg.predict(X_test)

# LightGBM模型
lgb_reg = LGBMRegressor(num_leaves=2**5-1, n_estimators=300, verbose=-1)
lgb_reg.fit(X_train, y_train)
#y_pred_lgb = lgb_reg.predict(X_test)



# 将预测结果保存到数据框中
data['average'] = (lin_reg.predict(X) + lgb_reg.predict(X))/2

# 保存结果
data.to_csv('datas/submit6.csv', index=False)


