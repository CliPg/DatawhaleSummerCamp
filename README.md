# DatawhaleAI夏令营笔记

## Task1

### 问题分析

#### 市场出清机制

##### 基本内容

收集买家和卖家的报价信息并排序，从最高买价和最低卖价进行匹配，直至到达**平衡**

##### 目的

确定交易物品的**价格**和**数量**

##### 名词解释

**市场出清价格：** 平衡点对应的价格



#### 任务

- 期待使用**Agent-based model**预测未来的市场出清价格



#### 数据集分析

| 变量名                        | 含义                                                   |
| ----------------------------- | ------------------------------------------------------ |
| demand                        | 需电量                                                 |
| clearing price (CNY/MWh)      | 市场出清价格(元/兆瓦时)                                |
| unit ID                       | 构造出的发电机组的ID                                   |
| Capacity(MW)                  | 装机容量                                               |
| utilization hour (h)          | 电厂的年平均运行小时数                                 |
| coal consumption (g coal/KWh) | 供电煤耗                                               |
| power consumption rate (%)    | 发电厂的利用率，指电厂单位时间内耗电量与发电量的百分比 |





#### 评价指标

```
c=（MSE+RMSE）/2
```



#### 提交格式

```Plain
submit.csv/submit.xlsx
day,time,clearing price （CNY/MWh）
2024/4/1 , 0:15 , 352.3340 
2024/4/1 , 0:30 , 355.5360
```



### 魔搭的基本使用

- [魔搭链接](https://modelscope.cn/my/mynotebook/preset)
- 启动Notebook
- 导入并运行代码





## Task2

### EDA

#### 含义

**探索性数据分析**，构造**强特征**

#### 基础统计指标

- 数据的集聚趋势
  - 均值：`df[feature].mean()`
  - 中位数 `df[feature].median()`
  - 最大值 `df[feature].max()`
  - 最小值 `df[feature].min()`
  - 众数 `df[feature].mode()`
- 数据的变异程度
  - 标准差 `df[feature].std()`
  - 极差 `df[feature].apply(lambda x: x.max() - x.min())`
  - 四分位数 `df[feature].quantile([0.25, 0.5, 0.75])`
  - 变异系数 `df[feature].std()/df[feature].mean()`
  - 偏度和峰度 `df[feature].skew()`, `df[feature].kurtosis()`



分析数据时，可以先从整体观察分析（绘制变化曲线），观察是否有异常点，异常点通常是解决问题的关键，以便进一步分析。

![img](blob:https://datawhaler.feishu.cn/da7919eb-4175-4041-a081-94ef023948eb)







