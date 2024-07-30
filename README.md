# DatawhaleAI夏令营笔记

## Intro

希望在task的指引下，从零开始建模，编写代码，并记录下自己的思路

库中含建模中用过的代码



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
| demand                        | 需电量（兆瓦）                                         |
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

编写代码并绘制图像发现：

![fig1](./images/Figure_1.png)

上图为各个需电量区间对应的数量，发现基本符合正态分布



![fig2](./images/Figure_2.png)

上图为各个出清价格区间对应的数量，发现存在负的出清价格，并具有较高的比例，除了该部分基本符合正态分布

为什么会存在负出清价格？	————	①





#### 分时统计指标

![fig4](./images/Figure_4.png)

(2021-2024一天各时段的平均需电量)

近三年需电量明显低于2021年，为什么？	————	②

图像大致呈M形	————	③



![fig5](./images/Figure_5.png)

(2021-2024各月份的平均需电量)

图像大致呈W形	————	④



![fig6](./images/Figure_6.png)

（2021-2023一天各时段的平均出清价格)

图像大致呈M形	————	⑤

21年低谷时更低，高峰时更高，22、23年数据相近	————	⑥



![fig7](./images/Figure_7.png)

（2021-2023各月份的平均出清价格)

图像大致呈倒V形	————	⑦



### 数据分析

①负出清价格

- 什么时候出现负出清价格？

  ![fig8](./images/Figure_8.png)

  一天中在凌晨三点和中午十二点左右常出现负出清价格

  

  ![fig9](./images/Figure_9.png)

  一年中在夏秋季节出现较少，春冬季节较多

- 为什么会出现负出清价格？

  
