# 2024/10/29 19:11:41  zt
# Calculate chemical equilibrium state of a premixed ethylene-air flame

import cantera as ct
import numpy as np

# Gas composition
gas = ct.Solution('./chem.yaml')

phi = 0.9

# 设定混合气体状态
## 温度和压力
gas.TP = 300.0, ct.one_atm
## 当量比
gas.set_equivalence_ratio(phi, 'C2H4', 'O2:1, N2:3.76')

# 计算平衡态
# HP 表示焓和压力不变
# 采用吉布斯自由能最小化法（收敛慢但精度高）
gas.equilibrate('HP', solver='gibbs')

# 输出结果
thres = 1.0e-6 # 组分浓度的阈值
print('Equilibrium state:')
print(' Temperature: {0:.1f} K'.format(gas.T))
for name in gas.species_names:
    Y = gas[name].Y[0]
    if Y > thres:
        #保留有效数字8位
        print(" Species: Y {0:5s} = {1:10.8f}".format(name,Y))

