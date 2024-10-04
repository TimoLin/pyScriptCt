#!/usr/bin/env python
# coding: utf-8

# <div style="color:black; background-color:#FFF3E9; border: 1px solid #FFE0C3; border-radius: 10px; margin-bottom:1rem">
#     <p style="margin:1rem; padding-left: 1rem; line-height: 2.5;">
#         ©️ <b><i>Copyright 2023 @ Authors</i></b><br/>
#         作者：<a style="font-weight:bold" href="mailto:zhangyz@dp.tech">蔡雨青 📨 </a><br>
#         日期：2023-09-1<br/>
#         共享协议：本作品采用<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">知识共享署名-非商业性使用-相同方式共享 4.0 国际许可协议</a>进行许可。</i><br/>
#         快速开始：点击上方的 <span style="background-color:rgb(85, 91, 228); color:white; padding: 3px; border-radius: 5px;box-shadow: 2px 2px 3px rgba(0, 0, 0, 0.3); font-size:0.75rem;">开始连接</span> 按钮，选择 <span style='color:rgb(85,91,228); font-weight:bold'>deepflame-realfluid:wzf2.0</span> 镜像及 <span style='color:rgb(85,91,228); font-weight:bold'>c8_m16_cpu</span> 节点配置，稍等片刻即可运行。
#     </p>
# </div>
# 

# # FPV火焰面模型在部分预混火焰中的应用

# ## 一、背景
# 为了实现高精度高保真的湍流燃烧过程，研究人员发展了多种高精度湍流燃烧模型以实现准确湍流-化学相互作用的捕捉。其中，基于有限化学反应速率的t-PDF和CMC是公认的普适方法。然而，有限化学反应速率直接求解化学ODE，而化学ODE的求解需要耗费大量的时间，例如求解一个组分为N的燃烧机理，一般求解化学ODE需要两个步骤——求解雅可比矩阵和线性方程组，前者计算复杂度是$N^{2}$，后者是$N^{3}$。因此，随着机理组分的提升，计算资源呈几何增加，这是我们无法接受的。
# 那有没有一种既能保持计算精度，又能提高计算效率的方法呢？答案是有的，火焰面模型就是其中的佼佼者。简单介绍一下，火焰面模型最初发展的契机。有一类非常经典的燃烧方式——预混火焰传播过程，下面是一条典型的一维预混火焰传播过程的温度、组分分布图。
# 

# - 一维预混火焰分布

# <img src="https://bohrium.oss-cn-zhangjiakou.aliyuncs.com/article/19016/11fb0e8b45d647d4a6a767bf2a57b4ea/yK3miE28I6idyKVACKCFgg/89Cx6tUY9VzTJon8YLl2LA.png" width="300" height="200" alt="一维预混火焰分布">
# 
# 

# 从图中可以看出，在火焰传播过程中，火焰面上的组分和温度分布不会随着火焰面位置的变动而变化，从物理的角度分析，这是因为火焰面是一个温度梯度非常大且反应非常彻底的快速反应区。但从结果上分析，这种结构表明，实际存在一种基于火焰面的手段，提炼出基础的火焰结构（相同火焰结构有相同的温度、组分分布），将物理空间的多维场降维到若干个火焰面标量场。这就是火焰面模型的由来。
# 这篇文章基于deepFlame的flareFGM一维火焰面建表模型，利用cantera的CounterflowDiffusionFlame接口，发展了Flamelet-Progress-Variable（FPV）燃烧模型，相比于deepFlame的flareFGM（部分预混火焰面模型，预混自由火焰传播），FPV是基于对冲火焰的建表方法，可以考虑来流速度引起的标量耗散率的变化，在计算非预混燃烧过程，FPV建表更加符合物理，下面开始对代码进行解析

# ## 二、代码架构
# FPV一维建表代码流程如下所示，其中Flare为flareFGM的实现流程，Flare_FPV为本文发展的FPV建表流程
# 
# 

# - 火焰面建表流程

# <img src="https://bohrium.oss-cn-zhangjiakou.aliyuncs.com/article/19016/11fb0e8b45d647d4a6a767bf2a57b4ea/mD3lYutWZJNTNQkwNR5TfA/g3ePVQhliyVs_Ir1l5ZlHg.jpeg" width="300" height="200" alt="流程图">
# 
# 
# 

# 各个模块的代码结构
# - 读取参数文件——read_commonDict
# - 一维对冲扩散火焰数值模拟——multiProcessing_canteraSim
# - 对Z，C做曲面建表法——normal_table
# - 二维插值——interpToMeshgrid
# - PDF积分——PDF_Sim
# - 组装表格——assemble
# 
# 下面给出各个模块的代码以及相关注释

# 安装cantera包
# In[ ]:
# In[ ]:

# ## 2.1 read_commonDict功能解析
# 在进行层流火焰面计算之前，我们需要读取一些必要的数据作为计算的输入。下面的表格汇总了所有需要用到的数据
# | 参数 | 功能 | 
# | :-----| ----: | 
# | chemMech | 机理文件的路径，一般提供yaml文件 |
# | n_points_z | 层流火焰面制表中，混合分数Z的分区数 |
# | n_points_c | 层流火焰面制表中，进度变量C的分区数|
# | f_max | 最大当量比 |
# | int_pts_z| 湍流火焰面制表，混合分数Z的分区数|
# | int_pts_c| 湍流火焰面制表，进度变量C的分区数|
# | int_pts_gz| 湍流火焰面制表，Z方差的分区数|
# | int_pts_gc| 湍流火焰面制表，C方差的分区数|
# | int_pts_gzc| 湍流火焰面制表，ZC的协同方差的分区数|

# In[ ]:


# read_commonDict
def read_commonDict(commonDict_path):
    import numpy as np
    import cantera as ct
    import ast

    #---------read commonDict.txt--------#
    with open(commonDict_path,"r") as f:
        commonDict = f.read()
        f.close()
    print(commonDict)
    cbDict = ast.literal_eval(commonDict)

    #Total number of species in all phases participating in the kinetics mechanism
    cbDict['nSpeMech'] = ct.Solution(cbDict['chemMech']).n_total_species

    #----------create meshgrid------------#
    z_space = np.linspace(0,1,cbDict['n_points_z']) 
    nn = int(cbDict['n_points_z']/20*19)
    for i in range(nn):
        z_space[i] = cbDict['f_max']*1.2/(nn-1)*i
    z_space[nn-1:] = np.linspace(cbDict['f_max']*1.2,1.0,
                                cbDict['n_points_z']-nn+1) 

    c_space = np.linspace(0,1,cbDict['n_points_c'])

    cbDict['z_space'] = z_space
    cbDict['c_space'] = c_space

    #----------create manifold------------#
    z_int = np.linspace(0,1,cbDict['int_pts_z']) 
    nn = int(cbDict['int_pts_z']/20*19)            
    for i in range(nn):
        z_int[i] = cbDict['f_max']*1.2/(nn-1)*i
    z_int[nn-1:] = np.linspace(cbDict['f_max']*1.2,1.0,
                                cbDict['int_pts_z']-nn+1)

    c_int = np.linspace(0,1,cbDict['int_pts_c'])  

    gz_int = np.zeros(cbDict['int_pts_gz'])
    gz_int[1:] = np.logspace(-4,-1,cbDict['int_pts_gz']-1)  
    # gz_int[1:] = np.logspace(-4,0,cbDict['int_pts_gz']-1)  

    gc_int = np.linspace(0,1-cbDict['small'],cbDict['int_pts_gc'])   
    # gc_int = np.linspace(0,1,cbDict['int_pts_gc'])   

    # gcor_int = np.linspace(-1,1,cbDict['int_pts_gcor']) 
    if (cbDict['int_pts_gcz'] <= 1):
        gcz_int = np.linspace(0,0,cbDict['int_pts_gcz'])
    else:
        gcz_int = np.linspace(-1,1,cbDict['int_pts_gcz'])   

    #save to cbDict
    cbDict['z'] = z_int
    cbDict['c'] = c_int
    cbDict['gz'] = gz_int
    cbDict['gc'] = gc_int
    # cbDict['gcor'] = gcor_int
    cbDict['gcz'] = gcz_int

    return cbDict


# ## 2.2 multiProcessing_canteraSim
# 完成了重要参数的读写，下面可以开展最重要的一维层流对冲火焰计算。我们采用cantera提供的CounterflowDiffusionFlame接口开展对冲火焰的计算。基于FPV和UFPV理论，拉伸率（标量耗散率）是影响火焰计算的一个重要参数，因为如果火焰的对冲速率过快，有可能没来得及反应，活性燃料就被吹散了，所以需要对这种对冲扩散燃烧进行准确的量化。下面给出FPV和UFPV的燃烧曲线。
# - FPV的稳态火焰
# 
# ![alt image.png](https://bohrium.oss-cn-zhangjiakou.aliyuncs.com/article/19016/cf092783767948c0be59d010620580fe/-e_kDGMbjG_0zNvWpkaojw.png)
# - UFPV的非稳态火焰
# ![alt image.png](https://bohrium.oss-cn-zhangjiakou.aliyuncs.com/article/19016/06f80c46ac264c8cb011effe79313c44/oQJPRX5wvNK4wkPKYZO5yw.png)
# 可以发现燃烧的S型曲线中间的那条是非稳态分支。但是总的来说，标量耗散率是层流火焰中不可或缺的一个指标，但是一维自由火焰传播无法评估这个指标，所以需要引入FPV。

# - 在开展FPV计算前，首先针对来流速度做不同标量耗散率下的火焰曲线，计算熄火极限。下面给出了熄火极限的计算函数

# In[ ]:


# 用来计算最大拉伸率，找到熄火极限
import ctypes
from ntpath import join
import cantera as ct
import numpy as np
import sys
import os
from multiprocessing import Pool,Array
import importlib
import matplotlib.pyplot as plt
import math
# import time
# import zipfile,tarfile
def cal_extinction(mdot_fuel, mdot_ox, fuel_species, CASENAME, chemMech, transModel, nSpeMech, nVarCant, \
        p, Lx, stoich_O2, T_fuel, X_fuel, T_ox, X_ox, solFln):
    
    hdf_output = importlib.util.find_spec('h5py') is not None

    if not hdf_output:
        # Create directory for output data files
        data_directory = 'diffusion_flame_extinction_data'
        if not os.path.exists(data_directory):
            os.makedirs(data_directory)
    # PART 1: INITIALIZATION
    gas = ct.Solution(chemMech)
    gas.TP = gas.T,p
    #width = Lx  # input wide
    f = ct.CounterflowDiffusionFlame(gas, width=Lx)

    # Define the operating pressure and boundary conditions
    f.P = p
    f.fuel_inlet.mdot = mdot_fuel  # kg/m^2/s
    f.fuel_inlet.X = X_fuel
    f.fuel_inlet.T = T_fuel  # K
    f.oxidizer_inlet.mdot = mdot_ox  # kg/m^2/s
    f.oxidizer_inlet.X = X_ox
    f.oxidizer_inlet.T = T_ox  # K

    # Set refinement parameters
    f.set_refine_criteria(ratio=3.0, slope=0.1, curve=0.2, prune=0.03)

    # Define a limit for the maximum temperature below which the flame is
    # considered as extinguished and the computation is aborted
    temperature_limit_extinction = max(f.oxidizer_inlet.T, f.fuel_inlet.T)

    # Initialize and solve
    print('Creating the initial solution')
    f.solve(loglevel=0, auto=True)

    if hdf_output:
        file_name = 'diffusion_flame_extinction.h5'
        f.write_hdf(file_name, group='initial_solution', mode='w', quiet=False,
                    description=('Initial solution'))
    else:
        # Save to data directory
        file_name = 'initial_solution.yaml'
        f.save(os.path.join(data_directory, file_name), name='solution',
                    description="Initial solution")
    
    # PART 2: COMPUTE EXTINCTION STRAIN

    # Exponents for the initial solution variation with changes in strain rate
    # Taken from Fiala and Sattelmayer (2014)
    exp_d_strain = - 1. / 2.
    exp_u_strain = 1. / 2.
    exp_V_strain = 1.
    exp_lam_strain = 2.
    exp_mdot_strain = 1. / 2.

    # Set normalized initial strain rate
    relative_strain = [1.]
    # Initial relative strain rate increase
    delta_strain = 1.
    # Factor of refinement of the strain rate increase
    delta_strain_factor = 50.
    # Limit of the refinement: Minimum normalized strain rate increase
    delta_strain_min = .001
    # Limit of the Temperature decrease
    delta_T_min = 1  # K

    # Iteration indicator
    n = 0
    # Indicator of the latest flame still burning
    n_last_burning = 0
    # List of peak temperatures
    T_max = [np.max(f.T)]
    # List of maximum axial velocity gradients
    a_max = [np.max(np.abs(np.gradient(f.velocity) / np.gradient(f.grid)))]

    # Simulate counterflow flames at increasing strain rates until the flame is
    # extinguished. To achieve a fast simulation, an initial coarse strain rate
    # increase is set. This increase is reduced after an extinction event and
    # the simulation is again started based on the last burning solution.
    # The extinction point is considered to be reached if the abortion criteria
    # on strain rate increase and peak temperature decrease are fulfilled.
    while True:
        n += 1
        # Update relative strain rates
        relative_strain.append(relative_strain[n_last_burning] + delta_strain)
        strain_factor = relative_strain[-1] / relative_strain[n_last_burning]
        # Create an initial guess based on the previous solution
        # Update grid
        # Note that grid scaling changes the diffusion flame width
        f.flame.grid *= strain_factor ** exp_d_strain
        normalized_grid = f.grid / (f.grid[-1] - f.grid[0])
        # Update mass fluxes
        f.fuel_inlet.mdot *= strain_factor ** exp_mdot_strain
        f.oxidizer_inlet.mdot *= strain_factor ** exp_mdot_strain
        # Update velocities
        f.set_profile('velocity', normalized_grid,
                      f.velocity * strain_factor ** exp_u_strain)
        f.set_profile('spread_rate', normalized_grid,
                      f.spread_rate * strain_factor ** exp_V_strain)
        # Update pressure curvature
        f.set_profile('lambda', normalized_grid, f.L * strain_factor ** exp_lam_strain)
        try:
            f.solve(loglevel=0)
        except ct.CanteraError as e:
            print('Error: Did not converge at n =', n, e)
        if not np.isclose(np.max(f.T), temperature_limit_extinction):
            # Flame is still burning, so proceed to next strain rate
            n_last_burning = n
            if hdf_output:
                group = 'extinction/{0:04d}'.format(n)
                f.write_hdf(file_name, group=group, quiet=True)
            else:
                file_name = 'extinction_{0:04d}.yaml'.format(n)
                f.save(os.path.join(data_directory, file_name),
                       name='solution', loglevel=0,
                       description=f"Solution at alpha = {relative_strain[-1]}")
            T_max.append(np.max(f.T))
            a_max.append(np.max(np.abs(np.gradient(f.velocity) / np.gradient(f.grid))))
            print('Flame burning at alpha = {:8.4F}. Proceeding to the next iteration, '
                  'with delta_alpha = {}'.format(relative_strain[-1], delta_strain))
        elif ((T_max[-2] - T_max[-1] < delta_T_min) and (delta_strain < delta_strain_min)):
            # If the temperature difference is too small and the minimum relative
            # strain rate increase is reached, save the last, non-burning, solution
            # to the output file and break the loop
            T_max.append(np.max(f.T))
            a_max.append(np.max(np.abs(np.gradient(f.velocity) / np.gradient(f.grid))))
            if hdf_output:
                group = 'extinction/{0:04d}'.format(n)
                f.write_hdf(file_name, group=group, quiet=True)
            else:
                file_name = 'extinction_{0:04d}.yaml'.format(n)
                f.save(os.path.join(data_directory, file_name), name='solution', loglevel=0)
            print('Flame extinguished at alpha = {0:8.4F}.'.format(relative_strain[-1]),
                  'Abortion criterion satisfied.')
            break
        else:
            # Procedure if flame extinguished but abortion criterion is not satisfied
            # Reduce relative strain rate increase
            delta_strain = delta_strain / delta_strain_factor

            print('Flame extinguished at alpha = {0:8.4F}. Restoring alpha = {1:8.4F} and '
                  'trying delta_alpha = {2}'.format(relative_strain[-1], relative_strain[n_last_burning], delta_strain))
            # Restore last burning solution
            if hdf_output:
                group = 'extinction/{0:04d}'.format(n_last_burning)
                f.read_hdf(file_name, group=group)
            else:
                file_name = 'extinction_{0:04d}.yaml'.format(n_last_burning)
                f.restore(os.path.join(data_directory, file_name),
                          name='solution', loglevel=0)
            
    # save to cbDict
    #cbDict['origin_strain_factor'] = 1.0

    # Plot the maximum temperature over the maximum axial velocity gradient
    plt.figure()
    plt.semilogx(a_max, T_max)
    plt.xlabel(r'$max_strain$ [1/s]')
    plt.ylabel(r'$max_T$ [K]')
    if hdf_output:
        plt.savefig('diffusion_flame_extinction_T_max_a_max.png')
    else:
        plt.savefig(os.path.join(data_directory, 'figure_T_max_a_max.png'))
    return relative_strain[-1]*0.9


# - 计算层流火焰比热容函数

# In[ ]:


# multiProcessing_canteraSim
import ctypes
from ntpath import join
import cantera as ct
import numpy as np
import sys
import os
from multiprocessing import Pool,Array
import importlib
import matplotlib.pyplot as plt
import math
# import time
# import zipfile,tarfile

    
def cal_cpe(reactants, p, T, chemMech):
    gas_cp = ct.Solution(chemMech) # 创建反应系统
    gas_cp.TPX = 298.15,p,reactants
    cp_0 = gas_cp.cp_mass # 定压比热容
    if abs(T - 298.15) < 0.1: # 25摄氏度左右的话
        cpe = 0.
    else: # 远离25摄氏度
        sum_CpdT = 0.
        dT = (T-298.15)/(int(100*abs(T-298.15))-1) # 温度变化率
        for kk in range(1,int(100*abs(T-298.15))): # 扩大100倍的zone遍历
            gas_cp.TPX = (298.15 + kk*dT),p,reactants # 重新给定反应系统的温度
            cp_1 = gas_cp.cp_mass
            sum_CpdT = sum_CpdT + 0.5*(cp_0 + cp_1)*dT # 梯形积分
            cp_0 = cp_1
        cpe = sum_CpdT     # 最终得到了cp和T的映射函数的面积
    return cpe
    


# - 开始针对层流对冲火焰，开展计算

# In[ ]:


# 这是做计算的函数，拉伸率需要先算一下最大的熄火极限，确定拉伸率的最大值，也就是说拉伸率的最大值是计算出来，而非自我设定
def Sim(i, mdot_fuel, mdot_ox, strain_factor, fuel_species, CASENAME, chemMech, transModel,\
        nSpeMech, nVarCant,p, Lx, stoich_O2, T_fuel, X_fuel, T_ox, X_ox, solFln):
    # 若干个参数的含义
    # i 要做的当量比计算的序号
    # mdot_fuel 燃油的流量
    # modt_ox 氧化剂的流量
    # strain_factor 若干个拉伸率的缩放因子
    # fuel_species 燃料组分
    # CASENAME 算例名称
    # chemMech 燃料机理
    # transModel 输运模型
    # nSpeMech 机理的组分数
    # nVarCant 反应进度、反应速率、温度、密度、定压比热容、总质量、生成焓、运动粘度、焓、释热率

    # reactants = {fuel_species: phi[i] / stoich_O2, 'O2': 1.0, 'N2': 3.76} # 在当量燃烧条件下，各个组分的浓度

    # Exponents for the initial solution variation with changes in strain rate
    # Taken from Fiala and Sattelmayer (2014)
    exp_d_strain = - 1. / 2.
    exp_u_strain = 1. / 2.
    exp_V_strain = 1.
    exp_lam_strain = 2.
    exp_mdot_strain = 1. / 2.

    ## Load chemical mechanism
    gas = ct.Solution(chemMech) # 加载燃烧机理
    gas.TP = gas.T, p;

    # Flame object
    f = ct.CounterflowDiffusionFlame(gas, width=Lx)
    
    # PART 1: INITIALIZATION
    gas = ct.Solution(chemMech)
    gas.TP = gas.T,p
    #width = Lx  # input wide
    f = ct.CounterflowDiffusionFlame(gas, width=Lx)

    # Define the operating pressure and boundary conditions
    f.P = p
    f.fuel_inlet.mdot = mdot_fuel  # kg/m^2/s
    f.fuel_inlet.X = X_fuel
    f.fuel_inlet.T = T_fuel  # K
    f.oxidizer_inlet.mdot = mdot_ox  # kg/m^2/s
    f.oxidizer_inlet.X = X_ox
    f.oxidizer_inlet.T = T_ox  # K

    # Set refinement parameters
    #f.set_refine_criteria(ratio=3, slope=0.07, curve=0.14, prune=0.03)

    # Initialize and solve
    #print('Creating the initial solution')
    #f.solve(loglevel=0, auto=True)
    
    # 加入选用的拉伸率
    print("factor: ", strain_factor[i])
    f.flame.grid *= strain_factor[i] ** exp_d_strain
    normalized_grid = f.grid / (f.grid[-1] - f.grid[0])
    # Update mass fluxes
    f.fuel_inlet.mdot *= strain_factor[i] ** exp_mdot_strain
    f.oxidizer_inlet.mdot *= strain_factor[i] ** exp_mdot_strain
    # Update velocities
    f.set_profile('velocity', normalized_grid, f.velocity * strain_factor[i] ** exp_u_strain)
    f.set_profile('spread_rate', normalized_grid, f.spread_rate * strain_factor[i] ** exp_V_strain)
    # Update pressure curvature
    f.set_profile('lambda', normalized_grid, f.L * strain_factor[i] ** exp_lam_strain)

    if transModel == 'Mix':    
        print("run on first branch")
        
        #Mix：不同组分的分子输运参数平均后计算；Mixture：不同组分用自己的分子输运参数
        # Solve with the energy equation disabled
        f.energy_enabled = False # 关掉能量方程
        f.transport_model = transModel
        f.solve(loglevel=1, refine_grid=False)

        # Solve with the energy equation enabled
        f.set_refine_criteria(ratio=3, slope=0.07, curve=0.14)
        f.energy_enabled = True
        f.solve(loglevel=1,auto=True)
        print('mixture-averaged flamespeed = {0:7f} m/s'.format(f.velocity[0]))   
        #为什么有无能量方程都求一遍？

    elif transModel == 'Multi':
        # Solve with multi-component transport properties
        f.transport_model = 'Multi'
        f.solve(loglevel=1, auto=True)
        print('multicomponent flamespeed = {0:7f} m/s'.format(f.velocity[0]))     
        #transModel == 'Multi'为什么还要再求一遍？

    else:
        sys.exit(' !! Error - incorrect transport model specified !!')
    print("max strain: ",f.strain_rate('max'))
    sL.append(f.velocity[0]) # 第一个是最终的层流火焰速度？收敛后的
    # sL.append(f.u[0]) # store flame speed in array sL

    # store useful data for future simulation
    iCO = gas.species_index('CO') # 一氧化碳的组分编号
    iCO2 = gas.species_index('CO2') # 二氧化碳的组分编号
    #iHO2 = gas.species_index('HO2')
    #iCH2O = gas.species_index('CH2O')
    iH2O = gas.species_index('H2O')
    iH2 = gas.species_index('H2')




    data = np.zeros((len(f.grid),nSpeMech+nVarCant+2))  # 设置一个序列，行数是网格的数量，列数是机理的组分数和是个重要的参数（反应进度、反应速率、温度、密度、定压比热容、总质量、生成焓、运动粘度、焓、释热率），应该添加一个混合分数，放到最后一个吧

    # unscaled progress variable，未归一化的进度变量，进度变量的定义有点古老。
    #data第0列：CO和CO2的质量分数之和，表示反应进度，更改一下
    data[:,0] = f.Y[iCO] + f.Y[iCO2] + f.Y[iH2O] # 首列是进度变量      
    #data[:,0] = 0.9*f.Y[iCO] + 1.2*f.Y[iCO2] + 2.7*f.Y[iHO2] + 1.2*f.Y[iH2O] + 1.5*f.Y[iCH2O] # 首列是进度变量     

    # Reaction rate of unscaled progress variable
    #data编号为1的序列：CO摩尔质量*CO净生成率+CO2摩尔质量*CO2净生成率，表示反应速率，为什么表示反应速率？
    # 答：把摩尔反应速率变成质量的反应速率
    data[:,1] = f.gas.molecular_weights[iCO]*f.net_production_rates[iCO,:] \
              + f.gas.molecular_weights[iCO2]*f.net_production_rates[iCO2,:] \
              + f.gas.molecular_weights[iH2O]*f.net_production_rates[iH2O,:]
        #0.9*f.gas.molecular_weights[iCO]*f.net_production_rates[iCO,:] \
        #+ 1.2*f.gas.molecular_weights[iCO2]*f.net_production_rates[iCO2,:] \
        #+ 2.7*f.gas.molecular_weights[iHO2]*f.net_production_rates[iHO2,:] \
        #+ 1.2*f.gas.molecular_weights[iH2O]*f.net_production_rates[iH2O,:] \
        #+ 1.5*f.gas.molecular_weights[iCH2O]*f.net_production_rates[iCH2O,:]
                   

    data[:,2] = f.T       #data第2列：温度
    data[:,3] = f.density_mass   #data第3列：密度
    data[:,4] = f.cp_mass    #data第4列：定压热容,这个得更新
    #data第5列：摩尔分数转置后与摩尔质量的点积，即质量，但这个质量应该还是有问题的
    data[:,5] = np.dot(np.transpose(f.X),f.gas.molecular_weights)   

    # formation enthalpy 生成焓
    for j in range(len(f.grid)): # 遍历所有网格，应该是要把每个网格的各种信息算出来
        dumGas = ct.Solution(chemMech) # dummy working variable
        dumGas.TPY = 298.15,p,f.Y[:,j]
        data[j,6] = dumGas.enthalpy_mass   

    data[:,7] = f.viscosity/f.density   #data第7列：运动粘度=动力粘度/密度
    data[:,8] = f.enthalpy_mass         #data第8列：焓
    data[:,9] = f.heat_release_rate     #data第9列：释热率
    data[:,10] = f.mixture_fraction('Bilger') #data第10列：混合分数
    data[:,11] = f.strain_rate('max') # data第11列，这个case的最大拉伸率

    data[:,nVarCant+2:nSpeMech+nVarCant+2] = np.transpose(f.Y) # 最后这里是所有网格的质量分数 
            #data最后nSpeMech个列：组分质量分数矩阵的转置
        
    # 更新比热容到有效等压比热容,采用原始方法，一个个的更新    
    for ii in range(len(f.grid)):
        sum_cpdT = cal_cpe(f.X[:,ii], p, f.T[ii], chemMech)
        data[ii,4] = sum_cpdT/(f.T[ii]-298.15)
        
    # 到此，对冲扩散火焰的一些计算结果被记录了，下面是保存
        
    # save flamelet data
    fln = solFln + CASENAME + '_diff' + '_' + '{:03d}'.format(i) + '.csv' # 新创建一个文件的名字    
    np.savetxt(fln,data)                                        # 然后把它保存下来   

    ###### Compute global flame properties #######
    ###### calculate flame thickness
    # DT = np.gradient(f.T,f.grid)  #温度梯度（相对空间位置），没想到numpy居然还能直接求梯度，有点意思
    #dl = (f.T[-1]-f.T[0])/max(DT)   #火焰厚度，(末端温度-初始温度)/最大温度梯度，定义有点粗暴

    strain_tabi = np.frombuffer(strain_tab[i],dtype=ctypes.c_double) # 当量比、混合物分数、网格点数、火焰速度、火焰厚度、热释放参数，使用np.frombuffer是为了把strain_tab[i]的数据转变成类型，但是使用counterflowdiffusionflame后，有些内容不应该放在这里了，比如当量比，混合分数，

    strain_tabi[0] = 0; #phi[i]                      #equivalence ratio
    #strain_tabi[1] = 0; #Z[i]                        #mixture fraction
    strain_tabi[2] = 0; #len(f.grid)                 #number of grid points
    strain_tabi[3] = 0; #f.velocity[0]                      #flame speed
    strain_tabi[4] = 0; #dl                          #flame thickness
    strain_tabi[5] = 0; # (f.T[-1]-f.T[0])/f.T[0]     #heat release parameter，这个变量有意思，一般f.T[0]是混合后的温度，f[-1]是火焰末端温度，二者的差比能代表什么呢？
    
    #c = (f.Y[iCO,:] + f.Y[iCO2,:]) / max(f.Y[iCO,:] + f.Y[iCO2,:])  #相对反应进度，归一化的进度变量
    #alpha = f.thermal_conductivity/f.density/f.cp_mass      # 这也是一个向量，因为每个位置的热物理参数都是不一样的           
    # 热导率/密度/定压比热=单位时间单位面积下的变化
    # 热导率单位 W/m/K，W*m^-1*K^-1
    # 密度单位 kg/m^3,
    # 定压比热 J/kg/K, J*kg^-1*K^-1
    # 量纲计算 J*s^-1*m^-1*K^-1*m^3*kg^-1*K*kg*J^-1 = s^-1*m^2
    # 这个单位总感觉怪怪的，因为量纲没有温度，先保留，再看看
    
    #Dc = np.gradient(c,f.grid)   # 反应进度的空间梯度，一维向量
    #Nc = alpha*Dc*Dc             # 这里只是单纯的单元与单元的相乘                                      
    #PDF_c = Dc*f.viscosity/f.density/f.velocity[0]                            
    #integ_1 = np.trapz(f.density*Nc*np.gradient(f.velocity,f.grid)*PDF_c,c) # 横轴是进度变量，纵轴是很复杂
    #integ_2 = np.trapz(f.density*Nc*PDF_c,c)
    strain_tabi[6] = 0; #dl/f.velocity[0] *integ_1/integ_2/strain_tabi[5]   #KcStar   
    
    # 这段古怪的计算是求得全局变量phi_tab的第七个参数

    ###### calculate integral of cp in T space from 298.15 to Tin
    #gasCP = ct.Solution(chemMech) # 创建反应系统
    #gasCP.TPX = 298.15,p,reactants # 赋值反应系统
    #cp_0 = gasCP.cp_mass # 反应系统的定压比热
    #if abs(Tin - 298.15) < 0.1: # 25摄氏度左右的话
    #    phi_tabi[7] = 0.
    #else: # 远离25摄氏度
    #    sum_CpdT = 0.
    #    dT = (Tin-298.15)/(int(100*abs(Tin-298.15))-1) # 温度变化率
    #    for kk in range(1,int(100*abs(Tin-298.15))): # 扩大100倍的zone遍历
    #        gasCP.TPX = (298.15 + kk*dT),p,reactants # 重新给定反应系统的温度
    #        cp_1 = gasCP.cp_mass
    #        sum_CpdT = sum_CpdT + 0.5*(cp_0 + cp_1)*dT # 梯形积分
    #        cp_0 = cp_1
    #    strain_tabi[7] = sum_CpdT     # 最终得到了cp和T的映射函数的面积
    strain_tabi[7] = 0;    
    # 这段是求第八个，最后一个参数，概念清晰



def multi_canSim(item):
    i, mdot_fuel, mdot_ox, strain_factor, fuel_species, CASENAME, chemMech, transModel, \
        nSpeMech, nVarCant, p, Lx, stoich_O2, T_fuel, X_fuel, T_ox, X_ox, solFln = item
    Sim(i, mdot_fuel, mdot_ox, strain_factor, fuel_species, CASENAME, chemMech, transModel, \
        nSpeMech, nVarCant, p, Lx, stoich_O2, T_fuel, X_fuel, T_ox, X_ox, solFln) # 这个估计是真正做计算的函数

# 对外的接口，接收控制参数和地址，第一个是准备好的所有参数，第二个是信息保存的地址
# 引入拉伸率，拉伸率是一个维度，两端的流速由拉伸率控制
def canteraSim(cbDict,solFln):
    global strain_tab, sL # 两个全局变量，方便调用别的函数的时候做修改，类似C++的引用传递

    # solFln = (os.getcwd() + '/canteraData/')
    # solFln = ('./canteraData/')
    if not os.path.isdir(solFln): os.mkdir(solFln) # 如果没有，那么创建

    CASENAME = cbDict['CASENAME'] # case name
    p =  cbDict['p']  # pressure [Pa]
    Lx = cbDict['Lx'] # Domain size for the simulation [m]
    chemMech = cbDict['chemMech'] # chemical mechanism
    transModel = cbDict['transModel'] # cantera只提供了两种输运模型
    nSpeMech=cbDict['nSpeMech'] # 组分数量
    nVarCant=cbDict['nVarCant'] # 方差数量，反应进度、反应速率、温度、密度、定压比热容、总质量、生成焓、运动粘度、焓、释热率

    ## Fuel characteristics
    fuel_species = cbDict['fuel_species'] # Fuel is assumed to be of the form CxHy
    fuel_C = cbDict['fuel_C'] # number of C atoms in the fuel
    fuel_H = cbDict['fuel_H'] # number of H atoms in the fuel
    stoich_O2 = fuel_C+fuel_H/4. # DO NOT CHANGE - stoich air mole fraction
    W_fuel = fuel_C * 12. + fuel_H * 1.0 # DO NOT CHANGE - fuel molar weight
    T_fuel = cbDict['T_fuel'] # Fuel temperature [K]
    X_fuel = cbDict['X_fuel'] # Fuel composition (in mole fraction)

    ## Oxidiser characteristics
    W_O2 = 2. * 16. # DO NOT CHANGE - molar weight of O2
    W_N2 = 2. * 14. # DO NOT CHANGE - molar weight of N2
    T_ox = cbDict['T_ox'] # oxidiser temperature [K]
    X_ox = cbDict['X_ox'] # oxidiser composition (in mole fraction)

    ## Mixture properties
    # DO NOT CHANGE - stoichiometric mixture fraction
    Zst = (W_fuel) / (W_fuel + stoich_O2 * ( W_O2 + 3.76 * W_N2) )
    # DO NOT CHANGE - array of mixture fraction of interest
    # Z = np.linspace(cbDict['f_min'],cbDict['f_max'],cbDict['nchemfile'])  # 计算多少个混合分数和当量比
        
    ## 流量信息
    mdot_fuel = cbDict['mdot_fuel']
    mdot_ox = cbDict['mdot_ox']

    # 计算最大的拉伸率
    inputList = []
    inputList.append( ( mdot_fuel, mdot_ox, fuel_species, CASENAME, \
                        chemMech, transModel, nSpeMech, nVarCant, p, Lx, stoich_O2, \
                        T_fuel, X_fuel, T_ox, X_ox, solFln ) )
    
    # 是否计算了最大拉伸率
    cal_strain = cbDict['cal_strain']
    
    if not cal_strain:
        # cal_extinction(inputList)
        cbDict['max_strain_factor'] = cal_extinction( mdot_fuel, mdot_ox, fuel_species, CASENAME, \
                        chemMech, transModel, nSpeMech, nVarCant, p, Lx, stoich_O2, \
                        T_fuel, X_fuel, T_ox, X_ox, solFln )
    
    # 导入拉伸率的信息
    max_strain_factor = cbDict['max_strain_factor']
    origin_strain_factor = cbDict['origin_strain_factor']
    
    
    print("origin_strain_factor: ",origin_strain_factor)
    print("max_strain_factor: ",max_strain_factor)

    nstrains = cbDict['nstrains']
    normal_nstrains = cbDict['normal_nstrains']
    strain_factor = np.zeros( nstrains )
    delta_factor = cbDict['delta_factor']

    strain_factor[:normal_nstrains] = np.linspace(origin_strain_factor,max_strain_factor*0.9,normal_nstrains)
    strain_factor[normal_nstrains:] = np.linspace(max_strain_factor-delta_factor,max_strain_factor,nstrains-normal_nstrains)
    strain_factor[-1] = max_strain_factor*2
    # DO NOT CHANGE BELOW THIS LINE
    # phi = Z*(1.0 - Zst) / (Zst*(1.0 - Z))     #当量比phi，当量比？
    # phi_tab = np.zeros((len(phi),8)) #各列为：当量比、混合物分数、网格点数、火焰速度、火焰厚度、热释放参数
    strain_tab=[]
    sL = []     #层流火焰速度
    tmpz = np.linspace(cbDict['f_min'], cbDict['f_max'], cbDict['nchemfile'])
    for jj in range (cbDict['nchemfile']):
        strain_tab.append(Array('d',8,lock=False)) # 这里使用的Array代表共享数组，8代表共有8个数，没有用进程锁
        strain_tab[jj][1] = tmpz[jj]
    
    nscal_BC = cbDict['nscal_BC'] # 边界参数，案例是7
    BCdata = np.zeros((2,nscal_BC+2*cbDict['nYis'])) # 做一个2行，边界数+2*关心的组分数 

    fln_strain_tab = solFln + 'lamParameters.txt' # 保存的地址   
    #在work_dir/canteraData/solution_00文件夹中创建文件lamParameters.txt    

    inputList = []
    for i in range(cbDict['nstrains']):
        inputList.append( (i, mdot_fuel, mdot_ox, strain_factor, fuel_species, CASENAME, \
                           chemMech, transModel, nSpeMech, nVarCant, p, Lx, stoich_O2, \
                           T_fuel, X_fuel, T_ox, X_ox, solFln ) )
        #Sim(i, mdot_fuel, mdot_ox, strain_factor, fuel_species, CASENAME, \
        #                   chemMech, transModel, nSpeMech, nVarCant, p, Lx, stoich_O2, \
        #                   T_fuel, X_fuel, T_ox, X_ox, solFln)
    
    
    #for i in range(cbDict['nchemfile']): # 要计算多少个拉伸率
    #    #inputList.append( (i, Z, phi, fuel_species, CASENAME, \
    #    #        chemMech, transModel, nSpeMech, nVarCant, p, Lx, stoich_O2, \
    #    #        T_fuel, X_fuel, T_ox, X_ox, solFln ) )
    #    inputList.append( (i, mdot_fuel, mdot_ox, strain_factor, fuel_species, CASENAME, \
    #                       chemMech, transModel, nSpeMech, nVarCant, p, Lx, stoich_O2, \
    #                       T_fuel, X_fuel, T_ox, X_ox, solFln ) )

    # with Pool(processes=cbDict['n_procs']) as pool:
    #     pool.map(Sim,range(cbDict['nchemfile']))
    with Pool() as pool:
        pool.map(multi_canSim,inputList) # pool.map相当于把inputList的信息传递到了multi_canSim函数中，正式simulation也是这里进行，pool的作用是把任务自动地分配到空闲的进程中

    # 更新一下lamParmerters的混合分数
    #strain_tab[:][1] = np.linspace( cbDict['f_min'], cbDict['f_max'], cbDict['nchemfile'] )

    ###### calculate boundary conditions for pure fuel and oxidiser
    # 给纯燃料和纯氧化剂的边界条件，修改BCdata，0行是燃料边界，1是氧化侧的边界
    gas_fuel = ct.Solution(chemMech,'gri30_multi') # 再次创建一套反应系统
    gas_fuel.TPX = T_fuel,p,X_fuel # 赋值燃油温度，压力，以及摩尔分数
    BCdata[0,0] = gas_fuel.T # 燃油温度              
    BCdata[0,1] = gas_fuel.density_mass # 燃油密度    

    # 根据混合温度不同，修正定压比热容
    gas_fuelCP = ct.Solution(chemMech)
    gas_fuelCP.TPX = 298.15,p,X_fuel
    cp_0 = gas_fuelCP.cp_mass
    if abs(T_fuel - 298.15) < 0.1:
        BCdata[0,2] = cp_0
    else:
        sum_CpdT = 0.                                       
        #integral of cp in T space from 298.15 to T_fuel
        dT = (T_fuel-298.15)/(int(100*abs(T_fuel-298.15))-1)
        for kk in range(1,int(100*abs(T_fuel-298.15))):
            gas_fuelCP.TPX = (298.15 + kk*dT),p,X_fuel
            cp_1 = gas_fuelCP.cp_mass
            sum_CpdT = sum_CpdT + 0.5*(cp_0 + cp_1)*dT
            cp_0 = cp_1
        BCdata[0,2] = sum_CpdT / (T_fuel-298.15)    

    # 将摩尔分数转化成对应的质量，这是第四个边界参数
    BCdata[0,3] = np.dot(np.transpose(gas_fuel.X),gas_fuel.molecular_weights) 

    # 再次创建反应系统，确定298.15K温度下的焓值
    gas_fuelHf = ct.Solution(chemMech)
    gas_fuelHf.TPX = 298.15,p,X_fuel
    BCdata[0,4] = gas_fuelHf.enthalpy_mass         

    # 燃油的运动粘度
    BCdata[0,5] = gas_fuel.viscosity/gas_fuel.density_mass  
    
    # 燃油的焓值，这是自定义燃油温度侧的反应系统的焓值
    BCdata[0,6] = gas_fuel.enthalpy_mass                    

    # 输出并加以判断一下比热容修正后的温度
    print('T_fuel_approx: {0:7f}'.format((BCdata[0,6]-BCdata[0,4])
                                        /BCdata[0,2]+298.15))

    
    # 氧化剂边界再来一遍
    gas_ox = ct.Solution(chemMech,'gri30_multi')
    gas_ox.TPX = T_ox,p,X_ox
    BCdata[1,0] = gas_ox.T                 
    BCdata[1,1] = gas_ox.density_mass      
    #氧化剂密度

    gas_oxCP = ct.Solution(chemMech)
    gas_oxCP.TPX = 298.15,p,X_ox
    cp_0 = gas_oxCP.cp_mass
    if abs(T_ox - 298.15) < 0.1:
        BCdata[1,2] = cp_0
    else:
        sum_CpdT = 0.0
        dT = (T_ox-298.15)/(int(100*abs(T_ox-298.15))-1)
        for kk in range(1,int(100*abs(T_ox-298.15))):
            gas_oxCP.TPX = (298.15 + kk*dT),p,X_ox
            cp_1 = gas_oxCP.cp_mass
            sum_CpdT = sum_CpdT + 0.5*(cp_0 + cp_1)*dT
            cp_0 = cp_1
        BCdata[1,2] = sum_CpdT / (T_ox-298.15)     
    #氧化剂的平均定压比热容

    BCdata[1,3] = np.dot(np.transpose(gas_ox.X),gas_ox.molecular_weights)   
    #氧化剂的质量

    gas_oxHf = ct.Solution(chemMech)
    gas_oxHf.TPX = 298.15,p,X_ox
    BCdata[1,4] = gas_oxHf.enthalpy_mass     
    #氧化剂的焓

    BCdata[1,5] = gas_ox.viscosity/gas_ox.density_mass    
    #氧化剂的运动粘性系数
    BCdata[1,6] = gas_ox.enthalpy_mass                    
    #氧化剂的焓  （'gri30_multi'）

    # 氧化剂侧修正比热容后的温度
    print('T_ox_approx: {0:7f}'.format((BCdata[1,6]-BCdata[1,4])/BCdata[1,2]
                                        +298.15))

    
    # species BCs，下面是剩下的组分边界
    gas = ct.Solution(chemMech) # 创建一个反应系统
    for s in range(cbDict['nYis']): # 感兴趣的组分数量，不把所有的都输出
        ispc = gas.species_index(cbDict['spc_names'][s])    # 感兴趣的组分名称
        #第s个组分的索引
        iBC = (nscal_BC-1) + s*2 + 1 # nscal_BC在案例中是7，就是上面已经列出的相关参数，s是第几个感兴趣的组分，乘2是因为这个组分有两个需要保存的参数，加1是因为个数比最大编号多1  
        #本循环中的结果存储在BCdata的最后2*nYis列  
        BCdata[0,iBC] = ispc     # 燃油侧，先保存s组分的在所有组分中的编号
        #第s个组分的索引
        BCdata[1,iBC] = ispc     # 氧化侧，同样保存s组分的编号，没区别，而且二者保存的内容是一样的？
        #第s个组分的索引
        
        iBC = (nscal_BC-1) + (s+1)*2 # s组分要保存的第二个参数（第一个是编号），不过这里写的确实有点抽象了，不太规范
        if gas_fuel.Y[ispc]>1.e-30: # 组分存在
            BCdata[0,iBC] = gas_fuel.Y[ispc] # 赋值进去，初始化的是0，用的np.zero   
            #燃料中第s个组分的质量分数
        if gas_ox.Y[ispc]>1.e-30:
            BCdata[1,iBC] = gas_ox.Y[ispc]      # 氧化剂侧也同理，不过燃油和氧化剂这里边界存储的一个东西
        #氧化剂中第s个组分的质量分数


    # save the laminar parameters of all the flamelets
    # 开始保存层流火焰的相关参数，这里应该是设定保存的两个文件的格式
    # fmt_str1 保存的是那八个参数，在phi_tab里面，具体如下
    # 第一个，equivalence ratio，浮点数
    # 第二个，mixture fraction，浮点数
    # 第三个，number of grid points，网格数，整型数
    # 第四个，flame speed，浮点数
    # 第五个，flame thickness，浮点数
    # 第六个，heat release parameter，浮点数
    # 第七个，进度变量源项封闭系数Kc，浮点数
    # 第八个，定压比热和对应温度的函数积分，浮点数
    fmt_str1=''
    for ff in range(8):  
        if ff == 2:
            fmt_str1 = fmt_str1 + '%04d '
        else:
            fmt_str1 = fmt_str1 + '%.5e '
        #
    # fmt_str1 保存边界参数，除了组分编号，其余全是浮点数
    fmt_str2=''
    for ff in range(len(BCdata[0,:])):    
        nn = ff - nscal_BC
        if nn >= 0 and nn % 2 == 0:
            fmt_str2 = fmt_str2 + '%04d '
        else:
            fmt_str2 = fmt_str2 + '%.5e '
        #

    # 按照设定的格式，开始储存数据了
    with open(fln_strain_tab,'w', encoding="utf-8") as strfile: # w是write，fln_strain_tab是路径
        strfile.write(CASENAME + '\n') # 先把casename写进去
        np.savetxt(strfile,strain_tab,fmt=fmt_str1.strip()) # 保存phi_tab那八个东西
        np.savetxt(strfile,BCdata,fmt=fmt_str2.strip()) # 保存边界的东西
    strfile.close()                               
    #将phi_tab、BCdata保存到lamParameters.txt
    #

    # # 将canteraData/文件夹压缩打包
    # with zipfile.ZipFile('canteraData.zip','w') as target:
    #     for i in os.walk(solFln):
    #         for n in i[2]:
    #             target.write(''.join((i[0],n)))

    # zipPath='canteraData.zip'

    # return zipPath

    # tarPath = 'canteraData.tar'
    # with tarfile.open(tarPath, 'w') as tar:
    #     tar.add(solFln)

    # return tarPath

    # return solFln


# ## 2.3 曲线建表
# 在FPV中，火焰面的数据分布非常狭窄，所以我们采用了曲线建表法，以进度变量方向为基准，量化一维火焰面数据。

# In[ ]:


#normal_table
import numpy as np
import numpy.matlib
import scipy
import tarfile
import os

import math

import cantera as ct

from scipy.interpolate import griddata
from scipy.interpolate import LinearNDInterpolator

# 混合分数插值,
# 交换，进度变量插值
def ntable(cbDict, solFln):

    z_min, z_max = cbDict['f_min'], cbDict['f_max']
    couple_z = np.linspace( z_min, z_max, cbDict['nchemfile'] )


    total_data = np.zeros( (cbDict['nstrains'], cbDict['nchemfile'],cbDict['nSpeMech']+cbDict['nVarCant'])  )

    #C0 = np.zeros( (0,cbDict['nSpeMech']+cbDict['nVarCant']) )

    # 先读取所有的拉伸率文件
    
    for i in range(cbDict['nstrains']):    #对于每个拉伸率    
        wdata = []
        fln = (solFln + '/' + cbDict['CASENAME'] + '_' + 'diff' + '_' + str('{:03d}'.format(i)) + '.csv')
        print('\nReading --> ' + fln)  #读取对应的文件CH4_diff_i.csv到fln中
        
        with open(fln) as f:
            j = 0
            for line in f:
                line = line.strip()
                wdata.append(line.split(' ')) 
                j += 1
        wdata = np.array(wdata,dtype='float') # 这是之前保存的文件，下一步在合适的混合分数范围内，插值这个
                                                  # 插值后，选中固定的行，然后一行一行的转移到data中
        # 插值，对混合分数插值
        zdata = wdata[:,10]
        wdata = np.delete(wdata, [10,11], axis=1)

        newdata = np.zeros( (couple_z.shape[0], cbDict['nSpeMech']+cbDict['nVarCant']) )
        
        for col in range(0, newdata.shape[1]):
            order_zdata = np.flip(zdata)
            order_wdata = np.flip(wdata[:,col])
            newdata[:,col] = np.interp(couple_z, order_zdata, order_wdata )

        print(newdata[:,0],'\n')
        print(wdata[:,0],'\n')
        print(zdata,'\n')
         
        # 写入总数据里面
        total_data[i,:,:] = newdata


    # 交换总数据的第一和第二维度
    total_data = total_data.transpose(1, 0, 2) # 交换，维度变成了 nchemfile*nstrains*表格信息

    for i in range(cbDict['nchemfile']):
        fln = solFln + cbDict['CASENAME'] + '_' + '{:03d}'.format(i) + '.csv' # 新创建一个文件的名字

        savedata = total_data[i,:,:]

        # 排序
        c_column = savedata[:, 0]

        # 获取排序后的索引
        sorted_indices = np.argsort(c_column)

        # 根据排序后的索引调整矩阵的行顺序
        sorted_savedata = savedata[sorted_indices]


        # 插值
        # 保证长度是normalc_npts，如果有过滤，那么补足
        final_data = np.zeros( (cbDict['normalc_npts'], sorted_savedata.shape[1]) )

        final_data[:,0] = np.linspace(0,np.max(sorted_savedata[:,0]),cbDict['normalc_npts'])

        for col in range(1, final_data.shape[1]):
            old_col = sorted_savedata[:, col]
            new_col = np.interp(final_data[:,0], sorted_savedata[:, 0], old_col)
            final_data[:, col] = new_col

        # 进度变量大于0.95最大进度变量的，赋值为0
        for j in range(len(final_data[:,0])):
            if(final_data[j,0]>=0.95*np.max(final_data[:,0])): #or final_data[j,0]<=0.4*np.max(final_data[:,0])):
                final_data[j,1] = 0

        # 修正final_data的密度
        #for j in range(cbDict['nstrains']):
        #    gas = ct.Solution(cbDict['chemMech'])
            #reactants = final_data[:,cbDict['nVarCant']:]
        #    gas.TPX = final_data[j,2],cbDict['p'],final_data[j,cbDict['nVarCant']:]
        #    final_data[j,3] = gas.density



        np.savetxt(fln, final_data)
        
    print("Convert Table done.\n")


# ## 2.4 网格插值

# In[ ]:


#interpToMeshgrid
import numpy as np
import numpy.matlib
import scipy
import tarfile
import os
# import zipfile
import math

def interpLamFlame(cbDict,solFln):

    # tar = tarfile.open(solFln,'r')
    # tar.extractall()
    # tar.close()

    # solFln = "./canteraData"

    # os.chdir(solFln)

    lamArr = []                                     
    # 读取lamparameters的数据，但在当前的对冲火焰中，没有用
    with open(solFln + '/' + 'lamParameters.txt') as f:   
        ll = 0
        for line in f:
            if ll == 0:
                casename = line.strip()     
                #Python strip() 方法用于移除字符串头尾指定的字符
                # （默认为空格或换行符）或字符序列。
            elif ll > cbDict['nchemfile']:  
              #只读lamParameters.txt中前nchemfile行数据    
              #nchemfile：需要计算的当量比的数量
                break
            else:
                line = line.strip()
                lamArr.append(line.split(' '))   
                #Python split() 通过指定分隔符对字符串进行切片，
                #如果参数 num 有指定值，则分隔 num+1 个子字符串
            ll += 1
    lamArr = np.array(lamArr,dtype='float')    
    #将lamArr的数据创建为一个数组lamArr，数据类型为float

    streamBC = np.loadtxt(solFln + '/' + 'lamParameters.txt',
                          skiprows=1+cbDict['nchemfile'])    
                          #将边界条件数据BCdata读取到streamBC中
    # 所谓的边界参数，就是纯燃料和纯氧化剂的相关热物性，我感觉，这里的意思是给Z=0和Z=1，每个混合分数都添加两行这个东西

    with open(solFln + '/' + cbDict['output_fln'] ,'w') as strfile:             
        strfile.write('%.5E' % streamBC[0,6] + '\t' +
                    '%.5E' % streamBC[1,6] + '\n')    
                    #燃料/氧化剂的焓  （'gri30_multi'）
        strfile.write(str(cbDict['nchemfile']) + '\n')
        np.savetxt(strfile,
                    np.transpose([lamArr[:,1],lamArr[:,3],lamArr[:,4],
                                lamArr[:,5],lamArr[:,6]]),
                    fmt='%.5E',delimiter='\t')         
                    #当量比、网格点数、火焰速度、火焰厚度、热释放参数
    strfile.close()

    # read cantera solutions & calculate c,omega_c
    nScalCant = cbDict['nSpeMech'] + cbDict['nVarCant']    
    #总组分数+nVarCant  #nVarCant=10
    
    # mainData = np.zeros((cbDict['nchemfile'],int(max(lamArr[:,2])),nScalCant))  
    # maindata是读取所有的csv文件
    mainData = np.zeros((cbDict['nchemfile'],cbDict['normalc_npts'],nScalCant))
    
    
    #矩阵大小：nchemfile*max(lamArr[:,2])*nScalCant   
    #nchemfile：要计算的当量比数目  #lamArr[:,2]：网格数目
    cIn = np.zeros(np.shape(mainData[:,:,0]))    # 维度是nchemfile*normalc_npts
    #np.shape(mainData[:,:,0])：maindata当nScalCant=0时的维数，即：nchemfile*max(lamArr[:,2])
    omega_cIn = np.zeros(np.shape(mainData[:,:,0]))    
    #矩阵大小：nchemfile*max(lamArr[:,2])
    Yc_eq = np.zeros((cbDict['nchemfile'])) 
    for i in range(cbDict['nchemfile']):    #对于每个当量比phi：
        fln = (solFln + '/' + casename + '_' + str('{:03d}'.format(i)) + '.csv')
        print('\nReading --> ' + fln)  #读取对应的文件CH4_i.csv到fln中

        # len_grid = int(lamArr[i,2])   #网格点数
        len_grid = cbDict['normalc_npts']
        
        with open(fln) as f:
            j = 0
            for line in f:
                if j >= len_grid:
                    break
                line = line.strip()
                mainData[i,j,:] = line.split(' ') #将CH4_i.csv的第j行值赋给mainData[i,j,:]
                j += 1

            # for j in range(int(max(lamArr[:,2]))):
            #         if j < int(lamArr[i,2]):
            #             mainData[i,j,:] = np.loadtxt(fln,skiprows=j,max_rows=1)
            #             end = j
            # else: mainData[i,j,:] = np.loadtxt(fln,skiprows=end,max_rows=1)

        imax = np.argmax(mainData[i,:,0])      
        #mainData[i,:,0]最大值的索引   #mainData[i,:,0]:CO和CO2的质量分数之和表示的反应进度
        cIn[i,:] = mainData[i,:,0] / mainData[i,imax,0]     
        #反应进度的向量/最大反应进度
        if mainData[i,imax,0]/mainData[i,len_grid-1,0] > 1.0:    
          #反应过度
            print('c_max/c_end =',mainData[i,imax,0]/mainData[i,len_grid-1,0],
                  ' --> overshooting')

        if(cbDict['scaled_PV']):             
          #是否使用相对的反应进度   
          Yc_eq[i] = mainData[i,imax,0]
          omega_cIn[i,:] = mainData[i,:,1] / Yc_eq[i]    
          #反应速率向量/最大反应进度
        else:
          omega_cIn[i,:] = mainData[i,:,1]


    z_coord = np.linspace(cbDict['f_min'], cbDict['f_max'], cbDict['nchemfile'])
    d2Yeq_table = []
    if(cbDict['scaled_PV']):
        d2Yeq_table = generateTable2(z_coord,Yc_eq,cbDict["z"],cbDict["gz"],cbDict["f_min"],cbDict["f_max"])     
        #如果使用相对值，调用generateTable2函数，计算截断后的进程变量Z1的概率密度分布    
        #输入：lamArr[:,1] 混合物分数； Yc_eq 反应速率； contVarDict 初始的z、c、gz、gc、gcz(都是向量)
        fln = (solFln + '/' + 'd2Yeq_table.dat')  
        #work_dir/canteraData/solution_00文件夹下创建d2Yeq_table.dat文件
        print('\nWriting --> ' + fln)
        np.savetxt(fln,d2Yeq_table,fmt='%.5E')   

    # interpolate in c space & write out for each flamelet
    MatScl_c = np.zeros((cbDict['nchemfile'],cbDict['cUnifPts'],      
                         nScalCant+1))
            #chemfile*cUnifPts*(nScalCant+1)    #chemfile:当量比的数量  #cUnifPts：用进程变量c插值的节点数  
            #nScalCant = nSpeMech+nVarCant
    for i in range(cbDict['nchemfile']):   
      #对于每个当量比i：
        
        # len_grid = int(lamArr[i,2])        
        len_grid = cbDict['normalc_npts']
                        
        #网格点数
        ctrim = cIn[i,:len_grid-1]         
        #反应进度cIn的第i行0:(len_grid-2)列
        # 0:c|1:omg_c
        MatScl_c[i,:,0] = np.linspace(0.,1.,cbDict['cUnifPts'])      
        #进程变量c的插值点     #当量比i下的MatScl_c的第0列：0,...,1   cUnifPts个点
        MatScl_c[i,:,1] = np.matlib.interp(MatScl_c[i,:,0],ctrim,
                                         omega_cIn[i,:len_grid-1])   
                                #当量比i下的MatScl_c的第1列：MatScl_c[i,:,0]插值得到的反应速率    
                                #MatScl_c[i,:,0]在x=ctrim, y=omega_cIn[i,:len_grid-1]上插值
        # 2:T|3:rho|4:cp|5:mw|6:hf_0|7:nu|8:h|9:qdot
        for k in range(2,nScalCant):
            MatScl_c[i,:,k] = np.matlib.interp(MatScl_c[i,:,0],ctrim,
                                             mainData[i,:len_grid-1,k])   
                #当量比i下的MatScl_c的第2到nScalCant列：插值得到的温度/密度/定压比热容/质量/生成焓/运动粘度/焓/释热率
        # cp-->cp_e，不在这里做了
        # MatScl_c[i,:,4] = calculateCp_eff(MatScl_c[i,:,2],MatScl_c[i,:,4],
        #                                  lamArr[i,7])    
            #把定压比热容改成有效定压比热容  #输入：MatScl_c[i,:,2]温度  MatScl_c[i,:,4]定压比热容  
            #lamArr[i,7]定压比热容在T=298.15~Tin的积分
        # Yc_max
        MatScl_c[i,:,-1] = mainData[i,len_grid-1,0]    
        #当量比i下的MatScl_c的最后一列：出口处的反应进度

        # write inpterpolated 1D profiles
        fln = (solFln + '/' + 'Unf_'+casename+'_'+str('{:03d}'.format(i))+'.dat')  
        #work_dir/canteraData/solution_00文件夹下创建Unf_CH4_i.dat文件
        print('\nWriting --> ' + fln)
        np.savetxt(fln,MatScl_c[i,:,:],fmt='%.5e')         
        #将MatScl_c第i个当量比的数据写入Unf_CH4_i.dat

    oriSclMat = np.zeros([cbDict['nchemfile']+2,cbDict['cUnifPts'],
                          nScalCant+1]) 

    ind_list_Yis = []
    for i in range(len(streamBC[:,0])):   
      #len(streamBC[:,0])=2

        if i == 0: j = len(oriSclMat[:,0,0]) - 1   
        #若i==0,则j=nchemfile+1
        else: j = 0                                
        #else, j=0
        for k in range(len(streamBC[0,:])):        
          #len(streamBC[0,:])=nscal_BC+2*nYis
            # for thermo scalars
            if k < len(streamBC[0,:]) - 2*cbDict['nYis']:  
              #若k<nscal_BC:
                oriSclMat[j,:,k+2] = streamBC[i,k]         
                #i=0,j=nchemfile+1=51时,oriSclMat[j,:,k+2]存放:
                #燃料的温度、密度、平均定压比热容、质量、焓、运动粘度、焓  （'gri30_multi'）
                #i=1,j=0时，oriSclMat[j,:,k+2]存放氧化剂的相应值
            else:
                nk = k                       
                #当循环到k=nscal_BC时, 令nk=nscal_BC,并终止k的循环
                break

        # for selected species
        for s in range(cbDict['nYis']):     
          #'H2O','CO','CO2'   nYis=3
            ispc=int(streamBC[i,nk+s*2])     
            #ispc=int(BCdata[i,nscal_BC+s*2])     #燃料/氧化剂中第s个组分的索引
            if i == 0: ind_list_Yis.append(ispc)    
            #若i=0, 则把ispc添加入ind_list_Yis
            iscal = cbDict['nVarCant']+ispc         
            #nVarCant=10
            oriSclMat[j,:,iscal]=streamBC[i,nk+s*2+1]   
            #i=0,j=nchemfile+1=51时，oriSclMat[j,:,iscal]存储第s个燃料的质量分数；i=1,j=0时，存放第s个氧化剂的质量分数

    oriSclMat[1:cbDict['nchemfile']+1,:,:] = MatScl_c    
    #oriSclMat的中间部分存放MatScl_c的结果

    intpSclMat = np.zeros([len(cbDict['z_space']),len(cbDict['c_space']),
                           cbDict['nScalars']])    
                           #大小：z_space*c_space*nScalars
    intpYiMat = np.zeros([len(cbDict['z_space']),len(cbDict['c_space']),
                        cbDict['nYis']])           
                        #大小：z_space*c_space*nYis

    lamZ = np.linspace( cbDict['f_min'], cbDict['f_max'], cbDict['nchemfile'] )

    Z_pts = lamZ
    Z_pts = np.insert(lamZ,0,[0.],axis=0)          
                        
    #Z_pts = np.insert(lamArr[:,1],0,[0.],axis=0)
                        
    
    Z_pts = np.insert(Z_pts,len(Z_pts),[1.],axis=0)  

    c_pts = MatScl_c[0,:,0]    


    np.array(ind_list_Yis)    


    print('\nInterpolating...')
    intpSclMat[:,:,0] = interp2D(oriSclMat[:,:,3],Z_pts,c_pts,cbDict['z_space'],cbDict['c_space']) # rho        

    for k in [1,4,5,6]: # omega_c,cp_e,mw,hf                                  
        intpSclMat[:,:,k] = interp2D(oriSclMat[:,:,k],Z_pts,c_pts,cbDict['z_space'],cbDict['c_space'])          

    intpSclMat[:,:,7] = interp2D(oriSclMat[:,:,2],Z_pts,c_pts,cbDict['z_space'],cbDict['c_space']) # T          

    intpSclMat[:,:,8] = interp2D(oriSclMat[:,:,7],Z_pts,c_pts,cbDict['z_space'],cbDict['c_space']) # nu         

    intpSclMat[:,:,9] = interp2D(oriSclMat[:,:,8],Z_pts,c_pts,cbDict['z_space'],cbDict['c_space']) # h          

    intpSclMat[:,:,10] = interp2D(oriSclMat[:,:,9],Z_pts,c_pts,cbDict['z_space'],cbDict['c_space']) # qdot


    # Yc_max
    if cbDict['scaled_PV']:           
      #如果使用相对反应进度
      intpSclMat[:,:,11] = 1.0        

    else:
      intpSclMat[:,:,11] = interp2D(oriSclMat[:,:,-1],Z_pts,c_pts,
                                    cbDict['z_space'],cbDict['c_space'])   


    for y in range(cbDict['nYis']):
        iy = ind_list_Yis[y]    

        intpYiMat[:,:,y] = interp2D(oriSclMat[:,:,iy+cbDict['nVarCant']],
                                            Z_pts,c_pts,cbDict['z_space'],cbDict['c_space'])    

    print('\nInterpolation done. ')

    if np.sum(np.isnan(intpSclMat)) > 0:   

        print('\nNumber of Nans detected: ', np.sum(np.isnan(intpSclMat)))    

    else: print('\nNo Nans detected. Well done!')

    print('\nwriting chemTab file...')
    Arr_c,Arr_Z = np.meshgrid(cbDict['c_space'],cbDict['z_space'])   

    Idx_outLmt = np.hstack([(Arr_Z>cbDict['f_max']).nonzero(),
                             (Arr_Z<cbDict['f_min']).nonzero()])   

    ind_rates=[1,9]                                                
    intpSclMat[:,:,ind_rates][Idx_outLmt[0],Idx_outLmt[1]] = 0.    

    chemMat = np.append(intpSclMat,intpYiMat,axis=2)     

    chemMat = np.insert(chemMat,0,Arr_c,axis=2)          

    chemMat = np.insert(chemMat,0,Arr_Z,axis=2)          

    stackMat = np.reshape(chemMat,[np.shape(chemMat)[0]*np.shape(chemMat)[1],
                              np.shape(chemMat)[2]])     
 

    fln = solFln + '/' + 'chemTab_' + str('{:02d}'.format(cbDict['solIdx'])) + '.dat'   

    np.savetxt(fln,stackMat,fmt='%.5E')    



    print('\ninterpToMeshgrid Done.')

    # with zipfile.ZipFile('interpToMeshgrid.zip','w') as target:
    #     for i in os.walk('./'):
    #         for n in i[2]:
    #             target.write(''.join((i[0],n)))

    # zipPath='canteraData.zip'

    # return zipPath

    #经过比较，用tar打包速度更快

    # tarPath = 'interpData.tar'
    # with tarfile.open(tarPath, 'w') as tar:
    #     tar.add('./' + cbDict['output_fln'])
    #     tar.add('./d2Yeq_table.dat')
    #     tar.add('./chemTab_' + str('{:02d}'.format(cbDict['solIdx'])) + '.dat')
    #     for i in range(cbDict['nchemfile']):   
    #         tar.add('./Unf_'+casename+'_'+str('{:03d}'.format(i))+'.dat')

        # tar.add('.')

    # return tarPath

    # return solFln

''' ===========================================================================

Subroutine functions

=========================================================================== '''
def interp2D(M_Zc,Z_pts,c_pts,z_space,c_space):   
    #对于(Z_pts,c_pts)上取值的M_Zc，在插值点meshgrid上插值
    f = scipy.interpolate.interp2d(c_pts,Z_pts,M_Zc,kind="linear")
    intpM_Zc = f(c_space,z_space)
    # import matplotlib.pyplot as plt
    # plt.plot(c_pts, M_Zc[0, :], 'ro-', c_space, intpM_Zc[0, :], 'b-')
    # plt.show()
    return intpM_Zc

def generateTable2(lamArr,Yc_eq,z,gz,f_min,f_max):     
    #输入：lamArr[:,1] 混合物分数； Yc_eq 反应速率； contVarDict 初始的z、c、gz、gc、gcz
    Z0 = lamArr          
    #混合物分数
    Yc_eq0 = Yc_eq       
    #反应速率

    from scipy.interpolate import UnivariateSpline
    sp = UnivariateSpline(Z0,Yc_eq0,s=0)    
    #样条曲线y=sp(x)拟合到（x,y）=(Z0,Yc_eq0)上   返回值相当于一个函数
    Z_low_cutoff = f_min            
    #截断时，混合物分数Z0的最小阈值, Z_low_cutoff>=0
    Z_high_cutoff = f_max          
    #截断时，混合物分数Z0的最大阈值，Z_high_cutoff<=1
    Z1 = np.linspace(Z0[0],Z_high_cutoff,101)     
    #Z1=Z0[0],...,Z_high_cutoff   #101个节点
    Yc_eq1 = sp(Z1)                               
    #用拟合出的样条函数sp得到插值的反应速率

    import matplotlib.pyplot as plt
    plt.plot(Z0,Yc_eq0,label='original')
    plt.plot(Z1,Yc_eq1,label='spline')
    plt.legend()
    plt.show()

    d2 = sp.derivative(n=2)   
    #sp对x求二阶导
    d2Yc_eq1 = d2(Z1)         
    #混合物分数的二阶导
    sp = UnivariateSpline(Z1,d2Yc_eq1)     
    #混合物分数的二阶导的拟合曲线
    d2Yc_eq2 = sp(Z1)         
    #混合物分数的二阶导的拟合结果

    from scipy.signal import savgol_filter     
    #Savitzky-Golay滤波器，用于数据流平滑除噪，在时域内基于局部多项式最小二乘法拟合的滤波方法。
    # 特点：在滤除噪声的同时保持信号的形状、宽度不变
    #scipy.signal.savgol_filter(x,window_length,polyorder)；x为要滤波的信号；
    #window_length为窗口长度，取值为奇数且不能超过len(x)，越大则平滑效果越明显；
    #polyorder为多项式拟合的阶数，越小则平滑效果越明显
    d2Yc_eq3 = savgol_filter(d2Yc_eq2, 11, 3)   
    #对d2Yc_eq2滤波，window_length=11，polyorder=3
    plt.plot(Z1,d2Yc_eq1,label='original')      
    plt.plot(Z1,d2Yc_eq3,label='spline')
    plt.legend()
    plt.show()

    z_int = z           
    #导入z的初始值
    gz_int = gz         
    #导入gz的初始值
    gradd2 = np.gradient(d2Yc_eq3,Z1[1]-Z1[0])   
    #求d2Yc_eq3的梯度，d2Yc_eq3相邻元素之间的间距为Z1[1]-Z1[0]   
    #numpy.gradient(f,*varages)  f:一个包含标量函数样本的N-dimensional数组；varages：可选参数，f值之间的间距

    from scipy.stats import beta
    d2Yeq_int = np.zeros((len(z_int),len(gz_int)))  
    #矩阵大小len(z_int)*len(gz_int)=int_pts_z*int_pts_gz
    for i in range(1,len(z_int)-1):   
        #d2Yeq_int[0,:]本来就是0
        if z_int[i] > Z_high_cutoff or z_int[i] < Z_low_cutoff:
            d2Yeq_int[i,:] = 0.0           
            #在z的截断区间（Z_low_cutoff，Z_high_cutoff）之外的，设为0
        else:
            d2Yeq_int[i,0] =np.interp(z_int[i],Z1,d2Yc_eq3)   
            #点x=z_int[i]插值曲线d2Yc_eq3=f(Z1)上插值    #d2Yc_eq3：混合物分数的二阶导的拟合后再滤波的结果
            if (gz_int[-1]==1): gz_len = len(gz_int)-1
            else: gz_len = len(gz_int)
            for j in range(1,gz_len):      
            #表格d2Yeq_int的y方向
                a = z_int[i]*(1.0/gz_int[j]-1.0)  
                    #z*(1/gz-1)
                b = (1.0 - z_int[i])*(1.0/gz_int[j]-1.0)  
                #(1-z)*(1/gz-1)
                Cb = beta.cdf(Z1,a,b)    
                #累积分布函数（F_X(x)=P(X<=x),表示：对离散变量而言，所有小于等于X的值出现概率之和）  
                #a和b是形状参数，beta.cdf中计算了gamma(a),gamma(b),gamma(b)  #0<=Z1<=1
                d2Yeq_int[i,j] = d2Yc_eq3[-1] - np.trapz(gradd2*Cb,Z1)   
                #物理意义是：进程变量Z1的概率密度分布    #d2Yc_eq3[-1]：出口处混合物分数的二阶导  
                #gradd2：混合物分数的二阶导的梯度  #Cb：Z1的累积分布函数  #Z1：要拟合的混合物分数节点向量  
                #索引[-1]指向的是向量的倒数第一个值

    d2Yeq_int_1D = np.zeros((d2Yeq_int.flatten()).shape)   
    #d2Yeq_int.flatten()：把d2Yeq_int降到一维，默认按照行的方向降(第一行-第二行-...)  #.shape: 读取矩阵在各维度的长度
    count = 0
    for i in range(len(z_int)):
        for j in range(len(gz_int)):
            d2Yeq_int_1D[count] = d2Yeq_int[i][j]   
            #把进程变量Z1的概率密度分布矩阵d2Yeq_int排成向量赋值给d2Yeq_int_1D，返回d2Yeq_int_1D的值
            count = count + 1

    return d2Yeq_int_1D


# ### 2.5 PDF积分
# 基于$\beta$-PDF方法，将一维火焰数据变成带有湍流信息的三维湍流火焰。

# In[ ]:


# PDF积分，将一维火焰转变成可用于三维的火焰
import numpy as np
import time
import multiprocessing as mp
from scipy.stats import beta
import scipy.io as sio
from math import exp
import os
def c_dYdccomputing(sc_vals_int, sc_vals_int1, sc_vals_int2, sc_vals_intScalars, space, space_diff, space_average, mean,
                    cdf001, n_points_c, index):
    dsdc = np.zeros(n_points_c + 2)
    Yc_0 = np.zeros(n_points_c + 2)
    Yc_1 = np.zeros(n_points_c + 2)
    if (index < 2 or index > 4): dsdc[1:n_points_c] = (sc_vals_int[2:n_points_c + 1] - sc_vals_int[
                                                                                       1:n_points_c]) / space_diff[
                                                                                                        1:n_points_c]
    if (index == 2): dsdc[1:n_points_c] = (sc_vals_int2[2:n_points_c + 1] / sc_vals_int1[
                                                                            2:n_points_c + 1] - sc_vals_int2[
                                                                                                1:n_points_c] / sc_vals_int1[
                                                                                                                1:n_points_c]) / space_diff[
                                                                                                                                 1:n_points_c]
    if (index == 3):
        Yc_0[1:n_points_c] = space[1:n_points_c] * sc_vals_intScalars[1:n_points_c]
        Yc_1[2:n_points_c + 1] = space[2:n_points_c + 1] * sc_vals_intScalars[2:n_points_c + 1]
        dsdc[1:n_points_c] = (np.multiply(Yc_1[2:n_points_c + 1], sc_vals_int2[2:n_points_c + 1] / sc_vals_int1[
                                                                                                   2:n_points_c + 1]) - np.multiply(
            Yc_0[1:n_points_c], sc_vals_int2[1:n_points_c] / sc_vals_int1[1:n_points_c])) / space_diff[1:n_points_c]
    if (index == 4):
        dsdc[1:n_points_c] = (mean * sc_vals_int2[2:n_points_c + 1] / sc_vals_int1[
                                                                      2:n_points_c + 1] - mean * sc_vals_int2[
                                                                                                 1:n_points_c] / sc_vals_int1[
                                                                                                                 1:n_points_c]) / space_diff[
                                                                                                                                  1:n_points_c]
    dsdc[n_points_c] = dsdc[n_points_c - 1]
    dsdc[n_points_c + 1] = dsdc[1]
    y_int = 0
    y_int = y_int - np.sum(0.5 * np.multiply((np.multiply(dsdc[1:n_points_c - 1],
                                                          cdf001[1:n_points_c - 1]) + np.multiply(dsdc[2:n_points_c],
                                                                                                  cdf001[
                                                                                                  2:n_points_c])),
                                             (space_average[2:n_points_c] - space_average[1:n_points_c - 1])))
    y_int = y_int - dsdc[n_points_c] * cdf001[n_points_c] * (space[n_points_c] - space[n_points_c - 1]) / 2.0 - dsdc[
        n_points_c + 1] * cdf001[1] * (space[2] - space[1]) / 2.0 + sc_vals_int[n_points_c]
    return y_int
def z_dYdccomputing(sc_vals_int, sc_vals_int1, sc_vals_int2, sc_vals_intScalars, space, space_diff, space_average, mean,
                    cdf001, n_points_z, index):
    dsdc = np.zeros(n_points_z + 2)
    Yc_0 = np.zeros(n_points_z + 2)
    Yc_1 = np.zeros(n_points_z + 2)
    if (index < 2 or index > 4): dsdc[1:n_points_z] = (sc_vals_int[2:n_points_z + 1] - sc_vals_int[
                                                                                       1:n_points_z]) / space_diff[
                                                                                                        1:n_points_z]
    if (index == 2): dsdc[1:n_points_z] = (sc_vals_int2[2:n_points_z + 1] / sc_vals_int1[
                                                                            2:n_points_z + 1] - sc_vals_int2[
                                                                                                1:n_points_z] / sc_vals_int1[
                                                                                                                1:n_points_z]) / space_diff[
                                                                                                                                 1:n_points_z]
    if (index == 3):
        Yc_0[1:n_points_z] = mean * sc_vals_intScalars[1:n_points_z]
        Yc_1[2:n_points_z + 1] = mean * sc_vals_intScalars[2:n_points_z + 1]
        dsdc[1:n_points_z] = (np.multiply(Yc_1[2:n_points_z + 1], sc_vals_int2[2:n_points_z + 1] / sc_vals_int1[
                                                                                                   2:n_points_z + 1]) - np.multiply(
            Yc_0[1:n_points_z], sc_vals_int2[1:n_points_z] / sc_vals_int1[1:n_points_z])) / space_diff[1:n_points_z]
    if (index == 4):
        dsdc[1:n_points_z] = (space[2:n_points_z + 1] * sc_vals_int2[2:n_points_z + 1] / sc_vals_int1[
                                                                                         2:n_points_z + 1] - space[
                                                                                                             1:n_points_z] * sc_vals_int2[
                                                                                                                             1:n_points_z] / sc_vals_int1[
                                                                                                                                             1:n_points_z]) / space_diff[
                                                                                                                                                              1:n_points_z]
    dsdc[n_points_z] = dsdc[n_points_z - 1]
    dsdc[n_points_z + 1] = dsdc[1]
    y_int = 0
    y_int = y_int - np.sum(0.5 * np.multiply((np.multiply(dsdc[1:n_points_z - 1],
                                                          cdf001[1:n_points_z - 1]) + np.multiply(dsdc[2:n_points_z],
                                                                                                  cdf001[
                                                                                                  2:n_points_z])),
                                             (space_average[2:n_points_z] - space_average[1:n_points_z - 1])))
    y_int = y_int - dsdc[n_points_z] * cdf001[n_points_z] * (space[n_points_z] - space[n_points_z - 1]) / 2.0 - dsdc[
        n_points_z + 1] * cdf001[1] * (space[2] - space[1]) / 2.0 + sc_vals_int[n_points_z]
    return y_int
def c_cdfFunc(space_average, alpha_c, beta_c, n_points_c):
    cdf001 = beta.cdf(space_average, alpha_c, beta_c)
    cdf001[n_points_c] = 1
    return cdf001
def z_cdfFunc(space_average, alpha_z, beta_z, n_points_z):
    cdf001 = beta.cdf(space_average, alpha_z, beta_z)
    cdf001[n_points_z] = 1
    return cdf001
def processdata(data,k,x,y):
    data_new=[]
    for i in range(0,k):
        data_new.append(data[:,:,i])
    data_new=np.array(data_new)
    return data_new
def intfac(x,xarray,loc_low):
    if(x<xarray[loc_low]): return 0
    if (x > xarray[loc_low+1]): return 1
    return (x-xarray[loc_low])/(xarray[loc_low+1]-xarray[loc_low])
def delta(z_mean,c_mean,z_space,c_space,sc_vals,Yi_vals,n_points_z,n_points_c,nScalars,nYis):
    y_int=np.zeros(nScalars+1)
    Yi_int=np.zeros(nYis+1)
    z_loc = locate(z_space, n_points_z, z_mean)
    z_fac = intfac(z_mean, z_space, z_loc)
    c_loc = locate(c_space, n_points_c, c_mean)
    c_fac = intfac(c_mean, c_space, c_loc)
    for i in range(1,3):
        y_int[i]=(1-c_fac) * (z_fac * sc_vals[i][z_loc + 1][c_loc] +(1-z_fac) * (sc_vals[i][z_loc][c_loc]))+ c_fac * (z_fac * sc_vals[i][z_loc + 1][c_loc+1] + (1-z_fac) * sc_vals[i][z_loc][c_loc+1])
    for i in range(5,nScalars+1):
        y_int[i] = (1 - c_fac) * (z_fac * sc_vals[i][z_loc + 1][c_loc] + (1 - z_fac) * (sc_vals[i][z_loc][c_loc])) + c_fac * (z_fac * sc_vals[i][z_loc + 1][c_loc+1] + (1 - z_fac) * sc_vals[i][z_loc][c_loc+1])
    y_int[2] = y_int[2]/ y_int[1] #Zhi: omega_c / rho
    y_int[3] = y_int[2] * c_mean * y_int[nScalars] # gc_source
    y_int[4] = y_int[2] * z_mean # gz_source
    for i in range(1,nYis+1):
        Yi_int[i] = (1 - c_fac) * (z_fac * Yi_vals[i][z_loc + 1][c_loc] + (1 - z_fac) * (Yi_vals[i][z_loc][c_loc])) + c_fac * (z_fac * Yi_vals[i][z_loc + 1][c_loc + 1] + (1 - z_fac) * Yi_vals[i][z_loc][c_loc + 1])
    y_int = np.where(abs(y_int) > 1e-20, y_int, 0)
    Yi_int = np.where(Yi_int > 1e-20, Yi_int, 0)
    return y_int,Yi_int
# cbeta function
def cbeta(mean, g_var, space, space_average, space_diff, z_mean, z_space, sc_vals, Yi_vals, n_points_z, n_points_c,
          nScalars, nYis):
    loc = locate(z_space, n_points_z, z_mean)
    fac = intfac(z_mean, z_space, loc)
    sc_vals_int = np.zeros((nScalars + 1, n_points_c + 1))
    Yi_vals_int = np.zeros((nYis + 1, n_points_c + 1))
    y_int = np.zeros(nScalars + 1)
    Yi_int = np.zeros(nYis + 1)
    alpha_c = mean * ((1 / g_var) - 1)
    beta_c = (1 - mean) * ((1 / g_var) - 1)
    sc_vals_int[1:nScalars + 1, 1:n_points_c + 1] = fac * sc_vals[1:nScalars + 1, loc + 1, 1:n_points_c + 1] + (
                1 - fac) * sc_vals[1:nScalars + 1, loc, 1:n_points_c + 1]
    Yi_vals_int[1:nYis + 1, 1:n_points_c + 1] = fac * Yi_vals[1:nYis + 1, loc + 1, 1:n_points_c + 1] + (
                1 - fac) * Yi_vals[1:nYis + 1, loc, 1:n_points_c + 1]
    cdf001 = c_cdfFunc(space_average, alpha_c, beta_c, n_points_c)
    sc_vals_int1 = sc_vals_int[1]
    sc_vals_int2 = sc_vals_int[2]
    sc_vals_intScalars = sc_vals_int[nScalars]
    for j in range(1, nScalars + 1):
        y_int[j] = c_dYdccomputing(sc_vals_int[j], sc_vals_int1, sc_vals_int2, sc_vals_intScalars, space, space_diff,
                                   space_average, z_mean, cdf001, n_points_c, j)
    for j in range(1, nYis + 1):
        Yi_int[j] = c_dYdccomputing(Yi_vals_int[j], sc_vals_int1, sc_vals_int2, sc_vals_intScalars, space, space_diff,
                                    space_average, z_mean, cdf001, n_points_c, 1)
    y_int = np.where(abs(y_int) > 1e-20, y_int, 0)
    Yi_int = np.where(Yi_int > 1e-20, Yi_int, 0)
    return y_int, Yi_int
def zbeta(mean, g_var, space, space_average, space_diff, c_mean, c_space, sc_vals, Yi_vals, n_points_z, n_points_c,
          nScalars, nYis):
    loc = locate(c_space, n_points_c, c_mean)
    fac = intfac(c_mean, c_space, loc)
    sc_vals_int = np.zeros((nScalars + 1, n_points_z + 1))
    Yi_vals_int = np.zeros((nYis + 1, n_points_z + 1))
    y_int = np.zeros(nScalars + 1)
    Yi_int = np.zeros(nYis + 1)
    alpha_z = mean * ((1 / g_var) - 1)
    beta_z = (1 - mean) * ((1 / g_var) - 1)
    sc_vals_int[1:nScalars + 1, 1:n_points_z + 1] = fac * sc_vals[1:nScalars + 1, 1:n_points_z + 1, loc + 1] + (
                1 - fac) * sc_vals[1:nScalars + 1, 1:n_points_z + 1, loc]
    Yi_vals_int[1:nYis + 1, 1:n_points_z + 1] = fac * Yi_vals[1:nYis + 1, 1:n_points_z + 1, loc + 1] + (
                1 - fac) * Yi_vals[1:nYis + 1, 1:n_points_z + 1, loc]
    cdf001 = z_cdfFunc(space_average, alpha_z, beta_z, n_points_z)
    sc_vals_int1 = sc_vals_int[1]
    sc_vals_int2 = sc_vals_int[2]
    sc_vals_intScalars = sc_vals_int[nScalars]
    for j in range(1, nScalars + 1):
        y_int[j] = z_dYdccomputing(sc_vals_int[j], sc_vals_int1, sc_vals_int2, sc_vals_intScalars, space, space_diff,
                                   space_average, c_mean, cdf001, n_points_z, j)
    for j in range(1, nYis + 1):
        Yi_int[j] = z_dYdccomputing(Yi_vals_int[j], sc_vals_int1, sc_vals_int2, sc_vals_intScalars, space, space_diff,
                                    space_average, c_mean, cdf001, n_points_z, 1)
    y_int = np.where(abs(y_int) > 1e-20, y_int, 0)
    Yi_int = np.where(Yi_int > 1e-20, Yi_int, 0)
    return y_int, Yi_int
def computedata(data,type,n):
    res = np.zeros_like(data)
    if(type=="average"):
        res[1:n]=(data[1:n]+data[2:n+1])/2
    if(type=="diff"):
        res[1:n]=data[2:n+1]-data[1:n]
    return res
def locate(xarray,n,x):
    if(x<xarray[1]):
        return 1
    if(x>=xarray[n]):
        return n-1
    for k in range(1,n):
        if(x>=xarray[k] and x<xarray[k+1]): return k
def readcopula(filename):
    roadef_info = sio.loadmat(filename)
    prob = roadef_info['y'][0]
    return prob
def readspace(cbDict,ih):
    z_space = np.zeros(cbDict['n_points_z'] + 1)
    c_space = np.zeros(cbDict['n_points_c'] + 1)
    Src_vals = np.zeros((cbDict['n_points_z'] + 1, cbDict['n_points_c'] + 1, cbDict['nScalars'] + 1))
    Yi_vals = np.zeros((cbDict['n_points_z'] + 1, cbDict['n_points_c'] + 1, cbDict['nYis'] + 1))
    str_ih='%02d'%ih
    start1 = time.time()
    print('Reading chemTab...')
    f = open('./canteraData/chemTab_'+str_ih+'.dat')
    for i in range(1,cbDict['n_points_z']+1):
        for j in range(1,cbDict['n_points_c']+1):
            data=f.readline()
            data=data.split()
            z_space[i]=eval(data[0])
            c_space[j]=eval(data[1])
            Src_vals[i][j][1:]=[eval(x) for x in data[2:cbDict['nScalars']+2]]
            Yi_vals[i][j][1:] = [eval(x) for x in data[cbDict['nScalars'] + 2:]]
    print('Reading done')
    print("Reading chemTab耗时", time.time() - start1, "s")
    start=time.time()
    Src_vals=processdata(Src_vals,cbDict['nScalars']+1,cbDict['n_points_z']+1,cbDict['n_points_c']+1)
    Yi_vals = processdata(Yi_vals, cbDict['nYis'] + 1, cbDict['n_points_z'] + 1, cbDict['n_points_c'] + 1)
    return z_space,c_space,Src_vals,Yi_vals
def Psicomputing(sc_vals,sc_vals1,sc_vals2,sc_valsScalars,c_space,z_space,n_points_z,n_points_c,nScalars,nYis,k):
    z_space_use = np.reshape(np.repeat(z_space, n_points_c + 1), (n_points_z + 1, n_points_c + 1))
    c_space_use = np.reshape(np.tile(c_space, n_points_z + 1), (n_points_z + 1, n_points_c + 1))
    if(k<2 or k>4): return sc_vals[1:n_points_z+1,1:n_points_c + 1]
    if(k==2): return sc_vals[1:n_points_z+1,1:n_points_c + 1]/ sc_vals1[1:n_points_z+1,1:n_points_c + 1]
    if(k==3): return np.multiply(np.multiply(np.multiply(sc_vals2[1:n_points_z+1,1:n_points_c + 1], 1/ sc_vals1[1:n_points_z+1,1:n_points_c + 1]), c_space_use[1:n_points_z+1,1:n_points_c + 1]), sc_valsScalars[1:n_points_z+1,1:n_points_c + 1])
    if(k==4): return np.multiply(np.multiply(sc_vals2[1:n_points_z+1,1:n_points_c + 1], 1 / sc_vals1[1:n_points_z+1,1:n_points_c + 1]), z_space_use[1:n_points_z+1,1:n_points_c + 1])
def CDF_ind(z_space, c_space, z_space_average, c_space_average, alpha_z, beta_z, alpha_c, beta_c, n_points_c,
            n_points_z):
    CDF_C = np.zeros(n_points_c + 2)
    CDF_Z = np.zeros(n_points_z + 2)
    CDF_Z[1:n_points_z + 1] = beta.cdf(z_space_average[1:n_points_z + 1], alpha_z, beta_z)
    CDF_C[1:n_points_c + 1] = beta.cdf(c_space_average[1:n_points_c + 1], alpha_c, beta_c)
    j = n_points_c
    CDF_C[j] = beta.cdf((c_space[j - 1] + 3 * c_space[j]) / 4.0, alpha_c, beta_c)
    j = n_points_c + 1
    CDF_C[j] = beta.cdf((3 * c_space[1] + 2 * c_space[2]) / 4.0, alpha_c, beta_c)
    i = n_points_z
    CDF_Z[i] = beta.cdf((z_space[i - 1] + 3 * z_space[i]) / 4.0, alpha_z, beta_z)
    i = n_points_z + 1
    CDF_Z[i] = beta.cdf((3 * z_space[1] + z_space[2]) / 4.0, alpha_z, beta_z)
    return CDF_C, CDF_Z
def CDF_copula(z_space, c_space,alpha_z, beta_z, alpha_c, beta_c, n_points_c,n_points_z,type,rho,parameters):
    CDF_C = np.zeros(n_points_c + 2)
    CDF_Z = np.zeros(n_points_z + 2)
    # print("zspace",z_space)
    # print("cspace",c_space)
    CDF_Z[1:n_points_z + 1] = beta.cdf(z_space[1:n_points_z + 1], alpha_z, beta_z)
    CDF_C[1:n_points_c + 1] = beta.cdf(c_space[1:n_points_c + 1], alpha_c, beta_c)
    # j = n_points_c
    # CDF_C[j] = beta.cdf((c_space[j - 1] + 3 * c_space[j]) / 4.0, alpha_c, beta_c)
    # i = n_points_z
    # CDF_Z[i] = beta.cdf((z_space[i - 1] + 3 * z_space[i]) / 4.0, alpha_z, beta_z)
    if (rho == 0): type = "independent"
    # print(CDF_C)
    X,Y = np.meshgrid(CDF_C, CDF_Z)
    # print("X",X)
    # print("Y",Y)
    if (type == "independent"): 
        CDF_multi=np.multiply(X,Y)
        return CDF_C, CDF_Z, CDF_multi
    if (type == "frank"):
        frankparameters=parameters['frank']
        index = int((rho + 1) / 0.01)
        if (index == 200):
            alpha = frankparameters[index]
        elif (round(rho, 2) >= rho):
            alpha = ((round(rho, 2) - rho) * frankparameters[index + 1] + (rho - round(rho, 2) + 0.01) *
                     frankparameters[index]) * 100
        else:
            alpha = ((rho - round(rho, 2)) * frankparameters[index + 1] + (round(rho, 2) - rho + 0.01) *
                     frankparameters[index]) * 100
        if (alpha > 35):   alpha = 35
        CDF_multi = -(1/alpha)*np.log(1+(np.multiply((np.exp(-alpha*X)-1),(np.exp(-alpha*Y)-1)))/(np.exp(-alpha)-1))
        return CDF_C, CDF_Z,CDF_multi
def dPsicomputing(Psi, dPsidc, c_space, c_space_average_use, z_space, z_space_average, z_space_diff, CDF_C, CDF_Z,n_points_z, n_points_c, bias):
    Q_int = np.zeros(n_points_z + 2)
    dQdz = np.zeros(n_points_z + 2)
    # time4=time.time()
    CDF_C_use = np.reshape(np.tile(CDF_C, n_points_z + 1), (n_points_z + 1, n_points_c + 2))
    Q_int[1:n_points_z + 1] = Q_int[1:n_points_z + 1] - np.sum(0.5 * np.multiply((np.multiply(dPsidc[1:n_points_z + 1, 1:n_points_c - 1], CDF_C_use[1:n_points_z + 1, 1:n_points_c - 1]) + np.multiply(dPsidc[1:n_points_z + 1, 2:n_points_c], CDF_C_use[1:n_points_z + 1, 2:n_points_c])), (c_space_average_use[1:n_points_z + 1,2:n_points_c] - c_space_average_use[1:n_points_z + 1,1:n_points_c - 1])),axis=1)
    Q_int[1:n_points_z + 1] = Q_int[1:n_points_z + 1] - np.multiply(np.multiply(dPsidc[1:n_points_z + 1, n_points_c], np.repeat(CDF_C[n_points_c], n_points_z)),np.repeat((c_space[n_points_c] - c_space[n_points_c - 1]) / 2.0, n_points_z)) - np.multiply(np.multiply(dPsidc[1:n_points_z + 1, n_points_c + 1], np.repeat(CDF_C[n_points_c + 1], n_points_z)),np.repeat((c_space[2] - c_space[1]) / 2.0, n_points_z)) + Psi[1:n_points_z + 1,n_points_c]
    dQdz[1:n_points_z] = (Q_int[2:n_points_z + 1] - Q_int[1:n_points_z] + bias) / z_space_diff[1:n_points_z]
    dQdz[n_points_z] = dQdz[n_points_z - 1]
    dQdz[n_points_z + 1] = dQdz[1]
    yint = -np.sum(
        0.5 * (dQdz[1:n_points_z - 1] * CDF_Z[1:n_points_z - 1] + dQdz[2:n_points_z] * CDF_Z[2:n_points_z]) * (
                    z_space_average[2:n_points_z] - z_space_average[1:n_points_z - 1]))
    yint = yint + dQdz[n_points_z] * CDF_Z[n_points_z] * (z_space[n_points_z] - z_space[n_points_z - 1]) / 2.0 + dQdz[
        n_points_z + 1] * CDF_Z[n_points_z + 1] * (z_space[2] - z_space[1]) / 2.0 + Q_int[n_points_z]
    # print("yint",yint)
    # time7=time.time()
    # print("Calculate4 time",time7-time6)
    return yint
def int_point_ind(z_mean, c_mean, c_var, z_var,z_space, c_space, z_space_average, c_space_average, z_space_diff,
              c_space_diff, c_space_average_use, Psi, YiPsi, dPsidc, dYiPsidc, n_points_z, n_points_c, nScalars, nYis):
    y_int = np.zeros(nScalars + 1)
    Yi_int = np.zeros(nYis + 1)
    alpha_z = z_mean * (((z_mean * (1 - z_mean)) / z_var) - 1)
    alpha_c = c_mean * (((c_mean * (1 - c_mean)) / c_var) - 1)
    beta_z = (1 - z_mean) * (((z_mean * (1 - z_mean)) / z_var) - 1)
    beta_c = (1 - c_mean) * (((c_mean * (1 - c_mean)) / c_var) - 1)
    CDF_C, CDF_Z= CDF_ind(z_space, c_space, z_space_average, c_space_average, alpha_z, beta_z, alpha_c, beta_c,
                                n_points_c, n_points_z)
    for k in range(2, nScalars + 1):
        # print(k)
        if (k < 5):
            y_int[k] = dPsicomputing(Psi[k], dPsidc[k], c_space, c_space_average_use, z_space, z_space_average,
                                     z_space_diff, CDF_C, CDF_Z, n_points_z, n_points_c, 1e-15)
        else:
            y_int[k] = dPsicomputing(Psi[k], dPsidc[k], c_space, c_space_average_use, z_space, z_space_average,
                                     z_space_diff, CDF_C, CDF_Z, n_points_z, n_points_c, 0)
    for k in range(1, nYis + 1):
        Yi_int[k] = dPsicomputing(YiPsi[k], dYiPsidc[k], c_space, c_space_average_use, z_space, z_space_average,
                                  z_space_diff, CDF_C, CDF_Z, n_points_z, n_points_c, 0)
    # time4=time.time()
    # print("dPsi time",time4-time3)
    y_int = np.where(abs(y_int) > 1e-20, y_int, 0)
    Yi_int = np.where(Yi_int > 1e-20, Yi_int, 0)
    # print("yint",y_int,Yi_int)
    return y_int, Yi_int
def int_point_copula(z_mean,c_mean,c_var,z_var,rho,z_space,c_space,Psi_compute,YiPsi_compute,n_points_z,n_points_c,nScalars,nYis,type,parameters):
    y_int=np.zeros(nScalars+1)
    Yi_int=np.zeros(nYis+1)
    alpha_z = z_mean * (((z_mean * (1-z_mean)) / z_var) - 1)
    alpha_c = c_mean * (((c_mean * (1-c_mean)) / c_var) - 1)
    beta_z = (1-z_mean) * (((z_mean * (1-z_mean)) / z_var) - 1)
    beta_c = (1-c_mean) * (((c_mean * (1-c_mean)) / c_var) - 1)
    CDF_C, CDF_Z, CDF_multi = CDF_copula(z_space, c_space,alpha_z, beta_z, alpha_c, beta_c, n_points_c, n_points_z,type,rho,parameters)
    CDF_multi_compute=np.zeros((n_points_z-1,n_points_c-1))
    CDF_multi_compute=(CDF_multi[1:n_points_z,1:n_points_c]+CDF_multi[2:n_points_z+1,2:n_points_c+1]-CDF_multi[1:n_points_z,2:n_points_c+1]-CDF_multi[2:n_points_z+1,1:n_points_c])
    # print(CDF_multi)
    for k in range(5,10):
        y_int[k] = np.sum(np.multiply(Psi_compute[k],CDF_multi_compute))
    # for k in range(1,nYis+1):
    #     Yi_int[k] = np.sum(np.multiply(YiPsi_compute[k],CDF_multi_compute))
    y_int = np.where(abs(y_int) > 1e-20, y_int, 0)
    Yi_int = np.where(Yi_int > 1e-20, Yi_int, 0)
    return y_int,Yi_int
def multiprocessingpdf(cbDict):
    start=time.time()
    pool=mp.Pool(processes=cbDict['n_procs'])
    pool2=mp.Pool(processes=cbDict['n_procs'])
    paramlists=[]
    ih=cbDict['n_points_h']
    z_space,c_space,Src_vals,Yi_vals=readspace(cbDict,ih)
    Psi=np.zeros((cbDict['nScalars']+1,cbDict['n_points_z'] + 2, cbDict['n_points_c'] + 2))
    YiPsi=np.zeros((cbDict['nYis']+1,cbDict['n_points_z'] + 2, cbDict['n_points_c'] + 2))
    Psi_compute=np.zeros((cbDict['nScalars']+1,cbDict['n_points_z'] -1, cbDict['n_points_c'] -1))
    YiPsi_compute=np.zeros((cbDict['nYis']+1,cbDict['n_points_z'] -1, cbDict['n_points_c'] -1))
    for k in range(2, cbDict['nScalars'] + 1):
        Psi[k,1:cbDict['n_points_z']+1,1:cbDict['n_points_c']+1]=Psicomputing(Src_vals[k],Src_vals[1],Src_vals[2],Src_vals[cbDict['nScalars']],c_space,z_space,cbDict['n_points_z'],cbDict['n_points_c'],cbDict['nScalars'],cbDict['nYis'],k)
    for k in range(1,cbDict['nYis']+1):
        YiPsi[k,1:cbDict['n_points_z']+1,1:cbDict['n_points_c']+1]=Psicomputing(Yi_vals[k],Src_vals[1],Src_vals[2],Src_vals[cbDict['nScalars']],c_space,z_space,cbDict['n_points_z'],cbDict['n_points_c'],cbDict['nScalars'],cbDict['nYis'],1)
    dPsidc = np.zeros((cbDict['nScalars'] + 1, cbDict['n_points_z'] + 2, cbDict['n_points_c'] + 2))
    dYiPsidc = np.zeros((cbDict['nYis'] + 1, cbDict['n_points_z'] + 2, cbDict['n_points_c'] + 2))
    c_space_diff = computedata(c_space, "diff", cbDict['n_points_c'])
    c_space_diff_use = np.reshape(np.tile(c_space_diff, cbDict['n_points_z'] + 1),(cbDict['n_points_z'] + 1, cbDict['n_points_c'] + 1))
    for k in range(2, cbDict['nScalars'] + 1):
        Psi[k, 1:cbDict['n_points_z'] + 1, 1:cbDict['n_points_c'] + 1] = Psicomputing(Src_vals[k], Src_vals[1],Src_vals[2],Src_vals[cbDict['nScalars']],c_space, z_space,cbDict['n_points_z'],cbDict['n_points_c'],cbDict['nScalars'],cbDict['nYis'], k)
    for k in range(1, cbDict['nYis'] + 1):
        YiPsi[k, 1:cbDict['n_points_z'] + 1, 1:cbDict['n_points_c'] + 1] = Psicomputing(Yi_vals[k], Src_vals[1],Src_vals[2],Src_vals[cbDict['nScalars']],c_space, z_space,cbDict['n_points_z'],cbDict['n_points_c'],cbDict['nScalars'],cbDict['nYis'], 1)
    for k in range(2, cbDict['nScalars'] + 1):
        if (k < 5):
            dPsidc[k, 1:cbDict['n_points_z'] + 1, 1:cbDict['n_points_c']] = (Psi[k, 1:cbDict['n_points_z'] + 1,2:cbDict['n_points_c'] + 1] - Psi[k,1:cbDict['n_points_z'] + 1,1:cbDict['n_points_c']] + 1e-15) / c_space_diff_use[1:cbDict['n_points_z'] + 1,1:cbDict['n_points_c']]
        else:
            dPsidc[k, 1:cbDict['n_points_z'] + 1, 1:cbDict['n_points_c']] = (Psi[k, 1:cbDict['n_points_z'] + 1,2:cbDict['n_points_c'] + 1] - Psi[k,1:cbDict['n_points_z'] + 1,1:cbDict['n_points_c']]) / c_space_diff_use[1:cbDict['n_points_z'] + 1,1:cbDict['n_points_c']]
        dPsidc[k, 1:cbDict['n_points_z'] + 1, cbDict['n_points_c']] = dPsidc[k, 1:cbDict['n_points_z'] + 1,cbDict['n_points_c'] - 1]
        dPsidc[k, 1:cbDict['n_points_z'] + 1, cbDict['n_points_c'] + 1] = dPsidc[k, 1:cbDict['n_points_z'] + 1, 1]
    for k in range(1, cbDict['nYis'] + 1):
        dYiPsidc[k, 1:cbDict['n_points_z'] + 1, 1:cbDict['n_points_c']] = (YiPsi[k, 1:cbDict['n_points_z'] + 1,2:cbDict['n_points_c'] + 1] - YiPsi[k,1:cbDict['n_points_z'] + 1,1:cbDict['n_points_c']]) / c_space_diff_use[1:cbDict['n_points_z'] + 1,1:cbDict['n_points_c']]
        dYiPsidc[k, 1:cbDict['n_points_z'] + 1, cbDict['n_points_c']] = dYiPsidc[k, 1:cbDict['n_points_z'] + 1,cbDict['n_points_c'] - 1]
        dYiPsidc[k, 1:cbDict['n_points_z'] + 1, cbDict['n_points_c'] + 1] = dYiPsidc[k, 1:cbDict['n_points_z'] + 1, 1]
    n_points_z=cbDict['n_points_z']
    n_points_c=cbDict['n_points_c']
    for k in range(2,cbDict['nScalars']+1):
        Psi_compute[k]=(1/4)*(Psi[k,1:n_points_z,1:n_points_c]+Psi[k,2:n_points_z+1,2:n_points_c+1]+Psi[k,1:n_points_z,2:n_points_c+1]+Psi[k,2:n_points_z+1,1:n_points_c])
    for k in range(1,cbDict['nYis']+1):
        YiPsi_compute[k]=(1/4)*(YiPsi[k,1:n_points_z,1:n_points_c]+YiPsi[k,2:n_points_z+1,2:n_points_c+1]+YiPsi[k,1:n_points_z,2:n_points_c+1]+YiPsi[k,2:n_points_z+1,1:n_points_c])
    # if(cbDict['pdf_type']=="independent"):
    for iz in range(1,cbDict['int_pts_z']+1):#1,cbDict['int_pts_z']+1
            paramlists.append((cbDict,iz,ih,z_space,c_space,Src_vals,Yi_vals,Psi_compute,YiPsi_compute,Psi,YiPsi,dPsidc,dYiPsidc))
    res=pool.map(pdf_multi,paramlists)
    pool.close()
    pool.join()
    print("time cost ",time.time()-start," s")
    return 0
    # if(cbDict['pdf_type']=="frank"):
    #     for iz in range(10,11):#1,cbDict['int_pts_z']+1
    #         str_iz='%02d' % iz
    #         str_ih='%02d' % ih
    #         for igcz in range(1,cbDict['int_pts_gcz']+1):#cbDict['int_pts_gcz']+1
    #             paramlists.append((cbDict,iz,ih,z_space,c_space,Src_vals,Yi_vals,Psi_compute,YiPsi_compute))
    #     res=pool2.map(pdf_multi,paramlists)
    #     pool2.close()
    #     pool2.join()
    #     for iz in range(10,11):#1,cbDict['int_pts_z']+1
    #         str_iz='%02d' % iz
    #         str_ih='%02d' % ih
    #         strs=[]
    #         gcz_int=cbDict['gcz']
    #         for k in range(1,cbDict['int_pts_gcz']+1):
    #             rho=gcz_int[k]
    #             str_rho = '%.1f' % rho
    #             fp=open('unit'+str_iz+'_h'+str_ih+'_rho_'+str_rho+'.dat',"r")
    #             strs.append(fp.readlines())
    #             fp.close()
    #         with open('unit'+str_iz+'_h'+str_ih+'.dat',"a") as fl:
    #             for i in range(len(strs[0])):
    #                 for k in range(cbDict['int_pts_gcz']):
    #                     fl.write(strs[k][i])
    #         fl.close()
    #         for k in range(1,cbDict['int_pts_gcz']+1):
    #             rho=gcz_int[k]
    #             str_rho = '%.1f' % rho
    #             os.remove('unit'+str_iz+'_h'+str_ih+'_rho_'+str_rho+'.dat')
def pdf_multi(item):
    cbDict,iz,ih,z_space,c_space,Src_vals,Yi_vals,Psi_compute,YiPsi_compute,Psi,YiPsi,dPsidc,dYiPsidc=item
    parameters = {}
    parameters['frank'] = readcopula("./frankparameters.mat")
    parameters['placket'] = readcopula("./placketparameters.mat")
    integrate(cbDict,z_space,c_space,Src_vals,Yi_vals,iz,ih,parameters,Psi_compute,YiPsi_compute,Psi,YiPsi,dPsidc,dYiPsidc)
def integrate(cbDict,z_space,c_space,Src_vals,Yi_vals,iz,ih,parameters,Psi_compute,YiPsi_compute,Psi,YiPsi,dPsidc,dYiPsidc):
        start = time.time()
        yint = np.zeros(cbDict['nScalars'] + 1)
        Yi_int = np.zeros(cbDict['nYis'] + 1)
        p=0
        str_iz='%02d' % iz
        str_ih='%02d' % ih
        z_int=cbDict['z']
        c_int=cbDict['c']
        gz_int=cbDict['gz']
        gc_int=cbDict['gc']
        gcz_int=cbDict['gcz']
        #   rho=gcz_int[igcz]
        #   str_rho = '%.1f' % rho
        z_space_average = computedata(z_space, "average", cbDict['n_points_z'])
        z_space_diff = computedata(z_space, "diff", cbDict['n_points_z'])
        c_space_average = computedata(c_space, "average", cbDict['n_points_c'])
        c_space_diff = computedata(c_space, "diff", cbDict['n_points_c'])
        c_space_average_use = np.reshape(np.tile(c_space_average, cbDict['n_points_z'] + 1),
                                        (cbDict['n_points_z'] + 1, cbDict['n_points_c'] + 1))
        #   print(cbDict["pdf_type"])
        #   if(cbDict["pdf_type"]=="independent"):
        f = open('./canteraData/unit'+str_iz+'_h'+str_ih+'.dat', "w")
    #   else:
    #     f = open('unit'+str_iz+'_h'+str_ih+'_rho_'+str_rho+'.dat', "w")
        p=0
        z_loc = locate(z_space, cbDict['n_points_z'], z_int[iz])
        for ic in range(1,cbDict['int_pts_c']+1):#1,cbDict['int_pts_c']+1
            for igz in range(1,cbDict['int_pts_gz']+1):#1,cbDict['int_pts_gz']+1
                for igc in range(1,cbDict['int_pts_gc']+1):#1,cbDict['int_pts_gc']+1
                    p=p+1
                    #print("computing unit"+str_iz+" case ",p," ",iz," ",ic," ",igz," ",igc," ",0)#显示计算进程，嫌烦这行可以删除
                    if((iz==1) or (iz==cbDict['int_pts_z'])):
                        yint[2]=0
                        yint[3]=0
                        yint[4]=0
                        if (iz == 1): z_loc=1
                        if (iz == cbDict['int_pts_z']): z_loc=cbDict['n_points_z']
                        for i in range(5,cbDict['nScalars']+1):
                            yint[i]=Src_vals[i][z_loc][1]
                        for i in range(1,cbDict['nYis']+1):
                            Yi_int[i]=Yi_vals[i][z_loc][1]
                    elif(((igz == 1 and igc == 1) or (igz == 1 and ic == 1)) or (igz == 1 and ic == cbDict['int_pts_c'])):
                        yint,Yi_int=delta(z_int[iz], c_int[ic], z_space, c_space,Src_vals,Yi_vals,cbDict['n_points_z'],cbDict['n_points_c'],cbDict['nScalars'],cbDict['nYis'])
                    elif(igz==1 and igc>1):
                        yint,Yi_int =cbeta(c_int[ic],gc_int[igc],c_space,c_space_average,c_space_diff,z_int[iz],z_space,Src_vals,Yi_vals,cbDict['n_points_z'],cbDict['n_points_c'],cbDict['nScalars'],cbDict['nYis'])
                    elif(igz>1 and igc==1):
                        yint,Yi_int =zbeta(z_int[iz],gz_int[igz],z_space,z_space_average,z_space_diff,c_int[ic],c_space,Src_vals,Yi_vals,cbDict['n_points_z'],cbDict['n_points_c'],cbDict['nScalars'],cbDict['nYis'])
                    else:
                        k=0
                        if((ic == 1) or (ic == cbDict['int_pts_c'])): k=1
                        if(k==0):
                            c_var=gc_int[igc]*(c_int[ic]*(1.0-c_int[ic]))
                            z_var=gz_int[igz]*(z_int[iz]*(1.0-z_int[iz]))
                            yint,Yi_int=int_point_ind(z_int[iz],c_int[ic],c_var,z_var,z_space,c_space,z_space_average,c_space_average,z_space_diff,c_space_diff,c_space_average_use, Psi, YiPsi, dPsidc, dYiPsidc,cbDict['n_points_z'],cbDict['n_points_c'],cbDict['nScalars'],cbDict['nYis'])#cbDict['pdf_type']
                    if(cbDict["pdf_type"]=="independent"):
                        data=[z_int[iz],c_int[ic],gz_int[igz],gc_int[igc],0]+[x for x in yint[2:(cbDict['nScalars']+1)]]+[x for x in Yi_int[1:]]
                        for i in range(0,len(data)):
                            f.write(str('%0.5E' % data[i]))
                            f.write(" ")
                        f.write("\n")
                    if(cbDict["pdf_type"]=="frank"):
                        # yint_use,Yi_int_use=yint,Yi_int
                        for igcz in range(1,cbDict['int_pts_gcz']+1):
                            if(igcz==((cbDict['int_pts_gcz']/2)+1)):
                                data=[z_int[iz],c_int[ic],gz_int[igz],gc_int[igc],gcz_int[igcz]]+[x for x in yint[2:(cbDict['nScalars']+1)]]+[x for x in Yi_int[1:]]
                            else:
                                yint_use,Yi_int_use=yint,Yi_int
                                if(igz!=1 and igc!=1 and iz>1 and iz<cbDict['int_pts_z'] and ic>1 and ic<cbDict['int_pts_c']):
                                    c_var=gc_int[igc]*(c_int[ic]*(1.0-c_int[ic]))
                                    z_var=gz_int[igz]*(z_int[iz]*(1.0-z_int[iz]))
                                    yint_copula,Yi_int_copula=int_point_copula(z_int[iz],c_int[ic],c_var,z_var,gcz_int[igcz],z_space,c_space,Psi_compute,YiPsi_compute,cbDict['n_points_z'],cbDict['n_points_c'],cbDict['nScalars'],cbDict['nYis'],cbDict['pdf_type'],parameters)
                                    yint_use[5:10]=yint_copula[5:10]
                                data=[z_int[iz],c_int[ic],gz_int[igz],gc_int[igc],gcz_int[igcz]]+[x for x in yint_use[2:(cbDict['nScalars']+1)]]+[x for x in Yi_int_use[1:]]
                            for i in range(0,len(data)):
                                f.write(str('%0.5E' % data[i]))
                                f.write(" ")
                            f.write("\n")
        # print("Writing done ",'unit'+str_iz+'_h'+str_ih+'_rho_'+str_rho+'.dat')
        # print("写该文件总耗时"," ",time.time()-start,"s")
        f.close()


# ### 2.6 组装表格
# 计算完成，在PDF计算中，会将Z-C标量空间的信息映射到Z-C-gz-gc空间，组装表格针对上述信息做了一些缩减工作。

# In[ ]:


# 组装表格

import numpy as np

def assemble(cbDict,solFln):

    for i in range(cbDict["n_points_h"]):
        #n_points_h=1
        print('Reading unit' + '%02d' % 1 + '_h' + '%02d' % (i+1) + ' ... \n')
        M = np.loadtxt(solFln + "/" + 'unit01_h' + '%02d' % (i+1) + '.dat')

        for j in range(1,cbDict["int_pts_z"]):
            print('Reading unit' + '%02d' % (j+1) + '_h' + '%02d' % (i+1) + ' ... \n')
            tmp = np.loadtxt(solFln + "/" +  'unit' + '%02d' % (j+1) + '_h' + '%02d' % (i+1) + '.dat')
            M = np.insert(tmp,0,M,axis=0) 

    # remove unwanted columns - h(12),qdot(13),Yc_max(14)
    n_column = np.shape(tmp)[1]
    if(cbDict['scaled_PV']): 
        rm_list = [n_column-cbDict['nYis']-3,n_column-cbDict['nYis']-2,n_column-cbDict['nYis']-1] 
    else:
        rm_list = [n_column-cbDict['nYis']-3,n_column-cbDict['nYis']-2]
    MM = np.delete(M,rm_list,axis=1)

    # write assembled table
    fln = solFln + "/" +   cbDict['output_fln'] 
    print('Writing assembled table ...')
    with open(fln,'a') as strfile:
        #写入混合物分数z
        strfile.write(str(cbDict['int_pts_z']) + '\n')
        np.savetxt(strfile,cbDict['z'][1:],fmt='%.5E',delimiter='\t') 

        #写入进程变量c
        strfile.write(str(cbDict['int_pts_c']) + '\n')
        np.savetxt(strfile,cbDict['c'][1:],fmt='%.5E',delimiter='\t')
        
        #写入混合物分数z的方差gz
        strfile.write(str(cbDict['int_pts_gz']) + '\n')
        np.savetxt(strfile,cbDict['gz'][1:],fmt='%.5E',delimiter='\t')

        #写入进程变量c的方差gc
        strfile.write(str(cbDict['int_pts_gc']) + '\n')
        np.savetxt(strfile,cbDict['gc'][1:],fmt='%.5E',delimiter='\t')

        #写入混合物分数z和进程变量c的协方差gcz
        strfile.write(str(cbDict['int_pts_gcor']) + '\n')
        np.savetxt(strfile,cbDict['gcz'][1:],fmt='%.5E',delimiter='\t')  

        #写入MM的第5列及以后的数据
        strfile.write(str(MM.shape[1]-5-cbDict['nYis']) + '\t' +
                        str(cbDict['nYis']) + '\n')
        np.savetxt(strfile,MM[:,5:],fmt='%.5E',delimiter='\t') 

        #写入截断后的混合物分数的概率密度分布
        if(cbDict['scaled_PV']):
            d2Yeq_table = np.loadtxt(solFln + "/" + 'd2Yeq_table.dat')
            np.savetxt(strfile,d2Yeq_table,fmt='%.5E')
    strfile.close()

    print("\n Done writing flare.tbl")


# ### 2.7 主函数运行

# In[ ]:


# 下面是主函数和相关运行结果
import imp
from numpy import array, double, float64
from pathlib import Path
import time
from typing import List
import ast
import os
import numpy as np
if __name__ == "__main__":

    #-------------load commonDict.txt--------------#
    
    start=time.time()

    cbDict = read_commonDict("commonDict.txt") 
    
    # print(cbDict['c'])
    # print(cbDict['work_dir'])
    solFln = ('./canteraData/')
    
    canteraSim(cbDict,solFln)
    
    ntable(cbDict,solFln)
    
    
    interpLamFlame(cbDict,solFln)
    # os.chdir('./canteraData/')
    cbDict['z'] = np.append([0],cbDict['z'])
    cbDict['c'] = np.append([0],cbDict['c'])
    cbDict['gz'] = np.append([0],cbDict['gz'])
    cbDict['gc'] = np.append([0],cbDict['gc'])
    cbDict['gcz'] = np.append([0],cbDict['gcz'])

    
    
    multiprocessingpdf(cbDict)
    
    assemble(cbDict,solFln)
    end=time.time()
    print('Running time: %s Seconds'%(end-start))


# ## 三、结果展示
# 火焰面数据表格部分结果展示（chemTable.dat）
# 
# 

# - Z-C的温度等值面

# <img src="https://bohrium.oss-cn-zhangjiakou.aliyuncs.com/article/19016/11fb0e8b45d647d4a6a767bf2a57b4ea/ub6Wx-UD_G5D_POePMGvpA/d7HR0nV8rvPNGFmnGdDW3g.png" width="300" height="200" alt="Z-C的温度等值面">
# 

# - Z-C的水质量分数等值面

# <img src="https://bohrium.oss-cn-zhangjiakou.aliyuncs.com/article/19016/11fb0e8b45d647d4a6a767bf2a57b4ea/O1yuKLI6bfGOMuCYIleboQ/7eNnVLACl38GXQ-QjxTCxA.png" width="300" height="200" alt="Z-C的水质量分数等值面">
# 
# 
# 

# - 拉伸率和最大温度对应关系（稳态分布）

# <img src="https://bohrium.oss-cn-zhangjiakou.aliyuncs.com/article/19016/11fb0e8b45d647d4a6a767bf2a57b4ea/3Zt_cV2tBM0Rrnv9zTyv_A/vgxuvwDtRps8a9xR-oblvQ.png" width="300" height="200" alt="拉伸率和最大温度对应关系（稳态分布）">
# 

# OpenFOAM相关计算结果（三维流场——Sandia FlameD）
# 
# 

# - 模拟实验对比结果

# <img src="https://bohrium.oss-cn-zhangjiakou.aliyuncs.com/article/19016/11fb0e8b45d647d4a6a767bf2a57b4ea/mkDu-TyRVD5csDNmlilZog/2Y_GDiSKwY4Uk1yJW30pAw.png" width="300" height="200" alt="flameD 模拟实验对比结果">

# - 进度变量源项等值线图

# <img src="https://bohrium.oss-cn-zhangjiakou.aliyuncs.com/article/19016/11fb0e8b45d647d4a6a767bf2a57b4ea/vSvlp5zfX3PzkdJeuJDJGQ/bdFbJMvdOeUfRFPC7DMcuA.png" width="300" height="200" alt="flameD 进度变量源项等值线图">

# 可以发现3D-FPV可以取得和4D-FGM建表相近的结果，同时比3D-FGM的更高。（3D建表比4D的内存更小，读取速度更快）

# - OpenFOAM的算例运行请参考deepFlame官网[deepFlame](https://github.com/deepmodeling/deepflame-dev "deepFlame")
# - 将FPV建表生成的flare.tbl导入到算例中，即可使用FPV的建表结果
