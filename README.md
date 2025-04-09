# 路径跟踪与避障环境用于深度强化学习控制

此仓库实现了一个用于AUV（自主水下航行器）的6自由度仿真模型，符合Stable baseline3（OpenAI）接口，用于强化学习控制。环境包含一个3D路径、障碍物以及海洋流干扰。代理的目标是引导AUV沿路径航行，同时克服干扰并避开轨迹上的障碍物。

## 快速开始

在虚拟环境中安装所需的所有软件包，请运行：

```
conda env create -f environment.yml
```

### 训练代理：

使用原项目 (https://github.com/ThomasNLarsen/gym-auv-3D)
进行单机控制器agent训练，放入log文件夹
项目中已有2种agent，第一个适用于PI控制器，第二个适用于PID控制器

## 在环境中运行代理

要在任何场景中运行代理，请使用：

```
python m_run3d.py --exp_id [x] --scenario [scenario] --controller_scenario [controller_scenario] --controller [y]
```

其中 x 是实验编号，scenario 是要运行的场景，controller_scenario 是控制器训练的场景，y 是要运行的代理编号。如果未提供 y，则选择名为 "last_model.pkl" 的代理。场景可以是 "m_beginner"（初学者）、"m_intermediate"（中级）、"m_proficient"（熟练）、"m_advanced"（高级）、"m_expert"（专家）、"m_test_path"（测试路径）、"m_test_path_current"（带干扰的路径跟踪）、"horizontal"（水平）、"vertical"（垂直）或 "deadend"（死胡同）。
