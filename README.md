# 路径跟踪与避障环境用于深度强化学习控制

此仓库实现了一个用于AUV（自主水下航行器）的6自由度仿真模型，符合稳定基线（OpenAI）接口，用于强化学习控制。环境包含一个3D路径、障碍物以及海洋流干扰。代理的目标是引导AUV沿路径航行，同时克服干扰并避开轨迹上的障碍物。

## 快速开始

在虚拟环境中安装所需的所有软件包，请运行：

```
conda env create -f environment.yml
```

### 训练代理：

所有超参数和设置可以在文件 [train.py](https://github.com/simentha/gym-auv/blob/master/train3d.py) 和 [__init__.py](https://github.com/simentha/gym-auv/blob/master/gym_auv/__init__.py) 中调整。

要训练代理，请运行：

```
python train.py --exp_id [x]
```

其中 x 是实验编号。

## 在环境中运行代理

要在任何场景中运行代理，请使用：

```
python run.py --exp_id [x] --scenario [scenario] --controller_scenario [controller_scenario] --controller [y]
```

其中 x 是实验编号，scenario 是要运行的场景，controller_scenario 是控制器训练的场景，y 是要运行的代理编号。如果未提供 y，则选择名为 "last_model.pkl" 的代理。场景可以是 "beginner"（初学者）、"intermediate"（中级）、"proficient"（熟练）、"advanced"（高级）、"expert"（专家）、"test_path"（测试路径）、"test_path_current"（带干扰的路径跟踪）、"horizontal"（水平）、"vertical"（垂直）或 "deadend"（死胡同）。
