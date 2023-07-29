# SAST 2023 Summer Training CV Assignment

本次作业是对学习的数据分析技能和人工神经网络知识的入门实践，其主体采用**代码填空**的形式进行。你需要按照本实验指导中的流程，尝试阅读并理解现有的实验框架，然后在现有实验框架的基础上做代码补全，以实现对应子任务所要求的功能。参考实现位于上述仓库的另一个分支中。

我们本次作业的最终目标是训练一个 Unconditional Latent Diffusion Model，作为生成模型的一次实践。

具体而言，在本次作业中我们要实现以下内容：

+ SubTask 0：环境配置与安装（15 p.t.s）
+ SubTask 1：数据预处理
+ SubTask 2：训练框架搭建
+ SubTask 3 (Optional)：代码整理与开源，分享结果

## 环境配置与安装（15 p.t.s）
### 准备 `Python` 环境（10 p.t.s）

我们在前面的课程已经学习过 `conda` 环境管理器的使用，你应该可以理解下面指令的作用。

```bash
conda create -n ldm
conda activate ldm
pip install -r requirements.txt
```


### 准备数据集（5 p.t.s）

