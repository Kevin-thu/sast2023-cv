# SAST 2023 Summer Training CV Assignment

本次作业是对之前所学 Python、PyTorch、VAE、Diffusion Models 等内容的一次综合实践，其主体采用**代码填空**的形式进行。你需要按照本实验指导中的流程，尝试阅读并理解现有的实验框架，然后在现有实验框架的基础上做代码补全，以实现对应子任务所要求的功能。

本次作业核心部分都在 `LatentDiffusion.ipynb` 中，部分模型定义放在 `src` 文件夹中。如果你有本地算力，你可以直接在本地运行 jupyter notebook 完成，也可以在 Colab 上完成。所有你需要填写或回答问题的部分都**使用 `TODO` 进行了标记**，大部分需要填空的部分都不超过三行，且提供了详细的线索指引，其中标记为 `Challenge` 的部分仅供感兴趣的同学选做。在作业公布三天后，我们会将一份参考实现放到上述仓库的另一个分支中。本次作业形式参考了去年由 [c7w](https://github.com/c7w) 布置的 [数据分析 & PyTorch 图像多分类作业](https://github.com/c7w/sast2022-pytorch-training/tree/master)，特此鸣谢。

我们本次作业的最终目标是训练一个 Unconditional Latent Diffusion Model，作为课上所学的生成模型的一次实践。本项目基于 [DiffusionFastForward](https://github.com/mikonvergence/DiffusionFastForward/tree/master) 开发，按照原作者指定的 LICENSE 使用，特此鸣谢。

具体而言，在本次作业中我们要实现以下内容：

+ SubTask 0：环境配置与安装（15 p.t.s）
+ SubTask 1：数据预处理（15 p.t.s）
+ SubTask 2：模型框架搭建（45 p.t.s）
+ SubTask 3：模型训练与可视化（25 p.t.s）
+ SubTask 4 (Optional)：代码整理与开源，分享结果（Bonus 10 p.t.s）

## 环境配置与安装（15 p.t.s）
### 准备 `Python` 环境（5 p.t.s）

我们在前面的课程已经学习过 `conda` 环境管理器的使用，你应该可以理解下面指令的作用。

```bash
conda create -n ldm
conda activate ldm
pip install -r requirements.txt
```

其中，由于 PyTorch 在不同系统和平台下安装方式略有不同，建议你根据 [PyTorch 官网](https://pytorch.org/) 指引选择合适的方式安装，然后删去 `requirements.txt` 中的第一行安装剩余依赖库。

### 准备数据集（5 p.t.s）
由于不同同学算力资源差距可能较大，本次作业不指定统一的数据集。两个推荐的数据集是：

1. [AFHQ 数据集](https://github.com/clovaai/stargan-v2/blob/master/README.md#animal-faces-hq-dataset-afhq)：在论文 [StarGAN v2](https://arxiv.org/abs/1912.01865) 中首次使用的数据集，包含约 15000 张 512*512 的高质量图片，主要是猫猫狗狗等可爱动物头像；
2. [MNIST-M 数据集](https://www.kaggle.com/datasets/aquibiqbal/mnistm?resource=download)：在论文 [Domain-Adversarial Training of Neural Networks](https://www.kaggle.com/datasets/aquibiqbal/mnistm?resource=download) 中首次使用的数据集，包含约 60000 张 28*28 的彩色手写数字图片。

预估使用 Colab 提供的单张 GPU 从零开始训练，使用第一个数据集至少需要 20 h，使用第二个数据集至少需要 5 h（仅为粗略估计，同学们可以自由探索，相互交流）。为方便同学下载，已将这两个数据集上传到[清华云盘](https://cloud.tsinghua.edu.cn/d/a747c0d1110d451099f9/)，请同学们遵守数据集主页上的 LICENSE 使用。下载后，放到合适的位置中，然后你可以在 argparse 处指定 train_dataset 的 default 值（在 notebook 中使用 argparse 仅仅是为了练习）。

### 配置默认参数（5 p.t.s）
在参数配置的代码框仿照示例形式为 parser 增加两个 arguments：lr 和 batch_size，为之后的训练做准备。

## 数据预处理（15 p.t.s）

在这一部分你将对数据集中的数据进行预处理。你需要首先了解 `torch.utils.data.Dataset` 和 `torch.utils.data.DataLoader` 的用法，然后按照 notebook 中的指引完成 `SimpleImageDataset` 类（5 + 5 p.t.s）和 `train_dataloader` （5 p.t.s）的定义。由于生成模型任务的特殊性，我们本次作业中并不会使用到 valid 和 test dataset。

## 模型框架搭建（45 p.t.s）
接下来你将完成核心模型框架的搭建。在 notebook 中运行 Model Training 之后的代码框前，请先完成 `src/LatentDiffusion.py` 和 `src/DenoisingDiffusionProcess/DenoisingDiffusionProcess.py` 中的代码填空部分。请按照 `TODO` 处提供的指引，阅读代码框架，查询相关资料，并完成代码填空或使用注释回答问题。每个代码填空处需要你补充的代码均只有一行，除特殊标记外每空均为 5 p.t.s。

## 模型训练与可视化（25 p.t.s）
首先，请采用 matplotlib 库进行可视化，检查你的数据集是否加载正确、加载的预训练 VAE 模型是否可以正确重建原图（10 p.t.s）。

然后，就可以开始~~快乐炼丹~~了！本次作业中使用了 PyTorch Lightning 提供的训练框架，你可以根据 TODO 中的指引对这一简单易用的框架进行简要了解。在这里，我们需要你自行了解与训练可视化和打日志相关的库，在模型中加入简单的 loss 或中间结果可视化代码，形式不限（只要写了就有 10 p.t.s，~~哪怕 print 也可以~~）。 虽然这不会直接影响你的训练过程，但对监视模型的训练非常重要。

在训练过程中，一定记得定时保存下中途训练得到的模型参数等信息（checkpoint），这样在训练意外停止后还可以接着上次的 checkpoint 继续训练，相关代码已经提供。由于本次作业非常消耗算力资源，在此，我们也非常鼓励算力资源丰富（~~家里有矿~~）的大佬在云盘或 huggingface 等平台分享自己训练得到的 checkpoint，其他同学可以在此基础上继续训练，这将为其他同学~~和地球环保~~做出巨大贡献，为此你也将获得一份丰厚的 Bonus。最后，训练结束后，请在 notebook 中留下你模型中的采样结果（5 p.t.s）。

在完成作业后，请将代码进行托管，如 GitHub， Tsinghua Git 等，然后在原仓库中新建 Issue，提交代码仓库地址。即使受限于硬件条件没有完成训练，你也可以获得大部分分数。

## Bonus: 代码整理与开源
请整理你的代码，留足充分的注释，撰写一份 README 作为报告向大家说明你使用的数据集、你的可视化结果和代码的使用方式，发布到开源社区。在向远程仓库推送文件时注意不要提交数据集与你的模型 checkpoint，checkpoint 的合理公开方式应该是放在云盘中并分享下载链接。也欢迎在微信群或其他社交平台分享你的训练结果或作业心得、反馈，~~让大家在猫猫狗狗的可爱图片中收获纯真快乐~~。