# Bird Eco DIP 鸟类生态图像处理系统

**Bird Eco DIP** 是一个基于经典数字图像处理（DIP）技术的鸟类羽毛图案分析系统，并将视觉特征与生态数据相关联。

## 项目简介

本项目处理 [CUB-200-2011](https://www.vision.caltech.edu/datasets/cub_200_2011/) 数据集（11,788 张图片，200 个物种），**仅使用 OpenCV 基础算子** —— 按照课程要求，分割与特征提取未使用任何深度学习大模型（SAM、YOLO 等）。

系统从每张图像中提取 7 个经典视觉特征（形状、颜色、纹理、对称性、分形维数），利用 PCA 进行降维与可视化，训练轻量级 Random Forest 分类器，并整合 [AVONET](https://figshare.com/s/b990722d72a26b5bfead) 生态数据集。

## 核心功能

- **图像预处理**：灰度转换、高斯滤波去噪、直方图均衡化增强对比度
- **图像分割**：Otsu 自动阈值 + 形态学操作（开运算/闭运算），纯 OpenCV 实现
- **特征提取（7维）**：
  - 形状特征：面积、周长、圆形度
  - 颜色特征：HSV 均值
  - 对称性：左右半图相关性
  - 纹理特征：GLCM 对比度与能量
  - 复杂度：分形维数（Box-counting 法）
- **PCA 分析**：将 10 维特征降至 2 维，绘制交互式散点图
- **物种分类**：轻量级 Random Forest 分类器
- **生态指标映射**：关联 AVONET 数据（喙长、翼长、跗蹠长、尾长）
- **Web 界面**：基于 Streamlit 的图形化界面，支持图片上传与实时分析

## 项目结构

```
bird_dip_project/
├── app.py                    # Streamlit 网页界面
├── config.py                 # 全局配置与路径
├── pipeline.py               # 批量处理脚本
├── requirements.txt          # Python 依赖包
├── src/                      # 核心源码
│   ├── preprocessing.py      # 图像预处理（OpenCV）
│   ├── segmentation.py       # Otsu + 形态学分割
│   ├── features.py           # 特征提取（7个指标）
│   ├── data_loader.py        # CUB-200-2011 标注加载
│   ├── pca_analysis.py       # PCA 降维分析
│   ├── classifier.py         # Random Forest 训练/预测
│   ├── visualization.py      # 绘图工具
│   ├── avonet_cleaner.py     # AVONET 数据清洗
│   └── eco_mapper.py         # CUB 与 AVONET 映射
├── data/                     # 数据集
│   ├── images/               # CUB-200-2011 图像（不上传仓库）
│   ├── annotations/          # CUB-200-2011 标注文件
│   └── avonet.csv            # AVONET 生态数据集
├── outputs/                  # 输出结果
│   ├── features.csv          # 提取的特征（11,788行）
│   ├── pca_results.csv       # PCA 投影结果
│   ├── eco_lookup.csv        # 生态指标查询表
│   └── models/               # 训练好的模型（.pkl）
├── assets/
│   └── logo.png              # 项目 Logo
└── tests/                    # 测试脚本
```

## 环境安装

### 1. 克隆仓库

```bash
git clone https://github.com/YOUR_USERNAME/bird_dip_project.git
cd bird_dip_project
```

### 2. 创建虚拟环境

```
python -m venv venv
```

#### Windows 系统激活：

```
venv\Scripts\activate
```

#### Mac/Linux 系统激活：

```
source venv/bin/activate
```

### 3. 安装依赖

```
pip install -r requirements.txt
```

### 4. 下载数据集

CUB-200-2011：从 Caltech 下载，解压到 data/images/ 和 data/annotations/
AVONET：从 Figshare 下载，保存为 data/avonet.csv

## 使用说明

### 批量处理（生成 features.csv）

```bash
python pipeline.py
```

### 启动 Web 界面

```bash
streamlit run app.py
```

然后在浏览器打开 http://localhost:8501

## 关键技术参数

| 参数名                 | 值     | 说明                            |
| ---------------------- | ------ | ------------------------------- |
| `SEED`                 | 42     | 随机种子，保证实验可复现        |
| `GAUSSIAN_BLUR_KERNEL` | (5, 5) | 高斯滤波核，用于去噪            |
| `MORPH_KERNEL_SIZE`    | (5, 5) | 形态学操作核，用于清理分割 mask |
| `PCA_COMPONENTS`       | 2      | PCA 降维后的维度数（用于可视化  |
| `RF_ESTIMATORS`        | 100    | Random Forest 决策树数量        |

## 技术栈

- **Python 3.10+**
- **OpenCV（经典图像处理算子）**
- **scikit-learn（PCA、Random Forest）**
- **scikit-image（GLCM 纹理特征）**
- **Streamlit（Web 界面）**
- **Plotly（交互式 PCA 可视化）**
- **Pandas / NumPy（数据处理）**

## 可复现性说明

所有随机操作均使用固定种子 SEED = 42。完整流程已在 11,788 张图像上运行，结果保存在 outputs/ 目录中。

## 许可证

本项目为南京师范大学《数字图像处理综合实践》课程作业。
