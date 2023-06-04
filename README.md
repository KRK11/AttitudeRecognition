# 多目标人体姿态识别

## 简介

这是一个基于CPN（Cascaded Pyramid Network）和YOLOv8实现的多目标人体姿态识别应用，可以通过视频进行实时的识别人体的十七个关键点。



## 快速运行：

首先从github上克隆整个项目：

```
git clone https://github.com/KRK11/AttitudeRecognition.git
```

接着转到先目录下后安装所需要的包：

```
pip install requirements.txt
```

**注意**：requirements.txt文件中注释掉了有关pytorch的下载，请自行安装pytorch。

快速使用视频实时检测功能：

```
python run.py --model x --source y
```

其中x为训练好的模型（可以填入data/my9-3.pth），y为视频地址或者为0（代表摄像头输入）





## 目录结构介绍：

（**注意：在github上由于无法上传超过100M的文件，因此以下目录结构可能会有部分缺失，同时由于模型文件大于100M，因此暂时无法上传模型文件**）

- 📂 **AttitudeRecognition** - 主项目目录
  - 📂 **cpn** - CPN模型相关代码
    - 📂 **resnet** - ResNet模型代码
      - 📄 bottle_neck.py - 残差网络的块定义
      - 📄 conv.py - 卷积操作定义
      - 📄 resnet.py - ResNet模型定义
      - 📄 restnet50.pth - ResNet50预训练模型
    - 📄 global_net.py - global网络定义
    - 📄 network.py - 网络定义
    - 📄 refine_net.py - refine网络定义
  - 📂 **data** - 存储训练结果的文件，其中已经预训练完成3个模型
    - 📄 my9-3.pth - 模型文件
    - 📄 my18-4.pth - 模型文件
    - 📄 original7-5.pth - 模型文件
  - 📂 **data_generate** - 数据集生成
    - 📄 version1.py - 数据集生成版本1
    - 📄 version2.py - 数据集生成版本2
    - 📄 version3.py - 数据集生成版本3
    - 📄 version4.py - 数据集生成版本4
    - 📄 version5.py - 数据集生成版本5
  - 📂 **image** - 图像素材文件夹
  - 📂 **result_analysis** - 结果分析文件夹，保存对结果进行分析的多份代码
    - 📄 analysis.py - 可视化分析代码
    - 📄 coco_calculate.py - COCO指标计算代码
    - 📄 my_calculate.py - 可视化分析代码
    - 📄 mytest.py - 生成label文件脚本
  - 📂 **run** - 运行结果保存文件夹
    - 📂 **my9-3 lim 0.18** - 对应模型阈值为0.18下的视频运行结果
    - 📂 **my18-4 lim 0.004** - 对应模型阈值为0.004下的视频运行结果
    - :video_camera: **merge.mp4** - 已经完成的一个有趣的视频结果
  - 📂 **test** - 测试相关
    - 📂 **DataSet** - 多轮训练后的数据相关
    - 📂 **mAP0.6-5** - mAP为0.605的result.json文件
  - 📂 **utils** - 工具函数
    - 📄 image_utils.py - 图像处理工具
    - 📄 model_utils.py - 模型处理工具
    - 📄 os.utils.py - 系统操作工具
  - 📂 **venv** - 虚拟环境，其中已经安装了部分包
  - 📂 **video** - 视频素材文件夹
  - 📄 arguments.py - 训练，预测，运行等控制台参数解析文件
  - 📄 dataset.py - 数据加载文件
  - 📄 main.py - 各类冗杂代码，为编写过程中测试各个函数功能的草稿
  - 📄 predict.py - 预测脚本
  - 📄 requirements.txt - 依赖文件
  - 📄 run.pt - 运行文件
  - 📄 test.py - 测试脚本



## 可运行文件

可根据以下命令行获取对应的控制台命令。

### train.py

```
python train.py --help
```

### test.py

```
python test.py --help
```

### predict.py

```
python predict.py --help
```

### run.py

```
python run.py --help
```



## 训练

标准文件格式或者数据集格式可以在dataset中进行修改，也可以通过data_generate下的不同版本生成对应数据集，但是格式均是一致的。等有时间再完善下面部分。
