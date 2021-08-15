# 【图像分类】使用飞桨全流程开发工具PaddleX实现美食识别任务

## 一、写在前面

	随着人们的生活质量日益增长，美食成为人们的高质量生活不可或缺的一部分。遇见美食，吃定美食。面对无数的美食，你是否也会觉得目不暇接？为了能够更好的在家庭聚会上展示我的见识，我选择使用PaddleX来完成美食识别这个任务。


## 二、准备工作



```python
#!rm -rf output
#!rm -rf work/*
#!rm -rf inference_model
```

### 安装依赖


```python
# 安装PaddleX
!pip install paddlex -i https://mirror.baidu.com/pypi/simple
```

### 关于数据

> 数据集组织格式参照：[图像分类](https://paddlex.readthedocs.io/zh_CN/release-1.3/data/annotation/classification.html)


```python
# 解压数据集，放在work下使其可以永久保存
!unzip -oq /home/aistudio/data/data70793/foods.zip -d work/
```


```python
# 查看数据集文件结构
!tree work/foods -L 1
```

    work/foods
    ├── apple_pie
    ├── baby_back_ribs
    ├── baklava
    ├── beef_carpaccio
    └── beef_tartare
    
    5 directories, 0 files



```python
# 使用PaddleX划分数据集
!paddlex --split_dataset --format ImageNet --dataset_dir work/foods --val_value 0.2 --test_value 0.1
```

    Dataset Split Done.[0m
    [0mTrain samples: 3500[0m
    [0mEval samples: 1000[0m
    [0mTest samples: 500[0m
    [0mSplit files saved in work/foods[0m
    [0m[0m


```python
import paddlex as pdx
from paddlex.cls import transforms
# 数据增强：定义训练和验证时的transforms
train_transforms = transforms.Compose([
    # 图像预处理代码：随机水平翻转、随机垂直翻转、随机旋转、标准化
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotate(),
    transforms.Normalize()
])

eval_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotate(),
    transforms.Normalize()
])

# 数据集读取
path ='work/foods'
train_dataset = pdx.datasets.ImageNet(
                    data_dir= path,
                    file_list= path + '/train_list.txt',
                    label_list= path + '/labels.txt',
                    transforms=train_transforms)
eval_dataset = pdx.datasets.ImageNet(
                    data_dir= path,
                    file_list= path + '/val_list.txt',
                    label_list= path + '/labels.txt',
                    transforms=eval_transforms)
```

    2021-08-15 17:27:26 [INFO]	Starting to read file list from dataset...
    2021-08-15 17:27:26 [INFO]	3500 samples in file work/foods/train_list.txt
    2021-08-15 17:27:26 [INFO]	Starting to read file list from dataset...
    2021-08-15 17:27:26 [INFO]	1000 samples in file work/foods/val_list.txt


## 三、模型选择
>


```python
# 设置使用0号GPU卡（如无GPU，执行此代码后仍然会使用CPU训练模型）
import matplotlib
matplotlib.use('Agg') 
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
```

    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/matplotlib/__init__.py:107: DeprecationWarning: Using or importing the ABCs from 'collections' instead of from 'collections.abc' is deprecated, and in 3.8 it will stop working
      from collections import MutableMapping
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/matplotlib/rcsetup.py:20: DeprecationWarning: Using or importing the ABCs from 'collections' instead of from 'collections.abc' is deprecated, and in 3.8 it will stop working
      from collections import Iterable, Mapping
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/matplotlib/colors.py:53: DeprecationWarning: Using or importing the ABCs from 'collections' instead of from 'collections.abc' is deprecated, and in 3.8 it will stop working
      from collections import Sized





```python
num_classes = len(train_dataset.labels)
model = pdx.cls.MobileNetV3_large_ssld(num_classes=num_classes)
model.train(num_epochs=1,
            train_dataset=train_dataset,
            train_batch_size=16,
            eval_dataset=eval_dataset,
            lr_decay_epochs=[6, 8],
            save_interval_epochs=1,
            learning_rate=0.00625,
            save_dir='output/mobilenetv3_large_ssld',
            use_vdl=True)
```

## 四、效果展示




```python
import paddlex as pdx
# 模型载入
model = pdx.load_model('output/mobilenetv3_large_ssld/best_model')
# 使用数据集文件夹下test.txt中的一张图片进行预测，打印预测结果）
image_name = 'work/foods/beef_carpaccio/3400.jpg'
result = model.predict(image_name)
print("Predict Result:", result)
```

    2021-08-15 17:30:07 [INFO]	Model[MobileNetV3_large_ssld] loaded.
    Predict Result: [{'category_id': 4, 'category': 'beef_tartare', 'score': 0.99981517}]


## 五、模型导出


```python
!paddlex --export_inference --model_dir=output/mobilenetv3_large_ssld/best_model --save_dir=inference_model
```

## 六、总结说明
#### 1.数据集获取
* 一开始选择的项目因为找不到合适的数据集而不得不放弃，然后选择了这个美食识别的项目，在飞浆里面可以得到很好的数据集
#### 2.模型训练
* 在模型训练中`model.train()`每次运行只能使用一次，需要多次使用时得重启环境；设置`train_batch_size`默认是32，可以根据模型大小设置，但是设太大容易爆内存
* 使用PaddleX很多时候会内存溢出，可以尝试重启环境释放内存以解决问题
#### 3.个人感受
* 关于使用PaddleX的感受，我觉得PaddleX很适合小白快速开发一个属于自己的项目，对于小白来说是个非常友好的开发工具
* 完成项目的过程中也有很多需要请教学长的地方，在这里也非常感谢热心指导我的各位学长

## 个人简介

> 作者：杨宗键


> 东北大学秦皇岛分校2020级自动化本科生


> 感兴趣方向：cv、深度学习


我在[AI Studio](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/568205)上获得青铜等级，点亮1个徽章，来互关呀！！
