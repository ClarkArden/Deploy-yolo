Deploy-yolo-multitask
===
这是一个初步的detection + segmentation的实现，我们可以在这里得到mask和bbox信息。
具体如何在C++上实现，我们需要先过一下pytorch上官方的实现方法。
但是，通过测速我们会发现其实这个实现的后处理会比较慢，后处理所占用的时间和GPU上forward的时间差不多了
主要是因为我们在CPU端做了

 - decode   : detection part
 - nms      : detection part
 - matmul   : segmentation part
 - sigmoid  : segmentation part
 - resize   : segmentation part

所以，其实我们可以考虑一下这些部分是否可以仍然在GPU上处理呢？
让trt在一次enqueue以后，得到的output binding只有一个，这里包含经过筛选过的bbox和bbox的confidence和mask
这样我们在cpu端只需要做绘图就好了，可以大大的缩减推理时间

因此，可以设计一个Plugin来实现加速，因为其实后处理的有多部分其实比较适合CUDA加速
大体上的思路是这样，这里以yolo经常做测试用的car.jpg为例，筛选过后的bbox有5个

- detection head:
    - decode  : 用conf thread对bbox过滤                           [1, 8400 , 116] -> [1, 96, 116]
    - decode  : 用affine matrix对将cx, cy, w, h转为x0, y0, x1, y1 [1, 96, 116]    -> [1, 96, 116]
    - decode  : nms处理                                           [1, 96, 116]    -> [1, 5, 116]

- 分离:
    - 将116个feature给分成84 + 32,用来分别处理:                   [1, 5, 116]     -> bbox: [1, 5, 84]
                                                                                -> mc:   [1, 5, 32]

- matmul与sigmoid得到mask feature
    - mc与proto做matmul:                                          [1, 5, 32] * [1, 32, 25600] -> [1, 5, 25600]
    - sigmoid处理:                                                [1, 5, 25600] -> [1, 5, 25600]

- concate:
    - 将bbox的feature和mask feature合并                           [1, 5, 25600] + [1, 5, 84] -> [1, 5, 25684]

- 输出
    - [1, 5, 25684]
 
这里需要注意的一点，就是从enqueue出来的feature是160*160规格的，不建议enqueue时把resize一起做了，
主要是因为如果在这个plugin中把resize也做了的话，输出的tensor的大小会非常大，之后在D2H的数据拷贝时容易成为瓶颈


目前int8量化会导致推理引擎生成失败。需要改造一下pytorch的框架。

###  trt-engine-explorer 工具使用
1. 安装trt-engine-explorer 
```
conda create -n env_trex python=3.9
conda activate env_trex
git clone https://github.com/NVIDIA/TensorRT.git
cd TensorRT/tools/experimental/trt-engine-explorer
python3 -m pip install -e.
pip install werkzeug==3.0.0 Flask==3.0.0
sudo apt-get --yes install graphviz
pip install tensorrt
```

2. 使用 trt-engine-explorer 提供的工具生成分析文件
```
python src/python/process_engine.py models/engine/yolo11s-int8.engine profile --profile-engine
```
3. 使用 trt-engine-explorer 提供的工具生成SVG图像
```
python src/python/draw_engine.py models/engine/yolo11s-int8.engine.graph.json
```

## 参考
1. [tensorrt_starter](https://github.com/kalfazed/tensorrt_starter)
2. [CSDN博客](https://blog.csdn.net/qq_40672115/article/details/141789558)
3. [trt-engine-explorer](https://github.com/NVIDIA/TensorRT/tree/release/10.7/tools/experimental/trt-engine-explorer/utils)