# 矿石数据集全自动生成指南 (YOLOv8-Pose 格式)

本项目包含一个使用 Blender Python API (bpy) 编写的自动化脚本 `generate_ore_data.py`。该脚本利用**域随机化 (Domain Randomization)** 技术，批量生成用于训练 YOLOv8-Pose 关键点检测模型的合成图像和自动标注标签。

## 📁 目录结构

确保您的工作区具备以下结构：

```text
/home/shark/model_train/
├── generate_ore_data.py    # 数据生成核心脚本
├── backgrounds/            # [需手动操作] 请在此处放置用于背景随机化的真实场景图片 (.jpg/.png)
├── dataset/                # 脚本会自动生成的输出目录
│   ├── images/             # 生成的合成图像 (10000张)
│   └── labels/             # 自动计算的 YOLOv8-Pose 标签文件
└── README.md               # 本使用说明
```

## 🛠️ 环境准备

### 1. 安装 Blender
如果您的系统中尚未安装 Blender，请执行以下命令进行安装（Ubuntu/Debian）：
```bash
sudo apt update
sudo apt install -y blender
```
*（注意：您也可以从 Blender 官网下载相应的免安装压缩包并解压使用）*

### 2. 准备背景图像
为了打破纯色背景带来的数据偏置 (Optical Bias)，您**必须**在 `backgrounds/` 文件夹内放入真实的背景图片（例如车间、赛场、桌面、杂乱环境的照片）。建议准备 20-50 张以获得更好的随机效果。

### 3. 配置 Blender 场景模型 (.blend)
在运行脚本之前，请准备一个包含以下物件的 `.blend` 场景工程文件（如 `ore_scene.blend`）：
- **`Ore_Model`**: 您的矿石 3D 网格模型，原点(Origin)需设置在矿石几何中心。
- **`Empty_P0` 到 `Empty_P7`**: 8个空物体(Empty)，分别绑定/放置在矿石的 8 个关键角点上，且必须作为 `Ore_Model` 的子对象 (便于跟随移动)。
- **`Camera`**: 默认渲染相机。
- **`Light_Main`**: 场景的主光源（点光、聚光灯或太阳光均可）。

## 🚀 启动生成任务

一切准备就绪后，在终端中运行以下命令（使用 Blender 的后台无 UI 模式 `-b` 执行 Python 脚本 `-P`）：

```bash
# 请将 ore_scene.blend 替换为您实际保存的工程文件路径
blender -b ore_scene.blend -P generate_ore_data.py
```

### 过程监控
脚本运行后，终端将会打印生成进度日志（如 `Generating image 0/10000`）。
所有生成好的图片将保存在 `dataset/images/`，对应的 `yolo` 格式标注 `.txt` 将保存在 `dataset/labels/`。

## 📝 标注格式说明 (YOLO-Pose)
输出的文本使用标准的 YOLO-Pose 格式：
`<class_id> <x_center> <y_center> <width> <height> <p0_x> <p0_y> <p0_vis> ... <p7_x> <p7_y> <p7_vis>`
* 坐标均已进行 `[0, 1]` 归一化。
* 关键点可见度标志 `vis` 被统一设定为 `2` (代表标注且可见)。
* 默认 `class_id = 0`。