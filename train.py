from ultralytics import YOLO

# 加载官方预训练的 Pose 模型
model = YOLO('yolov8n-pose.pt')

# 开始训练
results = model.train(
    data='ore_pose.yaml', 
    epochs=100, 
    imgsz=640,
    batch=16,
    device='cpu'  # 如果您有显卡，0 代表使用第一张 GPU；如果没有显卡，请删掉这一行
)
