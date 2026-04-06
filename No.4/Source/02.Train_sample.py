import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms as T
import torchvision.transforms.functional as F
from PIL import Image
import os
import time  # 1. 時間計測用のモジュールをインポート

def get_model(num_classes):
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def train_cylinder():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    #device = torch.device('cpu')
    print(f"Using device: {device}")

    model = get_model(num_classes=2)
    model.to(device)

    img_path = "cylinder.jpeg"
    if not os.path.exists(img_path):
        print(f"Error: {img_path} が見つかりません。")
        return

    img_raw = Image.open(img_path).convert("RGB")

    augmentation = T.Compose([
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.RandomRotation(degrees=5),
        T.ToTensor(),
    ])

    boxes = torch.tensor([[180.0, 380.0, 530.0, 830.0]], dtype=torch.float32).to(device)
    labels = torch.tensor([1], dtype=torch.int64).to(device)
    target = [{"boxes": boxes, "labels": labels}]

    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

    model.train()
    print("学習開始（水増しを適用中）...")
    
    # 2. 学習開始時刻を記録
    start_time = time.time() 

    epochs = 200
    for epoch in range(epochs):
        optimizer.zero_grad()
        img_tensor = augmentation(img_raw).to(device)
        erase = T.RandomErasing(p=0.5, scale=(0.02, 0.1))
        img_tensor = erase(img_tensor)

        loss_dict = model([img_tensor], target)
        losses = sum(loss for loss in loss_dict.values())
        losses.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}: Loss = {losses.item():.4f}")

    # 3. 学習終了時刻を記録し、差分を計算
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # 4. 結果の表示（秒、または分に換算）
    print("-" * 30)
    print(f"学習完了！ 経過時間: {elapsed_time:.2f} 秒")
    if elapsed_time > 60:
        print(f"（約 {elapsed_time / 60:.2f} 分）")
    print("-" * 30)

    save_path = "cylinder_model.pth"
    torch.save(model.state_dict(), save_path)
    print(f"'{save_path}' を保存しました。")

if __name__ == "__main__":
    train_cylinder()