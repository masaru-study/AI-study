import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image, ImageDraw, ImageFont
import os

# --- 追加：学習前（汎用モデル）用のCOCOクラス名リスト ---
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

def run_demo(image_path, is_trained=False, model_path="cylinder_model.pth"):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"\n--- 実行モード: {'【学習後】特化モデル' if is_trained else '【学習前】汎用モデル'} ---")
    print(f"Using device: {device}")

    # 1. モデルの準備
    if is_trained:
        # 学習後モード：構造を2クラス（背景+シリンダー）に変更して重みをロード
        model = fasterrcnn_resnet50_fpn(weights=None)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
        
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
            print(f"モデル '{model_path}' をロードしました。")
        else:
            print(f"Error: {model_path} が見つかりません。先に学習を行ってください。")
            return
        color = "cyan" # 青緑色
        class_names = ['__background__', 'Cylinder'] # 自作クラス名
    else:
        # 学習前モード：標準の学習済み重み（COCO 80種）をロード
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        model = fasterrcnn_resnet50_fpn(weights=weights)
        color = "red" # 赤色
        class_names = COCO_INSTANCE_CATEGORY_NAMES # COCOクラス名リストを使用

    model.to(device)
    model.eval()

    # 2. 画像の準備
    img = Image.open(image_path).convert("RGB")
    # 前処理は共通の基準を使用
    transform = FasterRCNN_ResNet50_FPN_Weights.DEFAULT.transforms()
    input_tensor = transform(img).unsqueeze(0).to(device)

    # 3. 推論実行
    with torch.no_grad():
        prediction = model(input_tensor)

    # 4. 結果の描画
    draw = ImageDraw.Draw(img)
    # フォントの設定（OS依存ですが、デモ用ならデフォルトでもOK。必要ならパスを指定）
    try:
        font = ImageFont.truetype("arial.ttf", 40)
    except IOError:
        font = ImageFont.load_default()

    found = False
    scores = prediction[0]['scores']
    labels = prediction[0]['labels'] # 判定されたクラスのID

    print(f"検出されたオブジェクト候補数: {len(scores)}")
    
    # 閾値を設定
    threshold = 0.5 if not is_trained else 0.15

    for i, score in enumerate(scores):
        if score > threshold:
            box = prediction[0]['boxes'][i].cpu().numpy()
            label_id = labels[i].item()
            
            class_name = class_names[label_id] if label_id < len(class_names) else "Unknown"

            # 1. 枠を描画
            draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline=color, width=8)
            
            # 2. テキストの準備
            text = f"{class_name}: {score:.2f}"
            
            if hasattr(font, 'getbbox'):
                bbox = font.getbbox(text)
                text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            else:
                text_w, text_h = draw.textsize(text, font)

            # --- 修正ポイント：表示位置を「枠の右下（内側）」に計算 ---
            # box[2]が右端、box[3]が下端の座標です
            text_x = box[2] - text_w - 10
            text_y = box[3] - text_h - 10
            
            # 枠からはみ出さないよう、最小値（左上）も考慮
            text_x = max(box[0], text_x)
            text_y = max(box[1], text_y)

            # テキスト背景を描画
            draw.rectangle([(text_x - 5, text_y - 5), (text_x + text_w + 5, text_y + text_h + 5)], fill=color)
            
            # テキストを描画
            text_color = "white" if color in ["red", "cyan"] else "black"
            draw.text((text_x, text_y), text, fill=text_color, font=font)
            
            print(f"【成功】物体を検知！ クラス: {class_name}, スコア: {score:.4f}")
            found = True

    if not found:
        print(f"閾値 {threshold} 以上の物体は見つかりませんでした。")

    img.show()
    save_name = "after_train_labeled.jpg" if is_trained else "before_train_labeled.jpg"
    img.save(save_name)

if __name__ == "__main__":
    # 実演で使用する画像ファイル
    img_file = "IMG_4227.jpeg" 

    # --- デモの進行に合わせてここを切り替えて実行 ---
    
    # ステップ1: 学習前（汎用モデル）で見つけられない（キーボードなどが検知される）ことを示す
    #run_demo(img_file, is_trained=False)
    
    # ステップ2: 学習後（特化モデル）でシリンダーを見つける様子を示す
    # ※先に train_cylinder.py で cylinder_model.pth を作成しておく必要があります
    run_demo(img_file, is_trained=True)