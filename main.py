import os
import glob
import yaml
from tqdm import tqdm
from ultralytics import YOLO

from utils.detection import detect_and_merge_boxes
from utils.segmentation import process_patches
from utils.io_utils import load_image, save_image

# === 讀設定 ===
base_dir = os.path.dirname(__file__)
config_path = os.path.join(base_dir, "config.yaml")

with open(config_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

origin_folder = os.path.abspath(os.path.join(base_dir, cfg["origin_folder"]))
output_folder = os.path.abspath(os.path.join(base_dir, cfg["output_folder"]))
detect_model_path = os.path.abspath(os.path.join(base_dir, cfg["detect_model_path"]))
seg_model_path = os.path.abspath(os.path.join(base_dir, cfg["seg_model_path"]))


resize_dim = cfg.get("resize_dim", 1024)
eps = cfg.get("eps", 50)
min_samples = cfg.get("min_samples", 1)

os.makedirs(output_folder, exist_ok=True)

# === 載入模型 ===
detect_model = YOLO(detect_model_path)
seg_model = YOLO(seg_model_path)


# === 處理所有 input 資料夾內的圖片 ===
image_paths = []
for ext in ["*.jpg", "*.JPG", "*.png"]:
    image_paths.extend(glob.glob(os.path.join(origin_folder, ext)))
image_paths.sort()

for img_path in tqdm(image_paths, desc="處理所有圖片"):
    orig_img = load_image(img_path)
    if orig_img is None:
        print(f"❌ 無法讀取圖片: {img_path}")
        continue

    H, W = orig_img.shape[:2]
    merged_boxes, scale_x, scale_y = detect_and_merge_boxes(detect_model, orig_img, resize_dim, eps=eps, min_samples=min_samples)

    if not merged_boxes:
        print(f"⚠️ 無偵測結果: {img_path}")
        result_img = orig_img
    else:
        result_img = process_patches(orig_img, merged_boxes, scale_x, scale_y, seg_model)

    base_name = os.path.splitext(os.path.basename(img_path))[0]
    save_path = os.path.join(output_folder, f"{base_name}.jpg")
    save_image(result_img, save_path)

print("✅ 全部處理完成！")
