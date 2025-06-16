import numpy as np
import cv2
from skimage.morphology import skeletonize

def process_patches(orig_img, merged_boxes, scale_x, scale_y, seg_model, img_name="image"):
    H, W = orig_img.shape[:2]
    total_count = 0
    normal_count = 0
    abnormal_count = 0

    for (x1, y1, x2, y2) in merged_boxes:
        x1_orig = int(x1 * scale_x)
        y1_orig = int(y1 * scale_y)
        x2_orig = int(x2 * scale_x)
        y2_orig = int(y2 * scale_y)

        x1_orig = max(0, min(x1_orig, W - 1))
        y1_orig = max(0, min(y1_orig, H - 1))
        x2_orig = max(0, min(x2_orig, W))
        y2_orig = max(0, min(y2_orig, H))

        patch = orig_img[y1_orig:y2_orig, x1_orig:x2_orig]
        patch_h, patch_w = patch.shape[:2]

        indiv_mask = np.zeros((H, W), dtype=np.uint8)

        # segmentation
        if patch_h > 512 or patch_w > 512:
            for y in range(0, patch_h, 512):
                for x in range(0, patch_w, 512):
                    sub_patch = patch[y:y+512, x:x+512]
                    h_, w_ = sub_patch.shape[:2]
                    canvas = np.zeros((512, 512, 3), dtype=np.uint8)
                    canvas[:h_, :w_] = sub_patch
                    seg_result = seg_model(canvas, verbose=False)[0]
                    if not seg_result.masks:
                        continue
                    mask = seg_result.masks.data[0].cpu().numpy()
                    mask = (mask * 255).astype(np.uint8)[:h_, :w_]
                    mask_area = mask > 0
                    oy1 = y1_orig + y
                    ox1 = x1_orig + x
                    oy2 = min(oy1 + h_, H)
                    ox2 = min(ox1 + w_, W)
                    indiv_mask[oy1:oy2, ox1:ox2][mask_area] = 255
        else:
            canvas = np.zeros((512, 512, 3), dtype=np.uint8)
            y_off = (512 - patch_h) // 2
            x_off = (512 - patch_w) // 2
            if y_off < 0 or x_off < 0:
                continue
            canvas[y_off:y_off+patch_h, x_off:x_off+patch_w] = patch
            seg_result = seg_model(canvas, verbose=False)[0]
            if not seg_result.masks:
                continue
            mask = seg_result.masks.data[0].cpu().numpy()
            mask = (mask * 255).astype(np.uint8)
            seg_crop = mask[y_off:y_off+patch_h, x_off:x_off+patch_w]
            mask_area = seg_crop > 0
            indiv_mask[y1_orig:y2_orig, x1_orig:x2_orig][mask_area] = 255

        # ===== 統計寬度資訊 =====
        if np.any(indiv_mask):
            widths = []
            for col in range(x1_orig, x2_orig):
                column_mask = indiv_mask[y1_orig:y2_orig, col]
                ys = np.where(column_mask > 0)[0]
                if ys.size > 0:
                    width = ys.max() - ys.min()
                    widths.append(width)

            if widths:
                widths = np.array(widths)
                mean_width = np.mean(widths)
                max_width = np.max(widths)
                avg_width_cm = mean_width * 0.67
                max_width_cm = max_width * 0.67

                # 判斷是否 overflow
                if max_width_cm > 40:
                    label = "Overflow"
                    rect_color = (0, 0, 0)  # 黑色
                else:
                    std_width = np.std(widths)
                    upper_bound = mean_width + std_width
                    filtered_widths = widths[widths <= upper_bound]
                    if len(filtered_widths) > 0:
                        avg_width_cm = np.mean(filtered_widths) * 0.67
                        max_width_cm = np.max(filtered_widths) * 0.67

                    # 統計結果
                    total_count += 1
                    if avg_width_cm > 10:
                        abnormal_count += 1
                        rect_color = (0, 0, 255)  # 紅色
                    else:
                        normal_count += 1
                        rect_color = (0, 100, 0)  # 綠色
    
                    label = f"avg {avg_width_cm:.1f}cm"
                    
                    orig_img[indiv_mask == 255] = rect_color

                # ===== 畫框與標籤 =====
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.5
                thickness = 5
                (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
                cv2.rectangle(orig_img, (x1_orig, y1_orig), (x2_orig, y2_orig), rect_color, thickness)
                cv2.rectangle(orig_img,
                              (x1_orig, y1_orig - text_height - baseline),
                              (x1_orig + text_width, y1_orig),
                              rect_color, thickness=cv2.FILLED)
                cv2.putText(orig_img, label,
                            (x1_orig, y1_orig - baseline),
                            font, font_scale, (255, 255, 255), thickness, lineType=cv2.LINE_AA)
        # 若無裂縫，則不計數

    print(f"[INFO] image_name: {img_name}, total: {total_count}, normal: {normal_count}, abnormal: {abnormal_count}")
    return orig_img
