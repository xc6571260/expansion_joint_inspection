import numpy as np
import cv2
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from geopy.distance import geodesic
from sklearn.cluster import DBSCAN

def detect_and_merge_boxes(model, orig_img, resize_dim=1024, eps=50, min_samples=1):
    H, W = orig_img.shape[:2]
    resized_img = cv2.resize(orig_img, (resize_dim, resize_dim))

    results = model(resized_img, verbose=False)[0].boxes
    if results is None or len(results) == 0:
        return [], None, None

    boxes = results.xyxy.cpu().numpy()
    centers = np.array([[(x1 + x2) / 2, (y1 + y2) / 2] for x1, y1, x2, y2 in boxes])

    db = DBSCAN(eps=eps, min_samples=min_samples).fit(centers)
    labels = db.labels_

    merged_boxes = []
    for label in np.unique(labels):
        group = boxes[labels == label]
        x1s, y1s, x2s, y2s = group[:, 0], group[:, 1], group[:, 2], group[:, 3]
        merged_box = [int(x1s.min()), int(y1s.min()), int(x2s.max()), int(y2s.max())]
        merged_boxes.append(merged_box)

    scale_x = W / resize_dim
    scale_y = H / resize_dim

    return merged_boxes, scale_x, scale_y

def get_gps_from_exif(img_path):
    img = Image.open(img_path)
    exif_data = img._getexif()
    if exif_data is None:
        return None

    gps_info = {}
    for tag, value in exif_data.items():
        decoded = TAGS.get(tag, tag)
        if decoded == "GPSInfo":
            for t in value:
                sub_decoded = GPSTAGS.get(t, t)
                gps_info[sub_decoded] = value[t]

    if "GPSLatitude" in gps_info and "GPSLongitude" in gps_info:
        lat = gps_info["GPSLatitude"]
        lon = gps_info["GPSLongitude"]
        lat_ref = gps_info.get("GPSLatitudeRef", "N")
        lon_ref = gps_info.get("GPSLongitudeRef", "E")

        def dms_to_dd(dms):
            if isinstance(dms[0], tuple):
                deg = dms[0][0] / dms[0][1]
                minute = dms[1][0] / dms[1][1]
                sec = dms[2][0] / dms[2][1]
            else:
                deg, minute, sec = dms
            return deg + minute / 60 + sec / 3600

        latitude = dms_to_dd(lat)
        longitude = dms_to_dd(lon)

        if lat_ref != "N":
            latitude = -latitude
        if lon_ref != "E":
            longitude = -longitude

        return (latitude, longitude)
    else:
        return None

def find_nearest_image(poi_coord, image_infos):
    best_img = None
    best_dist = float('inf')
    for img_info in image_infos:
        if img_info["used"]:
            continue
        dist = geodesic(poi_coord, img_info["gps"]).meters
        if dist < best_dist:
            best_dist = dist
            best_img = img_info
    return best_img
