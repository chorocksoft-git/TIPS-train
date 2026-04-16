import glob
from pathlib import Path
import cv2
import yaml
from ultralytics import YOLO

def load_dataset_yaml(yaml_path: str):
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    base = data.get("path", "")
    val_rel = data["val"]  # required

    names = data.get("names", None)
    if isinstance(names, dict):
        names_list = [names[i] for i in sorted(names.keys())]
    else:
        names_list = names

    val_img_dir = Path(base) / val_rel if base else Path(val_rel)
    return val_img_dir, names_list


def val_img_to_label_path(img_path: Path) -> Path:
    parts = list(img_path.parts)
    # 데이터셋 구조에 따라 images 폴더가 포함되어 있다고 가정
    if "images" in parts:
        idx = parts.index("images")
        parts[idx] = "labels"
    return Path(*parts).with_suffix(".txt")


def read_yolo_label_file(label_path: Path):
    if not label_path.exists():
        return []

    items = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            cls = int(float(parts[0]))
            xc, yc, w, h = map(float, parts[1:5])
            items.append((cls, xc, yc, w, h))
    return items


def yolo_norm_to_xyxy(xc, yc, w, h, img_w, img_h):
    x1 = (xc - w / 2) * img_w
    y1 = (yc - h / 2) * img_h
    x2 = (xc + w / 2) * img_w
    y2 = (yc + h / 2) * img_h
    return int(x1), int(y1), int(x2), int(y2)


def draw_boxes(img, boxes, names, color, prefix, show_conf=False):
    for b in boxes:
        if show_conf:
            cls, conf, x1, y1, x2, y2 = b
            cls_name = names[cls] if names and cls < len(names) else str(cls)
            label = f"{prefix}:{cls_name} {conf:.2f}"
        else:
            cls, x1, y1, x2, y2 = b
            cls_name = names[cls] if names and cls < len(names) else str(cls)
            label = f"{prefix}:{cls_name}"

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            img,
            label,
            (x1, max(0, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )


def main(dataset_ver="v8", model_ver="result_v8"):
    # ====== 사용자 설정 ======
    # 경로 수정: project/name 구조에 맞춰 수정 필요시 변경하세요.
    # 예: runs/detect/result_v8/weights/best.pt
    model_path = f"runs/detect/{model_ver}/weights/best.pt"
    dataset_yaml = f"datasets/{dataset_ver}/tacticalist.yaml"

    imgsz = 640
    conf = 0.25
    iou = 0.5

    device = 0          # GPU:0
    pred_batch = 32     # GPU 메모리 여유에 따라 조절
    half = True         # FP16 사용

    out_dir = Path(f"runs/detect/{model_ver}/val_gt_vs_pred")
    out_dir.mkdir(parents=True, exist_ok=True)
    # ========================

    # 1) YAML에서 val 이미지 경로 로드
    val_img_dir, names = load_dataset_yaml(dataset_yaml)
    if not val_img_dir.exists():
        raise FileNotFoundError(f"val image dir not found: {val_img_dir}")
    
    print(f"[INFO] Validation Images Directory: {val_img_dir}")

    # 2) 모델 로드
    if not Path(model_path).exists():
        print(f"[ERROR] Model not found: {model_path}")
        return
        
    model = YOLO(model_path)

    # 3) 배치 예측 실행 (source에 폴더 경로를 직접 전달하여 'Too many open files' 방지)
    # stream=True를 사용하여 결과를 제너레이터로 받습니다.
    results_iter = model.predict(
        source=str(val_img_dir), # 폴더 경로(str) 전달
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device=device,
        batch=pred_batch,
        half=half,
        stream=True,
        verbose=False,
        save=False,
    )

    # 4) 결과 반복 처리
    count = 0
    for preds in results_iter:
        # preds.path에 현재 처리된 이미지의 원본 경로가 들어있습니다.
        img_path = Path(preds.path)
        
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARN] failed to read: {img_path}")
            continue

        h, w = img.shape[:2]

        # ---- GT boxes ----
        label_path = val_img_to_label_path(img_path)
        gt_items = read_yolo_label_file(label_path)

        gt_boxes_xyxy = []
        for cls, xc, yc, bw, bh in gt_items:
            x1, y1, x2, y2 = yolo_norm_to_xyxy(xc, yc, bw, bh, w, h)
            gt_boxes_xyxy.append((cls, x1, y1, x2, y2))

        # ---- Pred boxes ----
        pred_boxes_xyxy = []
        if preds.boxes is not None and len(preds.boxes) > 0:
            # CPU로 이동 및 numpy 변환
            xyxy = preds.boxes.xyxy.detach().cpu().numpy()
            cls_ids = preds.boxes.cls.detach().cpu().numpy().astype(int)
            confs = preds.boxes.conf.detach().cpu().numpy()

            for (x1, y1, x2, y2), c, cf in zip(xyxy, cls_ids, confs):
                pred_boxes_xyxy.append((int(c), float(cf), int(x1), int(y1), int(x2), int(y2)))

        # 시각화 (GT: 초록색, Pred: 빨간색)
        vis = img.copy()
        draw_boxes(vis, gt_boxes_xyxy, names, color=(0, 255, 0), prefix="GT", show_conf=False)
        #draw_boxes(vis, pred_boxes_xyxy, names, color=(0, 0, 255), prefix="PRED", show_conf=True)

        save_path = out_dir / img_path.name
        cv2.imwrite(str(save_path), vis)
        count += 1
        
        # 진행 상황 표시 (선택 사항)
        if count % 100 == 0:
            print(f"[INFO] Processed {count} images...")

    print(f"[DONE] Processed {count} images. Saved to: {out_dir}")


if __name__ == "__main__":
    # main 함수 실행 시 설정값 확인하세요.
    main(dataset_ver="v10", model_ver="result_v10")
