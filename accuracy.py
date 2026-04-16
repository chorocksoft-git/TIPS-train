from ultralytics import YOLO

model = YOLO("runs/detect/result_v10/weights/best.pt")

metrics = model.val(
    data="datasets/v10/tacticalist.yaml",
    split="val",      # <- test split 사용
    imgsz=640,
#    conf=0.3
)

print("mAP50:", metrics.box.map50)
print("mAP50-95:", metrics.box.map)
print(metrics.box.maps)
