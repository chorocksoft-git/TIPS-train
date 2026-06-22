from ultralytics import YOLO

# model = YOLO("models/yolo11n.pt") # 기존 nano 모델
# 복잡한 각도를 학습하려면 모델 크기를 키우는 것도 고려 (s 또는 m)
model = YOLO("models/yolo11m.pt")

results = model.train(
    data="datasets/v14/tacticalist.yaml",

   # imgsz=[608, 1088],
   # rect=True,
    imgsz=640,
    epochs=200, # 어려운 데이터일수록 에포크를 좀 더 늘리는 게 좋습니다.
    batch=128,
    device=0,
    amp=True,
    cache='disk',
    workers=4,

    # --- [핵심 수정: 기하학적 증강 추가] ---
    perspective=0.001,  # 원근감 (트리거캠 시점 흉내) - 값을 조금씩 변경해보세요
    degrees=15.0,       # 회전 (총기 기울임 대응)
    shear=10.0,         # 기울임 (측면 시점 대응)
    translate=0.2,      # 평행 이동 (화면 구석에 있는 적 대응)
    scale=0.5,          # 크기 변화 (멀리 있는 적/가까이 있는 적)
    # -------------------------------------

    # 기존 증강 유지
    mosaic=1.0,
    mixup=0.1,
    copy_paste=0.1,
    auto_augment="randaugment",

    # [추가 증강 파라미터]
    bgr=0.1,        # 10% 확률로 색상 채널을 뒤집음 (파란 하늘 외에 역광 등 다양한 색감 대응)
    hsv_s=0.5,      # 채도 변환 폭 증가 (안개 낀 날씨 흉내)
    hsv_v=0.5,      # 명도 변환 폭 증가 (눈부신 환경 흉내)

    # 후반 안정화
    close_mosaic=30, # 에포크가 늘었으니 이것도 약간 늘려줍니다.
    patience=60,
    name="result_v14",
)