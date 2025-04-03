from ultralytics.utils.benchmarks import benchmark

benchmark(
    model="runs/train/weights/best.pt",
    data="data/processed/dataset.yaml", 
    imgsz=640, 
    half=False, 
    device=0
)