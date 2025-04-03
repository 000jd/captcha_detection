from src.utils.multi_captcha_detector import MultiCAPTCHADetector

detector = MultiCAPTCHADetector(
    model_path='runs/train/weights/best.pt',
    classes_file='data/raw/classes.txt',
    conf_threshold=0.25
)

# Detect CAPTCHA in an image
result = detector.detect_captcha('data/raw/images/0a1d3125-a6d8b03ff73ae89eb1f7665911d63842.jpeg')
print(f"Detected text: {result['text']}")