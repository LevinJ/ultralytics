import os
from pathlib import Path
from ultralytics import YOLO
import sys
sys.path.append(str(Path(__file__).parent))
from yolo11_detect_train import class_id_to_name, YoloDetectionDataPreparer

class YoloDetectionValidator:
    def __init__(self, model_path: str, data_yaml: str, imgsz: int = 640, batch: int = 128):
        self.model_path = model_path
        self.data_yaml = data_yaml
        self.imgsz = imgsz
        self.batch = batch

    def validate(self):
        yolo = YOLO(self.model_path)
        results = yolo.val(data=self.data_yaml, imgsz=self.imgsz, batch=self.batch, split='test')
        # print(results)
        return results

def main():
    # Path to the new data directory to validate on
    data_dir = '/media/levin/DATA/checkpoints/controlnet/data/EOL2/18.02.2025'
    views = ['top', 'bottom', 'left', 'right', 'front', 'back']
    temp_dir = Path(__file__).parent / 'temp/validate'
    model_path = '/media/levin/DATA/checkpoints/Factory/ultralytics/runs/detect/train2/weights/best.pt'

    preparer = YoloDetectionDataPreparer([data_dir], views, temp_dir, split_mode='test_only')
    yaml_path = preparer.prepare_if_needed(class_id_to_name)

    validator = YoloDetectionValidator(model_path, str(yaml_path), imgsz=640, batch=128)
    validator.validate()

if __name__ == '__main__':
    main()
