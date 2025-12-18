import os
import shutil
import random
from pathlib import Path
from typing import List, Dict

import yaml

# YOLOv8/YOLO11 training imports (assume ultralytics is installed)
from ultralytics import YOLO

class_id_to_name = {
        0: "Actuator_ok",
        1: "Actuator_breakage",
        2: "Actuator_pin_ok",
        3: "Actuator_pin_breakage",
        4: "Piston_ok",
        5: "Piston_oil",
        6: "Piston_breakage",
        7: "Screw_ok",
        8: "Screw_untightened",
        9: "Guiderod_ok",
        10: "Guiderod_oil",
        11: "Guiderod_malposed",
        12: "surface_scratch",
        13: "Spring_ok",
        14: "Spring_variant",
        15: "Marker_ok",
        16: "Marker_breakage",
        17: "Line_ok",
        18: "Line_unaligned",
        19: "Exhaust_screw_ok",
        20: "Exhaust_screw_abnormal",
        21: "Support_surface_ok",
        22: "Support_surface_scratch"
    }

class YoloDetectionDataPreparer:
    def prepare_if_needed(self, class_id_to_name: Dict[int, str]):
        """
        Checks if data is already prepared in temp_dir. If not, performs data preparation and returns yaml_path.
        Returns:
            yaml_path (Path): Path to the generated or existing data.yaml
        """
        if self.temp_dir.exists() and any(self.temp_dir.iterdir()):
            return self.temp_dir / 'data.yaml'
        self.collect_image_label_pairs()
        self.split_data()
        self.organize_for_yolo()
        return self.create_yaml(class_id_to_name)
    def __init__(self, data_dirs: List[str], views: List[str], temp_dir: str, split_ratio=(0.8, 0.1, 0.1), split_mode: str = 'default'):
        self.data_dirs = data_dirs
        self.views = views
        self.temp_dir = Path(temp_dir)
        self.split_ratio = split_ratio
        self.split_mode = split_mode  # 'default' or 'test_only'
        self.image_label_pairs = []
        self.splits = {'train': [], 'val': [], 'test': []}

    def collect_image_label_pairs(self):
        image_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
        for data_dir in self.data_dirs:
            date_segment = Path(data_dir).name
            for view in self.views:
                img_dir = Path(data_dir) / view
                label_dir = Path(data_dir) / 'label' / f'{date_segment}-txt' / view
                assert img_dir.exists() and label_dir.exists(), f"Missing directory: {img_dir if not img_dir.exists() else label_dir}"
                # Use rglob to get all images with any of the extensions
                for img_file in img_dir.iterdir():
                    if img_file.is_file() and img_file.suffix.lower() in image_exts:
                        label_file = label_dir / (img_file.stem + '.txt')
                        assert label_file.exists(), f"Missing label file: {label_file}"
                        self.image_label_pairs.append((img_file, label_file))

    def split_data(self):
        if self.split_mode == 'test_only':
            self.splits['train'] = []
            self.splits['val'] = []
            self.splits['test'] = self.image_label_pairs.copy()
        else:
            random.shuffle(self.image_label_pairs)
            n = len(self.image_label_pairs)
            n_train = int(n * self.split_ratio[0])
            n_val = int(n * self.split_ratio[1])
            self.splits['train'] = self.image_label_pairs[:n_train]
            self.splits['val'] = self.image_label_pairs[n_train:n_train + n_val]
            self.splits['test'] = self.image_label_pairs[n_train + n_val:]
    

    def organize_for_yolo(self):
        for split in ['train', 'val', 'test']:
            img_out = self.temp_dir / 'images' / split
            lbl_out = self.temp_dir / 'labels' / split
            img_out.mkdir(parents=True, exist_ok=True)
            lbl_out.mkdir(parents=True, exist_ok=True)
            for img_path, lbl_path in self.splits[split]:
                shutil.copy(img_path, img_out / img_path.name)
                shutil.copy(lbl_path, lbl_out / lbl_path.name)

    def create_yaml(self, class_id_to_name: Dict[int, str]):
        yaml_path = self.temp_dir / 'data.yaml'
        data = {
            'train': str((self.temp_dir / 'images' / 'train').resolve()),
            'val': str((self.temp_dir / 'images' / 'val').resolve()),
            'test': str((self.temp_dir / 'images' / 'test').resolve()),
            'nc': len(class_id_to_name),
            'names': [class_id_to_name[i] for i in range(len(class_id_to_name))]
        }
        with open(yaml_path, 'w') as f:
            yaml.dump(data, f)
        return yaml_path

class YoloDetectionTrainer:
    def __init__(self, data_yaml: str, model: str = 'yolov8n.pt', epochs: int = 100, imgsz: int = 640, batch: int = 256):
        self.data_yaml = data_yaml
        self.model = model
        self.epochs = epochs
        self.imgsz = imgsz
        self.batch = batch
        self.trained_model_path = None

    def train(self):
        yolo = YOLO(self.model)
        results = yolo.train(data=self.data_yaml, epochs=self.epochs, imgsz=self.imgsz, batch=self.batch)
        print("Evaluating on test set...")
        results = yolo.val(split='test')
        print(results)
        return 

    

def main():
    data_dirs = [
        '/media/levin/DATA/checkpoints/controlnet/data/EOL2/21.02.2025',
        # Add more directories as needed
    ]
    views = ['top', 'bottom', 'left', 'right', 'front', 'back']
    temp_dir = Path(__file__).parent / 'temp/detect'
    
    preparer = YoloDetectionDataPreparer(data_dirs, views, temp_dir)
    yaml_path = preparer.prepare_if_needed(class_id_to_name)

    trainer = YoloDetectionTrainer(str(yaml_path), model='yolov8n.pt', epochs=100, imgsz=640, batch=128)
    trainer.train()
    return

if __name__ == '__main__':
    main()
