
import sys
from pathlib import Path
from typing import List, Dict
# Ensure parent directory is in sys.path for import
sys.path.append(str(Path(__file__).parent.parent))
from yolo11_detect_train import class_id_to_name

class LabelStatsCollector:
    def __init__(self, data_dirs: List[str], views: List[str], class_id_to_name: Dict[int, str]):
        self.data_dirs = data_dirs
        self.views = views
        self.class_id_to_name = class_id_to_name
        self.label_files: List[Path] = []
        self.class_counts = {cid: 0 for cid in class_id_to_name}

    def collect_label_files(self):
        for data_dir in self.data_dirs:
            date_segment = Path(data_dir).name
            for view in self.views:
                label_dir = Path(data_dir) / 'label' / f'{date_segment}-txt' / view
                if not label_dir.exists():
                    continue
                for label_file in label_dir.glob('*.txt'):
                    if label_file.is_file():
                        self.label_files.append(label_file)

    def count_bboxes_per_class(self):
        for label_file in self.label_files:
            with open(label_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) < 1:
                        continue
                    try:
                        class_id = int(parts[0])
                        if class_id in self.class_counts:
                            self.class_counts[class_id] += 1
                    except ValueError:
                        continue

    def print_stats(self):
        print(f"Total label files: {len(self.label_files)}")
        print("Bounding box count per class:")
        for cid, count in self.class_counts.items():
            print(f"{cid}: {self.class_id_to_name[cid]}: {count}")

def main():
    data_dirs = [
        '/media/levin/DATA/checkpoints/controlnet/data/EOL2/20.02.2025',
        # Add more directories as needed
    ]
    views = ['top', 'bottom', 'left', 'right', 'front', 'back']
    collector = LabelStatsCollector(data_dirs, views, class_id_to_name)
    collector.collect_label_files()
    collector.count_bboxes_per_class()
    collector.print_stats()

if __name__ == '__main__':
    main()
