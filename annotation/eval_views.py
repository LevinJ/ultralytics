
import os
import json
from ultralytics import YOLO

class YoloViewClassifierEvaluator:
    def __init__(self, model_path, data_dirs, categories):
        self.model_path = model_path
        self.data_dirs = data_dirs
        self.categories = categories
        self.model = YOLO(self.model_path)
        self.image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.gif', '.webp'}

    def evaluate_directory(self, data_dir):
        category_map = {}
        print(f'Processing directory: {data_dir}')
        for cat in self.categories:
            print(f'  Processing category: {cat}')
            sub_dir = os.path.join(data_dir, cat)
            if not os.path.isdir(sub_dir):
                print(f'Skipping missing subdirectory: {sub_dir}')
                continue
            images = [f for f in os.listdir(sub_dir)
                      if os.path.isfile(os.path.join(sub_dir, f)) and os.path.splitext(f)[1].lower() in self.image_exts]
            if not images:
                print(f'No images found in {sub_dir}')
                continue
            images.sort()
            img_paths = [os.path.join(sub_dir, img) for img in images]
            # Run inference in batches
            batch_size = 256
            preds = []
            for i in range(0, len(img_paths), batch_size):
                batch_paths = img_paths[i:i+batch_size]
                batch_results = self.model(batch_paths, verbose=False)
                preds.extend([r.probs.top1 for r in batch_results])
            pred_label = self.model.names[preds[0]]
            category_map[cat] = pred_label
            for img_path, pred_idx in zip(img_paths[1:], preds[1:]):
                pred_label_img = self.model.names[pred_idx]
                if pred_label_img != pred_label:
                    print(f'Mismatch: {img_path} predicted as {pred_label_img}, expected {pred_label}')
        # Save category mapping
        mapping_path = os.path.join(data_dir, 'category_mapping.json')
        with open(mapping_path, 'w') as f:
            json.dump(category_map, f, indent=2)
        print(f'Category mapping saved to {mapping_path}\n')

    def run(self):
        for data_dir in self.data_dirs:
            self.evaluate_directory(data_dir)


if __name__ == '__main__':
    MODEL_PATH = '/media/levin/DATA/checkpoints/Factory/ultralytics/runs/classify/train13/weights/best.pt'
    DATA_DIRS = [
        '/media/levin/DATA/checkpoints/controlnet/data/EOL/18.02.2025',
        '/media/levin/DATA/checkpoints/controlnet/data/EOL/20.02.2025',
        '/media/levin/DATA/checkpoints/controlnet/data/EOL/21.02.2025',
    ]
    CATEGORIES = ['top', 'bottom', 'left', 'right', 'front', 'back']
    evaluator = YoloViewClassifierEvaluator(MODEL_PATH, DATA_DIRS, CATEGORIES)
    evaluator.run()
