import os
import random
from ultralytics import YOLO

class Yolo11ClassifierTrainer:
    def __init__(self,
                 data_dir,
                 config_path,
                 class_names,
                 temp_dir=None,
                 split_ratio=0.8,
                 seed=42,
                 epochs=10,
                 imgsz=224,
                 batch=512,
                 model_path='yolo11n-cls.pt'):
        self.data_dir = data_dir
        self.config_path = config_path
        self.class_names = class_names
        self.split_ratio = split_ratio
        self.seed = seed
        self.epochs = epochs
        self.imgsz = imgsz
        self.batch = batch
        self.model_path = model_path
        if temp_dir is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            temp_dir = os.path.join(script_dir, 'temp')
        self.temp_dir = temp_dir
        os.makedirs(self.temp_dir, exist_ok=True)

        self.split_root = os.path.join(self.temp_dir, 'view_cls_train')
        self.split_dirs = {
            'train': os.path.join(self.split_root, 'train'),
            'val': os.path.join(self.split_root, 'val'),
            'test': os.path.join(self.split_root, 'test')
        }


    def split_and_copy_images(self):
        import shutil
        random.seed(self.seed)
        image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.gif', '.webp'}
        for split in self.split_dirs.values():
            for cls in self.class_names:
                os.makedirs(os.path.join(split, cls), exist_ok=True)
        for cls in self.class_names:
            cls_dir = os.path.join(self.data_dir, cls)
            images = [f for f in os.listdir(cls_dir)
                      if os.path.isfile(os.path.join(cls_dir, f)) and os.path.splitext(f)[1].lower() in image_exts]
            images = [os.path.join(cls_dir, img) for img in images]
            random.shuffle(images)
            n = len(images)
            n_train = int(n * 0.8)
            n_val = int(n * 0.1)
            n_test = n - n_train - n_val
            train_imgs = images[:n_train]
            val_imgs = images[n_train:n_train+n_val]
            test_imgs = images[n_train+n_val:]
            for split_name, split_imgs in zip(['train', 'val', 'test'], [train_imgs, val_imgs, test_imgs]):
                split_cls_dir = os.path.join(self.split_dirs[split_name], cls)
                for img_path in split_imgs:
                    shutil.copy(img_path, os.path.join(split_cls_dir, os.path.basename(img_path)))


    def ensure_split_dirs(self):
        # Check if split folders exist and are non-empty for all classes
        ready = True
        for split in ['train', 'val', 'test']:
            for cls in self.class_names:
                split_cls_dir = os.path.join(self.split_dirs[split], cls)
                if not os.path.exists(split_cls_dir) or not os.listdir(split_cls_dir):
                    ready = False
        if not ready:
            print('Splitting and copying images into train/val/test folders...')
            self.split_and_copy_images()
            print('Image split and copy complete.')
        else:
            print('Split folders already exist and are populated. Skipping splitting step.')

    def train(self):
        self.ensure_split_dirs()
        model = YOLO(self.model_path)
        results = model.train(
            data=self.split_root,  # root folder containing train/val/test
            epochs=self.epochs,
            imgsz=self.imgsz,
            batch=self.batch,
          #  project=os.path.abspath(os.path.join(os.path.dirname(__file__), '../runs'))
        )
    print("Training complete.")
    # Evaluate the trained model on the test set
    print("Evaluating on test set...")
    test_results = model.val(data=self.split_dirs['test'])
    print("Evaluation complete.")
    print(test_results)


if __name__ == '__main__':
    data_dir = '/media/levin/DATA/checkpoints/controlnet/data/EOL/18.02.2025'
    config_path = '/media/levin/DATA/checkpoints/Factory/ultralytics/annotation/yolo11_cls_config.yaml'
    class_names = ['top', 'bottom', 'left', 'right', 'front', 'back']
    trainer = Yolo11ClassifierTrainer(data_dir=data_dir, config_path=config_path, class_names=class_names)
    trainer.train()
