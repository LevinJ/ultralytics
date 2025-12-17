import os
import json
import shutil
from pathlib import Path

class CategoryMappingCopier:
    def __init__(self, data_dirs):
        self.data_dirs = data_dirs

    def load_mapping(self, data_dir):
        mapping_path = os.path.join(data_dir, 'category_mapping.json')
        with open(mapping_path, 'r') as f:
            return json.load(f)

    def replace_segments(self, path, mapping):
        parts = Path(path).parts
        # Replace 'EOL' with 'EOL2' and mapping keys with values, except for the last segment (file name)
        if len(parts) <= 1:
            return path
        new_parts = []
        for i, part in enumerate(parts):
            if i == len(parts) - 1:
                # Last segment (file name), leave unchanged
                new_parts.append(part)
            elif part == 'EOL':
                new_parts.append('EOL2')
            elif part in mapping:
                new_parts.append(mapping[part])
            else:
                new_parts.append(part)
        return os.path.join(*new_parts)

    def process_directory(self, data_dir):
        mapping = self.load_mapping(data_dir)
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                src_path = os.path.join(root, file)
                # Only process files (skip mapping file itself)
                if file == 'category_mapping.json':
                    continue
                # Compute destination path
                rel_path = os.path.relpath(src_path, data_dir)
                orig_full_path = os.path.join(data_dir, rel_path)
                dest_full_path = self.replace_segments(orig_full_path, mapping)
                # Ensure destination directory exists
                dest_dir = os.path.dirname(dest_full_path)
                os.makedirs(dest_dir, exist_ok=True)
                # Copy file
                shutil.copy2(src_path, dest_full_path)
                print(f'Copied {src_path} -> {dest_full_path}')

    def run(self):
        for data_dir in self.data_dirs:
            print(f'Processing {data_dir}')
            self.process_directory(data_dir)

if __name__ == '__main__':
    DATA_DIRS = [
        '/media/levin/DATA/checkpoints/controlnet/data/EOL/18.02.2025',
    ]
    copier = CategoryMappingCopier(DATA_DIRS)
    copier.run()
