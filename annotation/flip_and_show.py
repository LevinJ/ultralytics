import cv2
import matplotlib.pyplot as plt

# Hardcoded image path
img_path = '/media/levin/DATA/checkpoints/controlnet/data/EOL2/18.02.2025/right/20250303_162900_7e49525e-d440-49b9-b905-d7ae864e5ee4.jpg'

# Read the image
img = cv2.imread(img_path)
if img is None:
    raise FileNotFoundError(f"Image not found: {img_path}")
# Convert BGR (OpenCV) to RGB (matplotlib)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Flip horizontally
img_flipped = cv2.flip(img_rgb, 1)

# Show side by side
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img_rgb)
plt.title('Original')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(img_flipped)
plt.title('Horizontally Flipped')
plt.axis('off')

plt.tight_layout()
plt.show()
