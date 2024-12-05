# compute_pca_embeddings.py

# compute_pca_embeddings.py

import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.decomposition import PCA
import pickle

def load_images(image_dir, max_images=None, target_size=(224, 224)):
    images = []
    image_names = []
    for i, filename in enumerate(os.listdir(image_dir)):
        if filename.endswith('.jpg'):
            img = Image.open(os.path.join(image_dir, filename))
            img = img.convert('L')  # Convert to grayscale ('L' mode)
            img = img.resize(target_size)  # Resize to target size
            img_array = np.asarray(img, dtype=np.float32) / 255.0  # Normalize pixel values to [0, 1]
            images.append(img_array.flatten())  # Flatten to 1D
            image_names.append(filename)
        if max_images and i + 1 >= max_images:
            break
    return np.array(images), image_names

# Directory containing images
image_dir = "./static/images"  # Adjust as needed

# Step 2: Train PCA on the first 2,000 images
train_images, train_image_names = load_images(image_dir, max_images=2000)
print(f"Loaded {len(train_images)} images for PCA training.")

# Apply PCA
k = 50  # You can adjust this number as needed
pca = PCA(n_components=k)
pca.fit(train_images)
print(f"Trained PCA on {len(train_images)} samples.")

# Step 3: Transform all images
transform_images, transform_image_names = load_images(image_dir, max_images=10000, target_size=(224, 224))
print(f"Loaded {len(transform_images)} images for transformation.")

# Apply PCA transformation
reduced_embeddings = pca.transform(transform_images)
print(f"Computed PCA embeddings for {len(transform_images)} images.")

# Create a DataFrame to store embeddings and file names
df_pca = pd.DataFrame({
    'file_name': transform_image_names,
    'pca_embedding': list(reduced_embeddings)
})

# Save PCA embeddings and PCA model
with open('pca_image_embeddings.pickle', 'wb') as f:
    pickle.dump(df_pca, f)

with open('pca_model.pickle', 'wb') as f:
    pickle.dump(pca, f)

print("Saved PCA embeddings and PCA model.")