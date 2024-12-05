from flask import Flask, request, render_template
import torch
import torch.nn.functional as F
from PIL import Image
import open_clip
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)

# Load the CLIP model and tokenizer
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
tokenizer = open_clip.get_tokenizer('ViT-B-32')
model.eval()

# Load the dataset with CLIP embeddings
df = pd.read_pickle("image_embeddings.pickle")  # Ensure this file is available

# Load PCA embeddings and PCA model
with open('pca_image_embeddings.pickle', 'rb') as f:
    df_pca = pickle.load(f)

with open('pca_model.pickle', 'rb') as f:
    pca = pickle.load(f)

# Maximum number of principal components
max_components = pca.n_components_

# Helper function to get top results for CLIP embeddings
def get_top_results(query_embedding, df, top_k=5):
    similarities = []
    for _, row in df.iterrows():
        embedding = torch.tensor(row['embedding']).unsqueeze(0)
        embedding = F.normalize(embedding)
        similarity = F.cosine_similarity(query_embedding, embedding, dim=1).item()
        file_name = row['file_name'].split('/')[-1]
        similarities.append((file_name, similarity))
    similarities = sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]
    return similarities

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Retrieve form data
        text_query = request.form.get("text_query", "")
        image_file = request.files.get("image_query")
        hybrid_weight = float(request.form.get("hybrid_weight", 0.5))
        query_type = request.form.get("query_type", "text")

        embedding_type = request.form.get("embedding_type", "clip")
        num_components = int(request.form.get("num_components", 50))

        # Initialize embeddings
        query_embedding = None
        text_embedding = None
        image_embedding = None

        # Validate num_components
        if num_components < 1 or num_components > max_components:
            num_components = max_components

        error_message = None

        if embedding_type == 'clip':
            # Use CLIP embeddings
            if query_type == 'text':
                if text_query:
                    text_tokens = tokenizer([text_query])
                    text_embedding = F.normalize(model.encode_text(text_tokens))
                    query_embedding = text_embedding
                else:
                    error_message = "Please provide a text query for Text Query type."
            elif query_type == 'image':
                if image_file and image_file.filename != '':
                    image = Image.open(image_file).convert('RGB')
                    image = preprocess(image).unsqueeze(0)
                    image_embedding = F.normalize(model.encode_image(image))
                    query_embedding = image_embedding
                else:
                    error_message = "Please provide an image for Image Query type."
            elif query_type == 'hybrid':
                if text_query and image_file and image_file.filename != '':
                    # Process both text and image queries
                    text_tokens = tokenizer([text_query])
                    text_embedding = F.normalize(model.encode_text(text_tokens))
                    image = Image.open(image_file).convert('RGB')
                    image = preprocess(image).unsqueeze(0)
                    image_embedding = F.normalize(model.encode_image(image))
                    # Combine embeddings
                    query_embedding = F.normalize(
                        hybrid_weight * text_embedding + (1 - hybrid_weight) * image_embedding
                    )
                else:
                    error_message = "Please provide both a text query and an image for Hybrid Query type."
            else:
                error_message = "Invalid query type selected."

            # Get top results using CLIP embeddings
            if query_embedding is not None:
                results = get_top_results(query_embedding, df)
            else:
                error_message = error_message or "Invalid query."

        elif embedding_type == 'pca':
            # Use PCA embeddings (only for Image Query)
            if query_type == 'image':
                if image_file and image_file.filename != '':
                    # Load and preprocess image
                    img = Image.open(image_file)
                    img = img.convert('L')  # Convert to grayscale
                    img = img.resize((224, 224))
                    img_array = np.asarray(img, dtype=np.float32) / 255.0  # Normalize pixels
                    img_flat = img_array.flatten().reshape(1, -1)
                    # Transform using PCA
                    query_embedding = pca.transform(img_flat)[:, :num_components]
                else:
                    error_message = "Please provide an image for Image Query type."
            else:
                error_message = "PCA embeddings are only available for Image Query."

            # Get top results using PCA embeddings
            if query_embedding is not None:
                # Use only the specified number of components
                pca_embeddings = np.vstack(df_pca['pca_embedding'].values)[:, :num_components]
                query_embedding = query_embedding[:, :num_components]
                distances = np.linalg.norm(pca_embeddings - query_embedding, axis=1)
                top_k = 5
                top_indices = np.argsort(distances)[:top_k]
                results = [(df_pca.iloc[i]['file_name'], distances[i]) for i in top_indices]
            else:
                error_message = error_message or "Invalid query."
        else:
            error_message = "Invalid embedding type selected."

        if error_message:
            return render_template(
                "index.html",
                error=error_message,
                text_query=text_query,
                hybrid_weight=hybrid_weight,
                query_type=query_type,
                embedding_type=embedding_type,
                num_components=num_components,
                max_components=max_components
            )

        # Render results
        return render_template(
            "index.html",
            results=results,
            text_query=text_query,
            hybrid_weight=hybrid_weight,
            query_type=query_type,
            embedding_type=embedding_type,
            num_components=num_components,
            max_components=max_components
        )

    # For GET requests, render the template with default values
    return render_template(
        "index.html",
        text_query="",
        hybrid_weight=0.5,
        query_type="text",
        embedding_type="clip",
        num_components=50,
        max_components=max_components
    )

if __name__ == "__main__":
    app.run(debug=True)