# Enhancing Zero-Shot Learning: Integrating CLIP Embeddings with Knowledge Graphs and Graph Convolutional Networks

**Authors:** Charith Purushotham, Arjyahi Bhattacharya  
**Course:** CSCI 5922, University of Colorado Boulder

---

## 🧠 Introduction

Generalizing to novel, unseen concepts is a hallmark of human intelligence—an ability that current machine learning systems still struggle to replicate effectively.

In many real-world applications such as medical diagnostics, wildlife classification, and industrial defect detection, collecting labeled data for every category is impractical. This motivates **Zero-Shot Learning (ZSL)**, where models must classify unseen classes using auxiliary semantic information.

Our goal is to develop a zero-shot classification system that leverages:
- Structured **knowledge graphs**
- **CLIP** multimodal embeddings
- **Graph Convolutional Networks (R-GCNs)**

---

## ⚙️ Installation

```bash
# 1. Create a virtual environment
python -m venv venv

# 2. Activate the environment
# Windows
venv/Scripts/activate
# macOS/Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```
---

## 📁 Project Structure
```
.
├── create_graph.py
├── train_rgcn.py
├── train_mlp.py
├── Dataloader/
│   └── dataloader.py
├── inference/
│   └── inference_pipeline.py
├── utils/
│   ├── generate_clip_embeddings.py
│   └── image_embedding_utils.py
├── test_module/
│   ├── rgcn_test.py
│   └── mlp_test.py
├── data/
│   └── Animals_with_Attributes2/
├── class_wise_embeddings/
├── test_requirements/
│   └── clip_requirement.py
├── checkpoints/
├── output/
└── requirements.txt
```
---
## 🚀 Running the Pipeline

### 🔹 Step 1: Create the Class Relationship Graph
Constructs a weighted pairwise class graph based on the image embeddings or class hierarchy.

```bash
python create_graph.py
```
### 🔹 Step 2: Train the R-GCN Model
Trains a Relational Graph Convolutional Network using the graph and CLIP embeddings.
```bash
python train_rgcn.py \
  data/Animals_with_Attributes2/JPEGImages \
  output/filtered_class_pairwise_weighted_graph.json \
  output/
```

### 🔹 Step 3: Test the R-GCN Module
Tests the trained R-GCN by predicting class embeddings for a given test image.
```bash
python test_module/rgcn_test.py \
  --image_path data/Animals_with_Attributes2/JPEGImages/chimpanzee/chimpanzee_10501.jpg
```

### 🔹 Step 4: Generate CLIP Embeddings
Generates CLIP-based image embeddings for the dataset.
```bash
python utils/generate_clip_embeddings.py \
  data/Animals_with_Attributes2/JPEGImages/
```

### 🔹 Step 5: Train the MLP Classifier
Trains an MLP on top of the embeddings generated to learn the mapping from visual to semantic space.
```bash
python train_mlp.py \
  class_wise_embeddings/train-image-embeddings-350 \
  output/
```

### 🔹 Step 6: Test the MLP Model
Tests the final trained model on a given image and compares the predicted class to the actual.
```bash
python test_module/mlp_test.py \
  checkpoints/final_model.pt \
  output/reordered_prototypes.pt \
  data/Animals_with_Attributes2/JPEGImages/chimpanzee/chimpanzee_10501.jpg
```

### 🔹 Step 7: Full Inference Pipeline
Runs the end-to-end inference pipeline from image to predicted label.
```bash
python inference/inference_pipeline.py \
  --checkpoint_path checkpoints/final_model.pt \
  --prototype_path output/reordered_prototypes.pt \
  --embedding_dir data/image-embeddings-2.0/test-image-embeddings-20
```








