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

## Concepts
### KnowledgeGraphRGCN

The **KnowledgeGraphRGCN** model integrates **Relational Graph Convolutional Networks (R-GCNs)** with **CLIP embeddings** for zero-shot image classification. It utilizes a knowledge graph of class relationships and CLIP-generated text embeddings to refine class representations through the R-GCN, enabling the model to classify unseen classes.

#### Key Components:
1. **R-GCN**: Refines class embeddings by considering relational information in a graph of classes and their relationships.
2. **CLIP Embeddings**: Uses CLIP to generate shared text and image embeddings for semantic understanding across modalities.
3. **Knowledge Graph**: Built from triples (subject, object, relationship) to represent class relationships, which guide the R-GCN.
4. **Training Objective**: Aligns R-GCN-learned embeddings with original CLIP embeddings using cosine similarity.

This model can classify images based on unseen classes by leveraging the knowledge graph and CLIP's multimodal embeddings.

### PrototypeRefinementLoss
The `PrototypeRefinementLoss` combines **alignment** and **separation** objectives to refine class prototypes:
1. **Alignment Loss**: Uses **cosine similarity** between the refined and original prototypes to encourage alignment, ensuring that the refined prototypes stay close to the original ones in the learned feature space.
2. **Separation Loss**: Uses **MSE loss** on the cosine similarity matrix of the refined prototypes, ensuring that distinct classes remain far apart. The separation is guided by a **target similarity matrix**, which provides explicit class relationship information.
#### **Why It's Unique**
This loss is unique because it combines **cosine similarity** for alignment with **MSE loss** for separation, while using a **target similarity matrix** to guide the separation. While similar concepts exist in zero-shot learning and metric learning, this specific combination of techniques for refining prototypes is relatively novel, especially with the added semantic guidance from the target similarity matrix.

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
---
## 📊 Evaluation Metrics

To assess the performance of our zero-shot classification system, we compute the following metrics:

### 🔹 Top-1 Accuracy
- **Definition:** The fraction of test samples where the top predicted class matches the ground truth label.
- **Interpretation:** Measures direct classification performance without accounting for semantic similarity.

### 🔹 Adjusted Accuracy (Top-2 + Semantic Match)
- **Definition:** Considers a prediction correct if:
  - The top-1 prediction is correct, **OR**
  - The second-best prediction is correct, **OR**
  - The predicted class belongs to the same semantic group as the true class.
- **Interpretation:** Provides a more relaxed and human-aligned metric, recognizing cases where semantically related predictions are acceptable (e.g., predicting “zebra” instead of “horse”).

### 🔹 Macro Precision
- **Definition:** Precision computed independently for each class and averaged.
- **Interpretation:** Ensures equal weight is given to all classes regardless of sample imbalance.

### 🔹 Macro Recall
- **Definition:** Recall computed independently for each class and averaged.
- **Interpretation:** Captures how well the model identifies each class overall.

### 🔹 Macro F1 Score
- **Definition:** Harmonic mean of macro precision and macro recall.
- **Interpretation:** Balances precision and recall at a class-averaged level, providing a single performance summary.


---








