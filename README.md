# 🌿 Black Sigatoka Early Stage Detection

## 📋 Project Overview

This project detects and classifies Black Sigatoka disease stages in banana leaves using YOLO object detection. The system can identify 7 classes: Healthy, Stage1, Stage2, Stage3, Stage4, Stage5, and Stage6.

---

## 📚 Documentation

**📖 [Complete Workflow Documentation](WORKFLOW_DOCUMENTATION.md)** - Comprehensive guide covering:
- Complete workflow diagram
- Preprocessing specifications and steps
- Hyperparameter tuning guide
- Model training process
- Validation and testing procedures
- Data split proportions (70/15/15)
- All three notebooks explained

**🐳 [Docker Deployment Guide](DOCKER_DEPLOYMENT.md)** - Complete Docker setup:
- Dockerfile and docker-compose.yml
- Quick start commands
- Production deployment tips
- Troubleshooting guide

**This README** - Quick reference and getting started guide

---

## 🎯 Ano ang Gagawin Mo? (What You Need to Do)

### **Step 1: Data Preprocessing** (`data-labeling-classification.ipynb`)

**Gawin mo:**
1. **Collect images** - Organize banana leaf images by stage folders (Stage1, Stage2, etc.)
2. **Quality check** - Run quality assessment to filter out blurry/dark images
3. **Image tiling** - Split large images into 256x256 tiles for more training data
4. **Data splitting** - Split into 70% train / 15% validation / 15% test
5. **Augmentation** - Apply augmentation to training set only
6. **YOLO conversion** - Convert to YOLO format (images + labels)

**Output:** YOLO-formatted dataset with `data.yaml` configuration file

---

### **Step 2: Dataset Merging** (`bsed-datasets-merge.ipynb`) [Optional but Recommended]

**Gawin mo:**
1. **Load datasets** - Load BSED dataset and Roboflow dataset
2. **Class mapping** - Map different class names to unified scheme:
   - Functional → Healthy
   - Mild → Stage4
   - Moderate → Stage5
   - Severe → Stage6
3. **Merge datasets** - Combine both datasets with updated class IDs
4. **Create combined data.yaml** - Generate unified configuration

**Output:** `combined_yolo_dataset/` with 7 classes (Healthy, Stage1-6)

**Why merge?**
- BSED dataset has Stage1, Stage2, Stage3
- Roboflow dataset has Functional, Mild, Moderate, Severe
- Combined = More training data with all 7 classes

---

### **Step 3: Model Training** (`bsed-training.ipynb`)

**Gawin mo:**
1. **Load dataset** - Load the YOLO dataset from preprocessing
2. **Setup model** - Load YOLO12n (or larger model)
3. **Configure hyperparameters** - Set learning rate, batch size, etc.
4. **Train model** - Run training with validation monitoring
5. **Monitor training** - Check for overfitting/underfitting
6. **Save best model** - Model automatically saves best weights

**Output:** Trained model weights (`best.pt` and `last.pt`)

---

### **Step 4: Evaluation** (`bsed-training.ipynb`)

**Gawin mo:**
1. **Test set evaluation** - Run final evaluation on test set (ONCE only!)
2. **Check metrics** - Review mAP50, precision, recall
3. **Analyze results** - Check per-class performance
4. **Visualize predictions** - See model predictions on test images

**Output:** Performance metrics and visualizations

---

## 📊 Data Split Proportions

### ✅ **ACTUAL SPLIT: 70% / 15% / 15%**

> **📖 For detailed specifications, see [WORKFLOW_DOCUMENTATION.md](WORKFLOW_DOCUMENTATION.md)**

```
┌─────────────────────────────────────────┐
│         TOTAL DATASET (100%)            │
├─────────────────────────────────────────┤
│                                         │
│  ┌──────────────┐  ┌──────────┐       │
│  │   TRAIN      │  │   VAL    │       │
│  │    70%       │  │   15%    │       │
│  │              │  │          │       │
│  │  • Learning  │  │  • Tuning │       │
│  │  • Fitting  │  │  • Early  │       │
│  │             │  │    Stop   │       │
│  └──────────────┘  └──────────┘       │
│                                         │
│         ┌──────────┐                  │
│         │   TEST   │                  │
│         │   15%    │                  │
│         │          │                  │
│         │  • Final │                  │
│         │  • Eval  │                  │
│         └──────────┘                  │
└─────────────────────────────────────────┘
```

**Bakit ganito ang split?**
- **70% Training**: Kailangan ng maraming data para matuto ang model
- **15% Validation**: Para sa hyperparameter tuning at early stopping
- **15% Test**: Para sa final evaluation (HUWAG TOUCHIN pagkatapos ng training!)

---

## 🔄 Complete Workflow

```
1. PREPROCESSING (data-labeling-classification.ipynb)
   ↓
   Raw Images → Quality Check → Tiling → Splitting → Augmentation → YOLO Format
   
2. DATASET MERGING (bsed-datasets-merge.ipynb) [NEW!]
   ↓
   BSED Dataset + Roboflow Dataset → Class Mapping → Combined Dataset
   
3. TRAINING (bsed-training.ipynb)
   ↓
   Load Dataset → Setup Model → Configure → Train → Monitor → Save Best Model
   
4. EVALUATION (bsed-training.ipynb)
   ↓
   Load Best Model → Test Set Evaluation → Metrics → Visualization
```

> **📖 See [WORKFLOW_DOCUMENTATION.md](WORKFLOW_DOCUMENTATION.md) for complete workflow diagram and detailed steps**

---

## 🔧 Preprocessing Steps (Detailed)

### **Step 1: Image Quality Assessment**

**Ano ang ginagawa:**
- Checks image blur (Laplacian variance ≥ 100.0)
- Checks brightness (20-240 range)
- Checks resolution (minimum 256x256)
- Checks file size (50KB - 10MB)

**Code location:** `data-labeling-classification.ipynb` Cell 2-4

---

### **Step 2: Image Tiling**

**Ano ang ginagawa:**
- Splits large images into 256x256 pixel tiles
- Creates more training samples from limited data
- Each tile inherits class label from parent image

**Code location:** `data-labeling-classification.ipynb` Cell 5-6

---

### **Step 3: Data Splitting**

**Ano ang ginagawa:**
```python
# Step 1: 70% train, 30% val+test
train_df, val_test_df = train_test_split(
    labels_df, 
    test_size=0.3,  # 30% for val+test
    stratify=labels_df['class_label']  # Maintain class balance
)

# Step 2: 30% → 15% val + 15% test
val_df, test_df = train_test_split(
    val_test_df,
    test_size=0.5,  # Split 30% into 15% val + 15% test
    stratify=val_test_df['class_label']
)
```

**Code location:** `data-labeling-classification.ipynb` Cell 8

---

### **Step 4: Data Augmentation**

**Ano ang ginagawa:**
- Applied to TRAIN set only (not val/test)
- Rotation, brightness adjustment, crop/zoom, horizontal flip
- Doubles training data (436 → 872 images)

**Code location:** `data-labeling-classification.ipynb` Cell 13-14

---

### **Step 5: YOLO Format Conversion**

**Ano ang ginagawa:**
- Converts annotations to YOLO format
- Creates `data.yaml` configuration file
- Organizes into train/valid/test folders with images/ and labels/ subfolders

**Code location:** `data-labeling-classification.ipynb` Cell 25-26

---

## ⚙️ Hyperparameter Tuning

### **Key Hyperparameters to Tune:**

| Parameter | Options | Default | Purpose |
|-----------|---------|---------|---------|
| **Learning Rate (lr0)** | 0.0001 - 0.01 | 0.001 | Controls how fast model learns |
| **Batch Size** | 8, 16, 32, 64 | 32 | Images processed per batch |
| **Image Size (imgsz)** | 640, 736, 1280 | 736 | Input image resolution |
| **Optimizer** | SGD, Adam, AdamW | AdamW | Optimization algorithm |
| **Epochs** | 10 - 100+ | 10 | Number of training cycles |

### **How to Tune:**

1. **Start with defaults** - Use recommended values first
2. **Train for few epochs** - Test with 5-10 epochs
3. **Check validation metrics** - Look at mAP50 on validation set
4. **Adjust one at a time** - Change one parameter, test, then change another
5. **Use validation set** - Never use test set for tuning!
6. **Document changes** - Keep track of what works

**Code location:** `bsed-training.ipynb` Cell 6 (training_config dictionary)

---

## 🚀 Model Training Process

### **Training Configuration (Current Setup):**

```python
training_config = {
    'epochs': 10,           # Training cycles
    'batch': 32,            # Images per batch
    'imgsz': 736,          # Image size
    'optimizer': 'AdamW',  # Optimizer
    'lr0': 0.001,          # Learning rate
    'patience': 20,        # Early stopping patience
    # ... more parameters
}
```

### **What Happens During Training:**

1. **For each epoch:**
   - Load batch from TRAIN set
   - Apply augmentation
   - Forward pass → predictions
   - Compute loss (box + classification + DFL)
   - Backward pass → gradients
   - Update weights

2. **After each epoch:**
   - Validate on VALIDATION set
   - Compute mAP50, precision, recall
   - Check for early stopping
   - Save best model

3. **Training stops when:**
   - Max epochs reached
   - Early stopping triggered (no improvement for 20 epochs)
   - Manual stop

**Code location:** `bsed-training.ipynb` Cell 6

---

## 📈 Validation & Testing

### **Validation (During Training):**

**Purpose:** Monitor training progress and prevent overfitting

**When:** After each epoch

**Metrics:**
- **mAP50**: Mean Average Precision at IoU=0.5 (main metric)
- **mAP50-95**: mAP across IoU 0.5-0.95 (stricter metric)
- **Precision**: How many detections are correct?
- **Recall**: How many actual objects were found?
- **Loss**: Training and validation loss

**What to look for:**
- ✅ **Good**: mAP50 > 0.7, train loss decreasing, val loss decreasing
- ⚠️ **Overfitting**: Train loss low but val loss high (gap > 0.15)
- ⚠️ **Underfitting**: Both train and val loss high, mAP50 < 0.5

---

### **Testing (Final Evaluation):**

**Purpose:** Get unbiased performance estimate on unseen data

**When:** ONCE ONLY, after training is complete

**Dataset:** TEST set (never used during training!)

**Metrics:**
- Overall mAP50, mAP50-95
- Per-class Average Precision
- Precision, Recall, F1 Score
- Confusion matrix

**⚠️ IMPORTANT:** 
- **HUWAG** gamitin ang test set during training
- **HUWAG** gamitin ang test set for hyperparameter tuning
- **ONCE LANG** i-run ang test set evaluation

**Code location:** `bsed-training.ipynb` Cell 7

---

## 📝 Step-by-Step Guide

### **Phase 1: Preprocessing** (Do this first!)

1. Open `data-labeling-classification.ipynb`
2. Run cells in order:
   - Cell 1-2: Setup and imports
   - Cell 3-4: Image quality assessment
   - Cell 5-6: Image tiling
   - Cell 8: Data splitting (70/15/15)
   - Cell 13-14: Data augmentation
   - Cell 25-26: YOLO format conversion
3. Check output: Should have `yolo_classification_dataset/` folder with `data.yaml`

---

### **Phase 2: Dataset Merging** (Optional but Recommended)

1. Open `bsed-datasets-merge.ipynb`
2. Run cells in order:
   - Cell 0-1: Load BSED and Roboflow datasets
   - Cell 3: Configure class mapping
   - Cell 4: Process Roboflow dataset
   - Cell 5: Process BSED dataset
   - Cell 6: Create combined data.yaml
3. Check output: Should have `combined_yolo_dataset/` folder with 7 classes

---

### **Phase 3: Training** (After preprocessing/merging)

1. Open `bsed-training.ipynb`
2. Run cells in order:
   - Cell 1-2: Install packages and check GPU
   - Cell 3: Imports
   - **Cell 4**: Verify data split proportions (check if 70/15/15)
   - Cell 5: Load dataset
   - Cell 6: Visualize sample annotations
   - Cell 7: Setup YOLO model
   - **Cell 8**: Train model (this takes time!)
   - Cell 9: Evaluate on test set
   - Cell 10: Overfitting analysis
   - Cell 11: Prediction visualization

---

### **Phase 4: Evaluation** (After training)

1. Check training results:
   - Look at `runs/detect/banana_pest_disease_yolo11/results.png`
   - Check mAP50 progression
   - Verify best model saved

2. Run test set evaluation:
   - Cell 9: Comprehensive performance evaluation
   - Check per-class metrics
   - Review confusion matrix

3. Analyze results:
   - Cell 10: Overfitting/underfitting detection
   - Cell 11: Visualize predictions on test images

---

## 🎯 Key Metrics Explained

### **mAP50 (Mean Average Precision at IoU=0.5)**
- **Main metric** for object detection
- **Range**: 0.0 to 1.0 (higher = better)
- **Good**: >0.7
- **Excellent**: >0.8

### **Precision**
- **Question**: How many detections are correct?
- **High precision** = Fewer false positives
- **Formula**: True Positives / (True Positives + False Positives)

### **Recall**
- **Question**: How many actual objects were found?
- **High recall** = Fewer false negatives
- **Formula**: True Positives / (True Positives + False Negatives)

### **F1 Score**
- **Balance** between precision and recall
- **Formula**: 2 × (Precision × Recall) / (Precision + Recall)

---

## ✅ Best Practices

### **DO:**
1. ✅ **Stratified split** - Maintain class distribution
2. ✅ **Use validation set** for hyperparameter tuning
3. ✅ **Save best model** based on validation mAP
4. ✅ **Monitor overfitting** (train vs val loss gap)
5. ✅ **Test only once** on final test set
6. ✅ **Document all hyperparameters** used

### **DON'T:**
1. ❌ **Don't use test set** during training/tuning
2. ❌ **Don't tune on test set** - causes data leakage
3. ❌ **Don't stop early** without patience mechanism
4. ❌ **Don't ignore class imbalance** - use weighted loss if needed
5. ❌ **Don't over-augment** - can hurt performance

---

## 📁 Project Structure

```
ai-banana-earlystage/
├── Data/
│   └── Sigatoka pics/
│       ├── Stage1/
│       ├── Stage2/
│       └── ...
├── notebook/
│   ├── data-labeling-classification.ipynb  # Preprocessing
│   ├── bsed-datasets-merge.ipynb           # Dataset merging
│   └── bsed-training.ipynb                 # Training & Evaluation
├── README.md                                # Quick reference (this file)
├── WORKFLOW_DOCUMENTATION.md                # Complete documentation
└── pyproject.toml
```

---

## 🔗 Quick Reference

### **Notebooks:**

1. **Preprocessing Notebook:**
   - **File**: `data-labeling-classification.ipynb`
   - **Purpose**: Convert raw images to YOLO format
   - **Output**: `yolo_classification_dataset/` with `data.yaml`

2. **Dataset Merging Notebook:**
   - **File**: `bsed-datasets-merge.ipynb`
   - **Purpose**: Combine BSED and Roboflow datasets with class mapping
   - **Output**: `combined_yolo_dataset/` with 7 classes

3. **Training Notebook:**
   - **File**: `bsed-training.ipynb`
   - **Purpose**: Train YOLO model and evaluate
   - **Output**: Trained model in `runs/detect/`

---

## 💡 Tips & Troubleshooting

### **If training is slow:**
- Check if GPU is available (Cell 2)
- Reduce batch size (16 or 8)
- Reduce image size (640 instead of 736)

### **If mAP50 is low (<0.5):**
- Train for more epochs
- Use larger model (yolo12s, yolo12m)
- Check data quality and labels
- Increase augmentation

### **If overfitting (val loss > train loss):**
- Increase augmentation
- Add more training data
- Reduce model size
- Use early stopping

### **If underfitting (both losses high):**
- Train for more epochs
- Use larger model
- Reduce augmentation
- Check data quality

---

## 📚 Additional Resources

- **YOLO Documentation**: https://docs.ultralytics.com/
- **Ultralytics GitHub**: https://github.com/ultralytics/ultralytics
- **YOLO Format Guide**: https://docs.ultralytics.com/datasets/

---

## 🐳 Docker Deployment

### **Quick Start:**

```bash
# Build and run with Docker Compose
docker-compose up --build

# Run in background
docker-compose up -d --build

# Test API
curl http://localhost:8000/health
```

### **Prerequisites:**
- Docker installed
- Model file (`best.pt`) in `models/weights/` directory

**📖 See [DOCKER_DEPLOYMENT.md](DOCKER_DEPLOYMENT.md) for complete guide**

---

## 🎓 Summary

**Ano ang ginagawa mo:**
1. **Preprocess** - Convert raw images to YOLO format (70/15/15 split)
2. **Train** - Train YOLO model with hyperparameters
3. **Evaluate** - Test on unseen test set (once only!)
4. **Deploy** - Dockerize and deploy API

**Important:**
- ✅ Use 70/15/15 split (train/val/test)
- ✅ Tune hyperparameters on validation set
- ✅ Test only once on test set
- ✅ Monitor for overfitting/underfitting
- ✅ Use Docker for easy deployment

**Good luck! 🍀**
