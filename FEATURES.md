# Marketing Campaign Response Predictor - Multi-Model Version

## 📊 New Features

### 1. **Multi-Model Support**

The application now supports training and comparing **4 different machine learning models**:

- **🌲 Random Forest** - Ensemble learning method using multiple decision trees
- **📍 K-Nearest Neighbors (KNN)** - Instance-based learning algorithm
- **⚔️ Support Vector Machine (SVM)** - Powerful binary classification
- **🧠 Neural Network (MLP)** - Multi-layer perceptron for complex patterns

### 2. **Enhanced Tabs**

#### Tab 1: Data Management

- Upload custom CSV datasets
- Load sample dataset
- View dataset overview, statistics, and missing values

#### Tab 2: Model Training (Enhanced)

- **Select which model to train** using a dropdown menu
- View model-specific information and characteristics
- Preprocess data with one click
- Train the selected model
- All models are stored in session state for comparison

#### Tab 3: Model Comparison (NEW)

- View metrics from all trained models side-by-side
- Compare Test Accuracy, Precision, Recall, and F1-Score
- Interactive bar charts showing performance comparison
- Identify which model performs best for each metric

#### Tab 4: Results (Enhanced)

- Display metrics for the currently selected model
- Confusion matrix visualization
- **Smart Feature Importance** - Only shows for models that support it (Random Forest)
- Detailed classification report

#### Tab 5: Make Prediction (Enhanced)

- **Select which trained model to use** for predictions
- Input customer features
- Get probability scores for both outcomes
- Visual probability breakdown

#### Tab 6: History & Database

- View prediction and model training history
- Load previously trained models
- Export data to CSV
- Clear database when needed

## 🔧 How to Use

### Training Multiple Models

1. **Upload or Load Data** (Tab 1)
   - Click "Load Sample Dataset" or upload your own CSV

2. **Train Models** (Tab 2)
   - Select a model from the dropdown (e.g., "Random Forest")
   - Click "Preprocess Data"
   - Click "Train Model"
   - Repeat for other models (KNN, SVM, Neural Network)

3. **Compare Performance** (Tab 3)
   - View side-by-side metrics comparison
   - Visual charts show relative performance
   - See which model is best for each metric

4. **Make Predictions** (Tab 5)
   - Select which trained model to use
   - Enter customer details
   - See predictions and confidence scores

## 📈 Model Parameters

Default hyperparameters are configured for each model:

```python
Random Forest: n_estimators=100, max_depth=10
KNN: n_neighbors=5
SVM: kernel='rbf', C=1.0, probability=True
Neural Network: hidden_layers=(100, 50), max_iter=500
```

## 💾 Database Features

- **Automatic Saving**: All predictions and model metrics are saved to SQLite database
- **Model Management**: Save and load trained models
- **History Tracking**: Complete training and prediction history
- **Export Capability**: Download predictions and metrics as CSV

## 🎯 Key Improvements

✅ Train and compare multiple models in one session  
✅ Visual performance comparison charts  
✅ Model selection for predictions  
✅ Smart feature importance (works for tree-based models)  
✅ Comprehensive model comparison tab  
✅ Sidebar showing all available models  
✅ Database history tracking

## 📝 Requirements

All dependencies are in `requirements.txt`:

- streamlit >= 1.28.0
- pandas >= 1.5.0
- numpy >= 1.23.0
- scikit-learn >= 1.2.0
- plotly >= 5.13.0

## 🚀 Getting Started

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Run the application:

   ```bash
   streamlit run app.py
   ```

3. Open your browser to `http://localhost:8501`

## 🔬 Model Comparison Example Workflow

1. Load sample data
2. Train Random Forest → Note accuracy ~0.85
3. Train KNN → Note accuracy ~0.82
4. Train SVM → Note accuracy ~0.83
5. Train Neural Network → Note accuracy ~0.84
6. Go to Tab 3 (Model Comparison) to see all metrics side-by-side
7. Use the best performing model for predictions in Tab 5
