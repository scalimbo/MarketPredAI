# Marketing Campaign Response Predictor - Multi-Model Version

## 📊 Core Features

### 1. **Dual Task Support: Classification & Regression**

The application now supports **BOTH classification AND regression** tasks with automatic detection:

- **🔢 Regression Models** - Predict continuous numeric values (price, duration, scores, etc.)
  - Random Forest Regressor
  - KNeighbors Regressor
  - Support Vector Regressor (SVR)
  - Neural Network Regressor (MLPRegressor)

- **📊 Classification Models** - Predict discrete categories (yes/no, response/no response, etc.)
  - Random Forest Classifier
  - K-Nearest Neighbors Classifier (KNN)
  - Support Vector Machine Classifier (SVM)
  - Neural Network Classifier (MLPClassifier)

**Automatic Task Detection**: App detects if target column has:

- **>50 unique values or float type** → Uses regression models & metrics
- **<50 unique values** → Uses classification models & metrics

### 2. **Flexible CSV Dataset Support**

- ✅ **Works with ANY CSV structure** - No fixed columns required
- ✅ **Auto-detects data types** - Numeric vs Categorical columns
- ✅ **Handles missing values** - Mean imputation for numeric, mode for categorical
- ✅ **Auto-target selection** - Last column automatically used as prediction target
- ✅ **Dynamic feature detection** - Creates UI elements based on actual column types

### 3. **Enhanced Tabs**

#### Tab 1: Data Management

- Upload custom CSV datasets with any structure
- Load sample dataset
- View dataset overview, statistics, and missing values
- Handles datasets with any number of columns

#### Tab 2: Model Training & Preprocessing

- **Auto-preprocesses data** - Handles missing values automatically
- **Detects task type** - Shows if classification or regression was detected
- **Select which model to train** using a dropdown menu
- View model-specific information and characteristics
- All models are stored in session state for comparison

#### Tab 3: Model Comparison

- View metrics from all trained models side-by-side
- **Smart metric comparison**:
  - Classification: Test Accuracy, Precision, Recall, F1-Score
  - Regression: MAE, RMSE, R² Score
- Interactive bar charts showing performance comparison
- Identify which model performs best for each metric

#### Tab 4: Results & Performance Analysis

- Display task-appropriate metrics:
  - **Classification**: Accuracy, Precision, Recall, F1-Score, Confusion Matrix, Classification Report
  - **Regression**: MAE, RMSE, R² Score
- Confusion matrix visualization (classification only)
- **Smart Feature Importance** - Shows for Random Forest and tree-based models only
- Detailed performance reports

#### Tab 5: Make Prediction

- **Dynamic input UI** - Generated based on actual feature types:
  - **Numeric features**: Sliders with data min/max ranges
  - **Categorical features**: Dropdown with actual category values
- **User-friendly display** - Shows actual values, not scaled numbers
- Get predictions:
  - **Classification**: Probability scores and confidence breakdown
  - **Regression**: Predicted numeric value
- Input summary showing exactly what was entered

#### Tab 6: History & Database

- View prediction and model training history
- Load previously trained models
- Export data to CSV
- Clear database when needed

## 🔧 How to Use

### Basic Workflow

1. **Upload or Load Data** (Tab 1)
   - Click "Load Sample Dataset" or upload your own CSV
   - App auto-detects columns and data types

2. **Preprocess & Train** (Tab 2)
   - Click "Preprocess Data" → App handles missing values automatically
   - App detects task type (regression or classification)
   - Select a model from dropdown
   - Click "Train Model"
   - Repeat for other models

3. **Compare Performance** (Tab 3)
   - View side-by-side metrics comparison
   - See appropriate metrics for your task type
   - Visual charts show relative performance

4. **Make Predictions** (Tab 5)
   - Use slider/dropdown inputs that match your data
   - Get predictions in human-readable format
   - View confidence scores (classification) or predicted values (regression)

## 📈 Task-Specific Capabilities

### Classification Mode

```
Target: Discrete categories (Yes/No, 0/1, Class A/B/C, etc.)
Models: RandomForest, KNN, SVM, Neural Network (all with probability support)
Metrics: Accuracy, Precision, Recall, F1-Score, Confusion Matrix
Output: Predicted class + probability breakdown
```

### Regression Mode

```
Target: Continuous numeric values (price, duration, score, etc.)
Models: RandomForest, KNeighbors, SVM, Neural Network (all numeric prediction)
Metrics: MAE, RMSE, R² Score
Output: Predicted numeric value
```

## 🔧 Model Hyperparameters

Default hyperparameters are configured for each model:

```python
Random Forest: n_estimators=100, max_depth=10, random_state=42
KNN: n_neighbors=5
SVM: kernel='rbf', C=1.0
Neural Network: hidden_layers=(100, 50), max_iter=500, early_stopping=True
```

## 💾 Database Features

- **Automatic Saving**: All predictions and model metrics are saved to SQLite database
- **Model Management**: Save and load trained models
- **History Tracking**: Complete training and prediction history
- **Export Capability**: Download predictions and metrics as CSV
- **Model persistence**: Models stored with scalers for consistent predictions

## 🎯 Key Improvements from Original

✅ Supports both classification AND regression automatically  
✅ Works with ANY CSV structure and column types  
✅ Auto-detects and handles missing values  
✅ Auto-selects last column as target  
✅ Dynamic input generation (sliders for numeric, dropdowns for categories)  
✅ Human-readable prediction interface (no scaled numbers shown to users)  
✅ Task-appropriate metrics display  
✅ Train and compare multiple models in one session  
✅ Visual performance comparison charts  
✅ Model selection for predictions  
✅ Smart feature importance (works for tree-based models)  
✅ Comprehensive model comparison tab  
✅ Database history tracking  
✅ Flexible deployment - works with any real-world dataset

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

## 📊 Example Use Cases

### Classification Example

- **Dataset**: Customer marketing response (yes/no)
- **Task Detection**: Automatic
- **Models Used**: All 4 classifiers
- **Output**: Probability of response + confidence breakdown

### Regression Example

- **Dataset**: Customer lifetime value prediction
- **Task Detection**: Automatic
- **Models Used**: All 4 regressors
- **Output**: Predicted numeric value (e.g., $5,250.00)

### Multi-Task Workflow

1. Train with classification target (e.g., "response")
2. Compare all 4 classification models
3. Later, train with regression target (e.g., "duration")
4. Compare all 4 regression models
5. Export results from both analyses
