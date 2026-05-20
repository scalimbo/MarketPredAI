# Marketing Campaign Response Predictor

A Streamlit application to predict whether customers will respond to marketing campaigns using machine learning.

## Features

**Data Management**

- Upload CSV datasets
- Load sample data for testing
- View data statistics and missing values

  **Model Training Pipeline**

- Data preprocessing (encoding, scaling)
- Random Forest classifier training
- Train/test split validation

  **Results & Analytics**

- Performance metrics (Accuracy, Precision, Recall, F1-Score)
- Confusion matrix visualization
- Feature importance analysis
- Detailed classification report

  **Prediction Interface**

- User-friendly input form for customer details
- Real-time predictions with probability scores
- Visual confidence breakdown

## Installation & Setup

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Run the Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## How to Use

### 1. **Data Management Tab**

- Click "Load Sample Dataset" to use the built-in sample data
- Or upload your own CSV file with the following columns:
  - `Age` (integer)
  - `Income` (integer)
  - `Previous_Purchases` (integer)
  - `Email_Opened` (Yes/No or 1/0)
  - `Website_Visits` (integer)
  - `Response` (0 for no, 1 for yes) - target variable

### 2. **Model Training Tab**

- Click "Preprocess Data" to prepare the dataset
- Click "Train Model" to train the Random Forest classifier
- The model automatically handles scaling and encoding

### 3. **Results Tab**

- View performance metrics (Accuracy, Precision, Recall, F1-Score)
- Analyze the confusion matrix
- Check feature importance rankings
- Review detailed classification metrics

### 4. **Prediction Tab**

- Enter customer details using the interactive inputs
- Age slider (18-100)
- Annual income input
- Previous purchases counter
- Email opened toggle (Yes/No)
- Website visits slider (0-100)
- Click "Make Prediction" to get results with probability

## Dataset Format Example

```csv
Age,Income,Previous_Purchases,Email_Opened,Website_Visits,Response
35,50000,5,Yes,15,1
42,75000,12,Yes,25,1
28,35000,2,No,5,0
55,120000,20,Yes,40,1
```

## Features in Detail

### Input Fields

- **Age**: Customer's age (18-100 years)
- **Income**: Annual income in dollars
- **Previous Purchases**: Number of previous purchases
- **Email Opened**: Whether the customer opened previous marketing emails
- **Website Visits**: Number of website visits in the last 30 days

### Output

- **Prediction**: Whether the customer will respond (Yes/No)
- **Probability**: Confidence percentage (0-100%)
- **Breakdown Chart**: Visual representation of response probability

## Model Information

- **Algorithm**: Random Forest Classifier
- **Training**: 80% training, 20% testing split
- **Preprocessing**:
  - Standard scaling of numerical features
  - One-hot encoding of categorical features
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score

## Future Enhancements

- Integration with real AI model when ready
- Model persistence (save/load trained models)
- A/B testing framework
- Customer segmentation analysis
- Historical prediction tracking

## Troubleshooting

**App won't start:**

- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check Python version is 3.8 or higher

**File upload issues:**

- Ensure CSV has required columns (case-sensitive)
- Check for proper data types

**Model training errors:**

- Ensure dataset has proper 'Response' column
- Check for NaN values in the dataset

## Requirements

- Python 3.8+
- Streamlit 1.28+
- pandas 2.0+
- scikit-learn 1.3+
- plotly 5.17+
