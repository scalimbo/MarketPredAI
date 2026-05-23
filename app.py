import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import plotly.graph_objects as go
import plotly.express as px
import io
import pickle
import os
from database import (
    init_database, save_prediction, get_prediction_history,
    save_model_metrics, get_model_metrics_history, save_model, load_model,
    get_model_history, get_statistics, clear_all_data
)

# Page configuration
st.set_page_config(
    page_title="Marketing Campaign Response Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize database
init_database()

# Initialize theme in session state
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'

# SIDEBAR THEME TOGGLE - Must be first to apply before rendering
with st.sidebar:
    st.markdown("---")
    st.subheader("⚙️ Settings")
    
    theme_col1, theme_col2 = st.columns([2, 1])
    with theme_col1:
        st.write("Theme Mode")
    with theme_col2:
        theme_option = st.radio(
            "Select theme",
            ["🌙 Dark", "☀️ Light"],
            label_visibility="collapsed",
            horizontal=True,
            key="theme_toggle"
        )
        
        new_theme = 'dark' if theme_option == "🌙 Dark" else 'light'
        st.session_state.theme = new_theme
    
    st.markdown("---")
    st.subheader("📊 Model Selection")
    st.write("**Multi-Model Support:** Train and compare multiple AI models")
    st.markdown("""
    - 🌲 **Random Forest** - Ensemble learning
    - 📍 **K-Nearest Neighbors** - Instance-based
    - ⚔️ **SVM** - Support Vector Machine
    - 🧠 **Neural Network** - Deep learning
    """)
    
    st.markdown("---")

# Custom CSS for Light and Dark Modes
def apply_theme(theme_mode):
    if theme_mode == 'dark':
        st.markdown("""
            <style>
            body {
                background-color: #0e1117 !important;
                color: #e0e0e0 !important;
            }
            .main-header {
                font-size: 3em;
                color: #00D9FF !important;
                text-align: center;
                margin-bottom: 10px;
                text-shadow: 0 0 10px rgba(0, 217, 255, 0.3);
            }
            .section-header {
                font-size: 1.8em;
                color: #00D9FF !important;
                margin-top: 20px;
                margin-bottom: 10px;
            }
            .stMetric {
                background: linear-gradient(135deg, #161b22 0%, #0d1117 100%) !important;
                border-radius: 8px;
                padding: 15px !important;
            }
            .stDataFrame {
                background-color: #161b22 !important;
            }
            </style>
        """, unsafe_allow_html=True)
    else:  # light mode
        st.markdown("""
            <style>
            body {
                background-color: #ffffff !important;
                color: #333333 !important;
            }
            .main-header {
                font-size: 3em;
                color: #1f77b4 !important;
                text-align: center;
                margin-bottom: 10px;
                text-shadow: 0 0 5px rgba(31, 119, 180, 0.1);
            }
            .section-header {
                font-size: 1.8em;
                color: #1f77b4 !important;
                margin-top: 20px;
                margin-bottom: 10px;
            }
            .stMetric {
                background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%) !important;
                border-radius: 8px;
                padding: 15px !important;
            }
            .stDataFrame {
                background-color: #f8f9fa !important;
            }
            </style>
        """, unsafe_allow_html=True)

# Apply the current theme IMMEDIATELY after sidebar toggle
apply_theme(st.session_state.theme)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'preprocessed_data' not in st.session_state:
    st.session_state.preprocessed_data = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'metrics' not in st.session_state:
    st.session_state.metrics = None
if 'confusion_mat' not in st.session_state:
    st.session_state.confusion_mat = None
if 'selected_model_type' not in st.session_state:
    st.session_state.selected_model_type = 'Random Forest'
if 'all_models' not in st.session_state:
    st.session_state.all_models = {}
if 'model_comparison' not in st.session_state:
    st.session_state.model_comparison = {}
if 'target_column' not in st.session_state:
    st.session_state.target_column = None
if 'feature_names' not in st.session_state:
    st.session_state.feature_names = []
if 'label_encoders' not in st.session_state:
    st.session_state.label_encoders = {}
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'y_pred' not in st.session_state:
    st.session_state.y_pred = None
if 'feature_metadata' not in st.session_state:
    st.session_state.feature_metadata = {}
if 'original_X' not in st.session_state:
    st.session_state.original_X = None
if 'task_type' not in st.session_state:
    st.session_state.task_type = 'classification'  # 'classification' or 'regression'

# Model configuration
MODEL_PARAMS = {
    'Random Forest': {
        'n_estimators': 100,
        'random_state': 42,
        'max_depth': 10
    },
    'KNN': {
        'n_neighbors': 5
    },
    'SVM': {
        'kernel': 'rbf',
        'C': 1.0,
        'probability': True,
        'random_state': 42
    },
    'Neural Network': {
        'hidden_layer_sizes': (100, 50),
        'max_iter': 500,
        'random_state': 42,
        'early_stopping': True
    }
}

def get_model(model_type, task_type='classification'):
    """Factory function to create model instances
    task_type: 'classification' or 'regression'
    """
    if task_type == 'regression':
        # Regression models
        if model_type == 'Random Forest':
            return RandomForestRegressor(**MODEL_PARAMS['Random Forest'])
        elif model_type == 'KNN':
            return KNeighborsRegressor(**MODEL_PARAMS['KNN'])
        elif model_type == 'SVM':
            svm_params = MODEL_PARAMS['SVM'].copy()
            svm_params.pop('probability', None)  # Remove probability param for regression
            svm_params.pop('random_state', None)  # Remove random_state for SVR
            return SVR(**svm_params)
        elif model_type == 'Neural Network':
            return MLPRegressor(**MODEL_PARAMS['Neural Network'])
        else:
            return RandomForestRegressor(**MODEL_PARAMS['Random Forest'])
    else:
        # Classification models (default)
        if model_type == 'Random Forest':
            return RandomForestClassifier(**MODEL_PARAMS['Random Forest'])
        elif model_type == 'KNN':
            return KNeighborsClassifier(**MODEL_PARAMS['KNN'])
        elif model_type == 'SVM':
            return SVC(**MODEL_PARAMS['SVM'])
        elif model_type == 'Neural Network':
            return MLPClassifier(**MODEL_PARAMS['Neural Network'])
        else:
            return RandomForestClassifier(**MODEL_PARAMS['Random Forest'])

# Main title
st.markdown("<h1 class='main-header'>Marketing Campaign Response Predictor</h1>", unsafe_allow_html=True)
st.markdown("---")

# Create tabs for different sections
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Data Management", "Model Training", "Model Comparison", "Results", "Make Prediction", "History & Database"])

# ==================== TAB 1: DATA MANAGEMENT ====================
with tab1:
    st.markdown("<h2 class='section-header'>Data Upload & Exploration</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type=['csv'])
    
    with col2:
        if st.button("Load Sample Dataset", key="sample_data"):
            # Create sample dataset
            sample_data = {
                'Age': np.random.randint(18, 75, 100),
                'Income': np.random.randint(20000, 150000, 100),
                'Previous_Purchases': np.random.randint(0, 20, 100),
                'Email_Opened': np.random.choice(['Yes', 'No'], 100),
                'Website_Visits': np.random.randint(0, 50, 100),
                'Response': np.random.choice([0, 1], 100)
            }
            st.session_state.data = pd.DataFrame(sample_data)
            st.success("Sample dataset loaded!")
    
    if uploaded_file is not None:
        st.session_state.data = pd.read_csv(uploaded_file)
        st.success("Dataset uploaded successfully!")
    
    # Display dataset
    if st.session_state.data is not None:
        st.subheader("Dataset Overview")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rows", st.session_state.data.shape[0])
        with col2:
            st.metric("Columns", st.session_state.data.shape[1])
        with col3:
            if 'Response' in st.session_state.data.columns:
                response_rate = (st.session_state.data['Response'].sum() / len(st.session_state.data)) * 100
                st.metric("Response Rate", f"{response_rate:.1f}%")
        
        st.write(st.session_state.data.head(10))
        
        st.subheader("Data Statistics")
        st.write(st.session_state.data.describe())
        
        st.subheader("Missing Values")
        missing = st.session_state.data.isnull().sum()
        if missing.sum() == 0:
            st.success("No missing values detected!")
        else:
            st.warning(missing[missing > 0])
    else:
        st.info("Upload a CSV file or load the sample dataset to get started")


# ==================== TAB 2: MODEL TRAINING ====================
with tab2:
    st.markdown("<h2 class='section-header'>Model Training Pipeline</h2>", unsafe_allow_html=True)
    
    if st.session_state.data is None:
        st.warning("Please upload or load a dataset first!")
    else:
        col1, col2 = st.columns(2)
        
        # Model selection
        with col1:
            st.subheader("Step 0: Select Model")
            selected_model = st.selectbox(
                "Choose a machine learning model",
                ['Random Forest', 'KNN', 'SVM', 'Neural Network'],
                key="model_selector"
            )
            st.session_state.selected_model_type = selected_model
            
            st.info(f"📊 **Selected Model:** {selected_model}")
        
        with col2:
            st.subheader("Model Information")
            model_info = {
                'Random Forest': '🌲 Ensemble method using multiple decision trees. Best for balanced performance.',
                'KNN': '📍 Instance-based learning. Good for small datasets.',
                'SVM': '⚔️ Support Vector Machine. Excellent for binary classification.',
                'Neural Network': '🧠 Multi-layer perceptron. Powerful for complex patterns.'
            }
            st.markdown(f"**{model_info[selected_model]}**")
        
        # PREPROCESSING
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Step 1: Data Preprocessing")
            
            # Auto-select last column as target (no selectbox needed)
            target_col = st.session_state.data.columns[-1]
            st.info(f"📍 **Target column (auto-selected):** {target_col}")
            
            if st.button("Preprocess Data", key="preprocess"):
                try:
                    with st.spinner("Preprocessing data..."):
                        df = st.session_state.data.copy()
                        
                        # Check if target column exists
                        if target_col not in df.columns:
                            st.error(f"Target column '{target_col}' not found!")
                        else:
                            # Separate features and target
                            y = df[target_col].copy()
                            X = df.drop(target_col, axis=1).copy()
                            
                            # Handle missing values in features
                            initial_missing = X.isnull().sum().sum()
                            if initial_missing > 0:
                                st.warning(f"⚠️ Found {initial_missing} missing values. Filling with column means...")
                                numeric_cols = X.select_dtypes(include=[np.number]).columns
                                X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].mean())
                                
                                # For categorical columns, fill with mode
                                categorical_cols = X.select_dtypes(include=['object']).columns
                                for col in categorical_cols:
                                    X[col].fillna(X[col].mode()[0] if len(X[col].mode()) > 0 else 'Unknown', inplace=True)
                            
                            # Handle missing values in target
                            if y.isnull().any():
                                st.warning(f"⚠️ Found missing values in target column. Removing rows...")
                                valid_idx = ~y.isnull()
                                X = X[valid_idx]
                                y = y[valid_idx]
                            
                            # Identify numeric and categorical columns
                            numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
                            categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
                            
                            # Encode categorical variables
                            X_encoded = X.copy()
                            label_encoders = {}
                            for col in categorical_cols:
                                le = LabelEncoder()
                                X_encoded[col] = le.fit_transform(X[col].astype(str))
                                label_encoders[col] = le
                            
                            # Handle target variable encoding if categorical
                            if y.dtype == 'object':
                                le_target = LabelEncoder()
                                y = pd.Series(le_target.fit_transform(y.astype(str)), index=y.index)
                            
                            # Combine preprocessed data
                            processed_df = pd.concat([X_encoded, y], axis=1)
                            processed_df.columns = X_encoded.columns.tolist() + [target_col]
                            
                            # Create feature metadata for user-friendly input
                            feature_metadata = {}
                            for col in X.columns:
                                if col in numeric_cols:
                                    feature_metadata[col] = {
                                        'type': 'numeric',
                                        'min': float(X[col].min()),
                                        'max': float(X[col].max()),
                                        'mean': float(X[col].mean())
                                    }
                                else:
                                    feature_metadata[col] = {
                                        'type': 'categorical',
                                        'categories': sorted(X[col].unique().tolist())
                                    }
                            
                            # Store in session state
                            st.session_state.preprocessed_data = processed_df
                            st.session_state.target_column = target_col
                            st.session_state.feature_names = X_encoded.columns.tolist()
                            st.session_state.label_encoders = label_encoders
                            st.session_state.feature_metadata = feature_metadata
                            st.session_state.original_X = X  # Store original data for reference
                            
                            # Determine task type: classification vs regression
                            unique_values = len(y.unique())
                            if unique_values > 50 or y.dtype in ['float64', 'float32']:
                                st.session_state.task_type = 'regression'
                                task_label = "🔢 REGRESSION (Predicting continuous values)"
                            else:
                                st.session_state.task_type = 'classification'
                                task_label = "📊 CLASSIFICATION (Predicting categories)"
                            
                            st.success("✅ Data preprocessed successfully!")
                            st.write(f"**Rows:** {len(processed_df)} | **Features:** {len(X_encoded.columns)}")
                            st.info(f"**Task Type:** {task_label}")
                            st.write(processed_df.head())
                
                except Exception as e:
                    st.error(f"Error during preprocessing: {str(e)}")
        
        # MODEL TRAINING
        with col2:
            st.subheader("Step 2: Train Model")
            
            if st.button("Train Model", key="train_model"):
                if st.session_state.preprocessed_data is None:
                    st.error("Please preprocess the data first!")
                else:
                    try:
                        with st.spinner(f"Training {selected_model}..."):
                            df = st.session_state.preprocessed_data.copy()
                            target_col = st.session_state.target_column
                            task_type = st.session_state.task_type
                            
                            # Prepare features and target
                            if target_col not in df.columns:
                                st.error(f"Target column '{target_col}' not found in preprocessed data")
                                st.stop()
                            
                            X = df.drop(target_col, axis=1)
                            y = df[target_col]
                            
                            # Ensure no NaN values remain
                            if X.isnull().any().any() or y.isnull().any():
                                st.warning("Removing rows with missing values...")
                                valid_idx = ~(X.isnull().any(axis=1) | y.isnull())
                                X = X[valid_idx]
                                y = y[valid_idx]
                            
                            if len(X) < 10:
                                st.error("Not enough valid data samples after removing missing values!")
                                st.stop()
                            
                            # Split data
                            test_size = min(0.2, 0.5)  # Use at least some training data
                            stratify_param = y if (task_type == 'classification' and y.nunique() == 2) else None
                            X_train, X_test, y_train, y_test = train_test_split(
                                X, y, test_size=test_size, random_state=42, stratify=stratify_param
                            )
                            
                            # Scale features
                            scaler = StandardScaler()
                            X_train_scaled = scaler.fit_transform(X_train)
                            X_test_scaled = scaler.transform(X_test)
                            
                            # Train model with appropriate type
                            model = get_model(selected_model, task_type=task_type)
                            model.fit(X_train_scaled, y_train)
                            
                            # Make predictions
                            y_pred_train = model.predict(X_train_scaled)
                            y_pred_test = model.predict(X_test_scaled)
                            
                            # Store in session
                            st.session_state.model = model
                            st.session_state.scaler = scaler
                            st.session_state.model_trained = True
                            st.session_state.selected_model_type = selected_model
                            st.session_state.X_test = X_test
                            st.session_state.y_test = y_test
                            st.session_state.y_pred = y_pred_test
                            
                            # Store model in comparison dictionary
                            st.session_state.all_models[selected_model] = {
                                'model': model,
                                'scaler': scaler,
                                'X_train_scaled': X_train_scaled,
                                'X_test_scaled': X_test_scaled,
                                'y_train': y_train,
                                'y_test': y_test
                            }
                            
                            # Calculate metrics based on task type
                            if task_type == 'regression':
                                metrics = {
                                    'train_mae': mean_absolute_error(y_train, y_pred_train),
                                    'test_mae': mean_absolute_error(y_test, y_pred_test),
                                    'train_mse': mean_squared_error(y_train, y_pred_train),
                                    'test_mse': mean_squared_error(y_test, y_pred_test),
                                    'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
                                    'r2_score': r2_score(y_test, y_pred_test)
                                }
                            else:
                                metrics = {
                                    'train_accuracy': accuracy_score(y_train, y_pred_train),
                                    'test_accuracy': accuracy_score(y_test, y_pred_test),
                                    'precision': precision_score(y_test, y_pred_test, average='weighted', zero_division=0),
                                    'recall': recall_score(y_test, y_pred_test, average='weighted', zero_division=0),
                                    'f1': f1_score(y_test, y_pred_test, average='weighted', zero_division=0)
                                }
                            
                            st.session_state.metrics = metrics
                            if task_type == 'classification':
                                st.session_state.confusion_mat = confusion_matrix(y_test, y_pred_test)
                            st.session_state.feature_names = X.columns.tolist()
                            
                            # Store metrics for comparison
                            st.session_state.model_comparison[selected_model] = metrics
                            
                            # Save metrics to database
                            save_model_metrics(metrics, model_name=selected_model, data_rows=len(df))
                            
                            # Save model and scaler
                            primary_metric = metrics.get('test_accuracy', metrics.get('r2_score', 0))
                            save_model(model, scaler, primary_metric, model_name=selected_model)
                            
                            st.success(f"✅ {selected_model} trained successfully on {len(X)} samples!")
                            
                            # Display metrics
                            if task_type == 'regression':
                                st.write(f"**Train MAE:** {metrics['train_mae']:.4f} | **Test MAE:** {metrics['test_mae']:.4f} | **R² Score:** {metrics['r2_score']:.4f}")
                            else:
                                st.write(f"**Train Accuracy:** {metrics['train_accuracy']:.4f} | **Test Accuracy:** {metrics['test_accuracy']:.4f}")
                    
                    except Exception as e:
                        st.error(f"❌ Error during model training: {str(e)}")
                        st.write("**Tips:** Ensure your dataset has valid data and enough samples for train/test split.")


# ==================== TAB 3: MODEL COMPARISON ====================
with tab3:
    st.markdown("<h2 class='section-header'>Model Performance Comparison</h2>", unsafe_allow_html=True)
    
    if not st.session_state.model_comparison:
        st.info("Train at least one model to see comparison results!")
    else:
        st.subheader("Trained Models")
        trained_models = list(st.session_state.model_comparison.keys())
        st.write(f"**Models trained:** {', '.join(trained_models)}")
        
        # Comparison metrics table
        st.subheader("Metrics Comparison")
        
        comparison_df = pd.DataFrame(st.session_state.model_comparison).T
        comparison_df = comparison_df.round(4)
        st.dataframe(comparison_df, use_container_width=True)
        
        # Visual comparison
        st.subheader("Visual Performance Comparison")
        
        # Prepare data for visualization
        metrics_data = []
        for model_name, metrics in st.session_state.model_comparison.items():
            for metric_name, value in metrics.items():
                metrics_data.append({
                    'Model': model_name,
                    'Metric': metric_name.replace('_', ' ').title(),
                    'Value': value
                })
        
        metrics_plot_df = pd.DataFrame(metrics_data)
        
        # Create comparison chart
        fig = px.bar(
            metrics_plot_df,
            x='Metric',
            y='Value',
            color='Model',
            barmode='group',
            title='Model Performance Metrics Comparison'
        )
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)
        
        # Best model analysis
        st.subheader("Best Models by Metric")
        
        if st.session_state.task_type == 'regression':
            # For regression, find best by R² score, MAE, RMSE
            col1, col2, col3 = st.columns(3)
            
            if 'r2_score' in next(iter(st.session_state.model_comparison.values())):
                best_r2 = max(st.session_state.model_comparison.items(), key=lambda x: x[1].get('r2_score', -1))
                best_mae = min(st.session_state.model_comparison.items(), key=lambda x: x[1].get('test_mae', float('inf')))
                best_rmse = min(st.session_state.model_comparison.items(), key=lambda x: x[1].get('test_rmse', float('inf')))
                
                with col1:
                    st.metric("🏆 Best R² Score", f"{best_r2[0]}", f"{best_r2[1]['r2_score']:.4f}")
                with col2:
                    st.metric("📊 Best MAE", f"{best_mae[0]}", f"{best_mae[1]['test_mae']:.4f}")
                with col3:
                    st.metric("📈 Best RMSE", f"{best_rmse[0]}", f"{best_rmse[1]['test_rmse']:.4f}")
        else:
            # For classification, find best by accuracy, precision, recall, F1
            col1, col2, col3, col4, col5 = st.columns(5)
            
            if 'test_accuracy' in next(iter(st.session_state.model_comparison.values())):
                best_accuracy = max(st.session_state.model_comparison.items(), key=lambda x: x[1]['test_accuracy'])
                best_precision = max(st.session_state.model_comparison.items(), key=lambda x: x[1]['precision'])
                best_recall = max(st.session_state.model_comparison.items(), key=lambda x: x[1]['recall'])
                best_f1 = max(st.session_state.model_comparison.items(), key=lambda x: x[1]['f1'])
                best_train = max(st.session_state.model_comparison.items(), key=lambda x: x[1].get('train_accuracy', 0))
                
                with col1:
                    st.metric("🎯 Best Accuracy", f"{best_accuracy[0]}", f"{best_accuracy[1]['test_accuracy']:.4f}")
                with col2:
                    st.metric("🔍 Best Precision", f"{best_precision[0]}", f"{best_precision[1]['precision']:.4f}")
                with col3:
                    st.metric("📊 Best Recall", f"{best_recall[0]}", f"{best_recall[1]['recall']:.4f}")
                with col4:
                    st.metric("⚖️ Best F1", f"{best_f1[0]}", f"{best_f1[1]['f1']:.4f}")
                with col5:
                    st.metric("🏋️ Best Train Acc", f"{best_train[0]}", f"{best_train[1].get('train_accuracy', 0):.4f}")


# ==================== TAB 4: RESULTS ====================
with tab4:
    st.markdown("<h2 class='section-header'>Model Performance Metrics</h2>", unsafe_allow_html=True)
    
    if not st.session_state.model_trained:
        st.info("Train the model first to see results!")
    else:
        # Show task type
        task_emoji = "🔢" if st.session_state.task_type == 'regression' else "📊"
        st.info(f"{task_emoji} **Task Type:** {st.session_state.task_type.upper()}")
        
        # Metrics display
        st.subheader("Performance Metrics")
        
        if st.session_state.task_type == 'regression':
            # Regression metrics
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("Test MAE", f"{st.session_state.metrics.get('test_mae', 0):.4f}")
            with col2:
                st.metric("Test RMSE", f"{st.session_state.metrics.get('test_rmse', 0):.4f}")
            with col3:
                st.metric("R² Score", f"{st.session_state.metrics.get('r2_score', 0):.4f}")
            with col4:
                st.metric("Train MAE", f"{st.session_state.metrics.get('train_mae', 0):.4f}")
            with col5:
                st.metric("Train MSE", f"{st.session_state.metrics.get('train_mse', 0):.4f}")
        else:
            # Classification metrics
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("Accuracy", f"{st.session_state.metrics.get('test_accuracy', 0):.3f}")
            with col2:
                st.metric("Precision", f"{st.session_state.metrics.get('precision', 0):.3f}")
            with col3:
                st.metric("Recall", f"{st.session_state.metrics.get('recall', 0):.3f}")
            with col4:
                st.metric("F1-Score", f"{st.session_state.metrics.get('f1', 0):.3f}")
            with col5:
                st.metric("Train Acc.", f"{st.session_state.metrics.get('train_accuracy', 0):.3f}")
        
        # Confusion Matrix (only for classification)
        if st.session_state.task_type == 'classification':
            st.subheader("Confusion Matrix")
            
            cm = st.session_state.confusion_mat
            fig = go.Figure(data=go.Heatmap(
                z=cm,
                x=['No Response', 'Response'],
                y=['No Response', 'Response'],
                text=cm,
                texttemplate='%{text}',
                colorscale='Blues'
            ))
            fig.update_layout(
                xaxis_title="Predicted",
                yaxis_title="Actual",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Feature Importance
        st.subheader("Feature Importance")
        
        if hasattr(st.session_state.model, 'feature_importances_'):
            importances = st.session_state.model.feature_importances_
            features = st.session_state.feature_names
            
            importance_df = pd.DataFrame({
                'Feature': features,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(f"⚠️ Feature Importance is not available for {st.session_state.selected_model_type} model.")
        
        # Classification Report (only for classification tasks)
        if st.session_state.task_type == 'classification':
            st.subheader("Detailed Classification Report")
            
            report = classification_report(st.session_state.y_test, st.session_state.y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.write(report_df)


# ==================== TAB 5: PREDICTION ====================
with tab5:
    st.markdown("<h2 class='section-header'>Customer Response Prediction</h2>", unsafe_allow_html=True)
    
    if not st.session_state.model_trained:
        st.warning("Please train a model first before making predictions!")
    else:
        # Model selector for predictions
        st.subheader("Select Model for Prediction")
        available_models = list(st.session_state.all_models.keys()) if st.session_state.all_models else ['No models trained']
        
        pred_model_choice = st.selectbox(
            "Choose a trained model for predictions",
            available_models if available_models != ['No models trained'] else []
        ) if available_models != ['No models trained'] else None
        
        if pred_model_choice and pred_model_choice in st.session_state.all_models:
            selected_pred_model = st.session_state.all_models[pred_model_choice]['model']
            selected_pred_scaler = st.session_state.all_models[pred_model_choice]['scaler']
        else:
            selected_pred_model = st.session_state.model
            selected_pred_scaler = st.session_state.scaler
            pred_model_choice = st.session_state.selected_model_type
        
        st.info(f"📊 **Using model:** {pred_model_choice}")
        st.markdown("---")
        
        if hasattr(st.session_state, 'feature_names') and st.session_state.feature_names and st.session_state.feature_metadata:
            st.write(f"Enter values for the **{len(st.session_state.feature_names)} features** below:")
            
            # Display field information for non-technical users
            with st.expander("📋 Field Guide - What Each Field Means", expanded=False):
                guide_data = []
                for feature in st.session_state.feature_names:
                    metadata = st.session_state.feature_metadata.get(feature, {})
                    if metadata.get('type') == 'numeric':
                        min_val = metadata.get('min', 0)
                        max_val = metadata.get('max', 100)
                        guide_data.append({
                            'Field': feature,
                            'Type': '📊 Number',
                            'Range': f"{float(min_val):.0f} to {float(max_val):.0f}",
                            'Description': f"Enter a value between {float(min_val):.0f} and {float(max_val):.0f}"
                        })
                    else:
                        categories = metadata.get('categories', [])
                        guide_data.append({
                            'Field': feature,
                            'Type': '📋 Choice',
                            'Range': f"{len(categories)} options",
                            'Description': ', '.join(str(c) for c in categories[:3]) + ("..." if len(categories) > 3 else "")
                        })
                guide_df = pd.DataFrame(guide_data)
                st.dataframe(guide_df, use_container_width=True, hide_index=True)
            
            st.markdown("---")
            
            # Create input fields based on feature metadata
            feature_values = {}
            user_friendly_values = {}
            cols = st.columns(min(3, len(st.session_state.feature_names)))
            
            for idx, feature in enumerate(st.session_state.feature_names):
                with cols[idx % len(cols)]:
                    metadata = st.session_state.feature_metadata.get(feature, {})
                    
                    if metadata.get('type') == 'numeric':
                        # For numeric features, use slider with human-readable labels
                        min_val = metadata.get('min', 0)
                        max_val = metadata.get('max', 100)
                        mean_val = metadata.get('mean', (min_val + max_val) / 2)
                        
                        st.write(f"**📊 {feature}**")
                        st.write(f"Range: {float(min_val):.0f} to {float(max_val):.0f}")
                        
                        user_value = st.slider(
                            label=f"Select {feature}",
                            min_value=float(min_val),
                            max_value=float(max_val),
                            value=float(mean_val),
                            step=(max_val - min_val) / 100 if (max_val - min_val) / 100 > 0 else 0.1,
                            key=f"slider_{feature}",
                            label_visibility="collapsed"
                        )
                        st.write(f"**Selected: {user_value:.2f}**")
                        user_friendly_values[feature] = user_value
                        feature_values[feature] = user_value
                    
                    elif metadata.get('type') == 'categorical':
                        # For categorical features, use dropdown with actual values
                        categories = metadata.get('categories', [])
                        
                        st.write(f"**📋 {feature}**")
                        st.write(f"Choose one of: {', '.join(str(c) for c in categories)}")
                        
                        user_value = st.selectbox(
                            label=f"Select {feature}",
                            options=categories,
                            key=f"select_{feature}",
                            label_visibility="collapsed"
                        )
                        st.write(f"**Selected: {user_value}**")
                        user_friendly_values[feature] = user_value
                        # Encode the categorical value
                        le = st.session_state.label_encoders.get(feature)
                        if le:
                            feature_values[feature] = le.transform([user_value])[0]
                        else:
                            feature_values[feature] = user_value
            
            st.markdown("---")
            
            if st.button("Make Prediction", key="predict"):
                try:
                    with st.spinner(f"Making prediction with {pred_model_choice}..."):
                        # Prepare input in correct order (already encoded)
                        input_data = np.array([[feature_values[f] for f in st.session_state.feature_names]])
                        
                        # Scale input using the stored scaler
                        input_scaled = selected_pred_scaler.transform(input_data)
                        
                        # Make prediction
                        prediction = selected_pred_model.predict(input_scaled)[0]
                        
                        # Only call predict_proba if the model supports it (classification models)
                        if hasattr(selected_pred_model, 'predict_proba'):
                            try:
                                probability = selected_pred_model.predict_proba(input_scaled)[0]
                            except:
                                probability = None
                        else:
                            probability = None
                        
                        # Save prediction to database
                        try:
                            save_prediction(
                                age=user_friendly_values.get(st.session_state.feature_names[0], 0),
                                income=user_friendly_values.get(st.session_state.feature_names[1], 0) if len(st.session_state.feature_names) > 1 else 0,
                                previous_purchases=user_friendly_values.get(st.session_state.feature_names[2], 0) if len(st.session_state.feature_names) > 2 else 0,
                                email_opened=str(user_friendly_values.get(st.session_state.feature_names[3], "No")) if len(st.session_state.feature_names) > 3 else "Unknown",
                                website_visits=user_friendly_values.get(st.session_state.feature_names[4], 0) if len(st.session_state.feature_names) > 4 else 0,
                                prediction=prediction,
                                probability=probability
                            )
                        except:
                            pass  # Skip database save if parameters don't match
                        
                        # Display results
                        st.markdown("---")
                        st.subheader("Prediction Result")
                        
                        # Check if this is a regression or classification based on probability
                        is_regression = (probability is None) or (st.session_state.task_type == 'regression')
                        
                        if is_regression:
                            # Regression prediction
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.success(f"✅ Predicted Value: {prediction:.4f}")
                            
                            with col2:
                                st.metric("Prediction", f"{prediction:.4f}")
                            
                            st.info(f"🔢 **Regression Model:** Predicts continuous values")
                        
                        else:
                            # Classification prediction
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                if prediction == 1:
                                    st.success("✅ Positive Prediction")
                                else:
                                    st.error("❌ Negative Prediction")
                            
                            with col2:
                                response_prob = probability[1] * 100 if len(probability) > 1 else probability[0] * 100
                                st.metric("Confidence Score", f"{response_prob:.1f}%")
                            
                            # Probability chart
                            st.subheader("Confidence Breakdown")
                            
                            if len(probability) == 2:
                                prob_data = {
                                    'Outcome': ['Positive', 'Negative'],
                                    'Probability (%)': [probability[1] * 100, probability[0] * 100]
                                }
                            else:
                                prob_data = {
                                    'Outcome': [f'Class {i}' for i in range(len(probability))],
                                    'Probability (%)': [p * 100 for p in probability]
                                }
                            
                            prob_df = pd.DataFrame(prob_data)
                            
                            fig = px.bar(prob_df, x='Outcome', y='Probability (%)', color='Outcome')
                            fig.update_layout(height=400, showlegend=False)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Input summary
                        st.subheader("Your Input Summary")
                        summary_df = pd.DataFrame({
                            'Feature': list(user_friendly_values.keys()),
                            'Your Input': [str(v) for v in user_friendly_values.values()]
                        })
                        st.dataframe(summary_df, use_container_width=True)
                
                except Exception as e:
                    st.error(f"❌ Error making prediction: {str(e)}")
                    st.write(f"Details: {str(e)}")
        else:
            st.warning("⚠️ No model trained yet. Please train a model first in the Model Training tab.")


# ==================== TAB 6: HISTORY & DATABASE ====================
with tab6:
    st.markdown("<h2 class='section-header'>Prediction History & Database Management</h2>", unsafe_allow_html=True)
    
    # Database Statistics
    st.subheader("Database Statistics")
    stats = get_statistics()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Predictions", stats['total_predictions'])
    with col2:
        st.metric("Total Model Trainings", stats['total_trainings'])
    with col3:
        st.metric("Positive Response Rate", f"{stats['response_rate']:.1f}%")
    
    st.markdown("---")
    
    # Create subtabs for different views
    subtab1, subtab2, subtab3, subtab4 = st.tabs(["Prediction History", "Model Training History", "Model Files", "Export & Manage"])
    
    # SUBTAB 1: Prediction History
    with subtab1:
        st.subheader("Prediction History")
        
        pred_history = get_prediction_history()
        
        if pred_history.empty:
            st.info("No predictions made yet. Make some predictions to see history here.")
        else:
            st.write(f"Total predictions: {len(pred_history)}")
            
            # Display with pagination
            if len(pred_history) > 10:
                page = st.slider("Page", 1, (len(pred_history) + 9) // 10, 1)
                start_idx = (page - 1) * 10
                end_idx = start_idx + 10
                st.dataframe(pred_history.iloc[start_idx:end_idx], use_container_width=True)
            else:
                st.dataframe(pred_history, use_container_width=True)
    
    # SUBTAB 2: Model Training History
    with subtab2:
        st.subheader("Model Training History")
        
        metrics_history = get_model_metrics_history()
        
        if metrics_history.empty:
            st.info("No models trained yet. Train a model to see history here.")
        else:
            st.write(f"Total trainings: {len(metrics_history)}")
            st.dataframe(metrics_history, use_container_width=True)
    
    # SUBTAB 3: Saved Models
    with subtab3:
        st.subheader("Saved Model Files")
        
        model_history = get_model_history()
        
        if model_history.empty:
            st.info("No models saved yet.")
        else:
            st.write(f"Total saved models: {len(model_history)}")
            st.dataframe(model_history, use_container_width=True)
            
            st.subheader("Load a Previously Trained Model")
            
            if not model_history.empty:
                selected_model = st.selectbox(
                    "Select a model to load",
                    range(len(model_history)),
                    format_func=lambda x: f"{model_history.iloc[x]['Timestamp']} - {model_history.iloc[x]['Model Name']}"
                )
                
                if st.button("Load Selected Model", key="load_model"):
                    try:
                        selected_row = model_history.iloc[selected_model]
                        model_file = selected_row['Model File']
                        scaler_file = selected_row['Scaler File']
                        
                        model, scaler = load_model(model_file, scaler_file)
                        
                        st.session_state.model = model
                        st.session_state.scaler = scaler
                        st.session_state.model_trained = True
                        
                        # Set dummy values for feature names
                        st.session_state.feature_names = ['Age', 'Income', 'Previous_Purchases', 'Email_Opened_Encoded', 'Website_Visits']
                        
                        st.success(f"Model loaded successfully!")
                        st.info("You can now use this model for predictions in the 'Make Prediction' tab.")
                    
                    except Exception as e:
                        st.error(f"Error loading model: {str(e)}")
    
    # SUBTAB 4: Export & Manage
    with subtab4:
        st.subheader("Export Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Export Predictions to CSV"):
                pred_history = get_prediction_history()
                if pred_history.empty:
                    st.warning("No predictions to export.")
                else:
                    csv = pred_history.to_csv(index=False)
                    st.download_button(
                        label="Download Predictions CSV",
                        data=csv,
                        file_name="predictions_export.csv",
                        mime="text/csv"
                    )
                    st.success(f"Ready to download {len(pred_history)} predictions!")
        
        with col2:
            if st.button("Export Metrics to CSV"):
                metrics_history = get_model_metrics_history()
                if metrics_history.empty:
                    st.warning("No metrics to export.")
                else:
                    csv = metrics_history.to_csv(index=False)
                    st.download_button(
                        label="Download Metrics CSV",
                        data=csv,
                        file_name="metrics_export.csv",
                        mime="text/csv"
                    )
                    st.success(f"Ready to download {len(metrics_history)} metric entries!")
        
        st.markdown("---")
        st.subheader("Database Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🗑️ Clear All Predictions", key="clear_pred"):
                st.warning("⚠️ This will delete all predictions from the database.")
                if st.button("Confirm - Delete All Predictions", key="confirm_clear_pred"):
                    try:
                        conn = __import__('sqlite3').connect('predictions.db')
                        cursor = conn.cursor()
                        cursor.execute('DELETE FROM predictions')
                        conn.commit()
                        conn.close()
                        st.success("All predictions cleared!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error clearing predictions: {str(e)}")
        
        with col2:
            if st.button("🗑️ Clear All Metrics", key="clear_metrics"):
                st.warning("⚠️ This will delete all model metrics from the database.")
                if st.button("Confirm - Delete All Metrics", key="confirm_clear_metrics"):
                    try:
                        conn = __import__('sqlite3').connect('predictions.db')
                        cursor = conn.cursor()
                        cursor.execute('DELETE FROM model_metrics')
                        conn.commit()
                        conn.close()
                        st.success("All metrics cleared!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error clearing metrics: {str(e)}")
        
        st.info("💡 Tip: Your predictions and model metrics are automatically saved to the SQLite database when you make predictions or train models.")

st.markdown("---")
st.write("Built with  using Streamlit | Marketing Campaign Response Predictor v1.0")