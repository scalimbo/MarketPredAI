import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
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

# Custom CSS for Light and Dark Modes
def apply_theme(theme_mode):
    if theme_mode == 'dark':
        st.markdown("""
            <style>
            :root {
                --primary-color: #00D9FF;
                --secondary-color: #1f77b4;
                --bg-color: #0e1117;
                --text-color: #e0e0e0;
                --card-bg: #161b22;
                --border-color: #30363d;
            }
            .main-header {
                font-size: 3em;
                color: #00D9FF;
                text-align: center;
                margin-bottom: 10px;
                text-shadow: 0 0 10px rgba(0, 217, 255, 0.3);
            }
            .section-header {
                font-size: 1.8em;
                color: #00D9FF;
                margin-top: 20px;
                margin-bottom: 10px;
            }
            .metric-card {
                background: linear-gradient(135deg, #161b22 0%, #0d1117 100%);
                border: 1px solid #30363d;
                border-radius: 8px;
                padding: 15px;
                color: #e0e0e0;
            }
            </style>
        """, unsafe_allow_html=True)
    else:  # light mode
        st.markdown("""
            <style>
            :root {
                --primary-color: #1f77b4;
                --secondary-color: #ff7f0e;
                --bg-color: #ffffff;
                --text-color: #333333;
                --card-bg: #f8f9fa;
                --border-color: #e0e0e0;
            }
            .main-header {
                font-size: 3em;
                color: #1f77b4;
                text-align: center;
                margin-bottom: 10px;
                text-shadow: 0 0 5px rgba(31, 119, 180, 0.1);
            }
            .section-header {
                font-size: 1.8em;
                color: #1f77b4;
                margin-top: 20px;
                margin-bottom: 10px;
            }
            .metric-card {
                background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
                border: 1px solid #e0e0e0;
                border-radius: 8px;
                padding: 15px;
                color: #333333;
            }
            </style>
        """, unsafe_allow_html=True)

# Apply the current theme
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

# Sidebar Theme Toggle
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
        
        if theme_option == "🌙 Dark":
            st.session_state.theme = 'dark'
        else:
            st.session_state.theme = 'light'
    
    st.markdown("---")

# Main title
st.markdown("<h1 class='main-header'>Marketing Campaign Response Predictor</h1>", unsafe_allow_html=True)
st.markdown("---")

# Create tabs for different sections
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Data Management", "Model Training", "Results", "Make Prediction", "History & Database"])

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
        
        # PREPROCESSING
        with col1:
            st.subheader("Step 1: Data Preprocessing")
            
            if st.button("Preprocess Data", key="preprocess"):
                try:
                    with st.spinner("Preprocessing data..."):
                        df = st.session_state.data.copy()
                        
                        # Check required columns
                        required_cols = ['Age', 'Income', 'Previous_Purchases', 'Email_Opened', 'Website_Visits']
                        missing_cols = [col for col in required_cols if col not in df.columns]
                        
                        if missing_cols:
                            st.error(f"Missing columns: {', '.join(missing_cols)}")
                        else:
                            # Encode categorical variables
                            df['Email_Opened_Encoded'] = df['Email_Opened'].map({'Yes': 1, 'No': 0})
                            
                            # Check if Email_Opened_Encoded has NaN values
                            if df['Email_Opened_Encoded'].isnull().any():
                                st.warning("Found unexpected values in Email_Opened column. Converting...")
                                df['Email_Opened_Encoded'] = df['Email_Opened'].astype(str).str.lower().map({'yes': 1, 'no': 0, '1': 1, '0': 0})
                            
                            # Drop original categorical column
                            df = df.drop('Email_Opened', axis=1)
                            
                            # Handle any remaining missing values
                            df = df.fillna(df.mean(numeric_only=True))
                            
                            st.session_state.preprocessed_data = df
                            st.success("Data preprocessed successfully!")
                            
                            st.write("Processed data shape:", df.shape)
                            st.write(df.head())
                
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
                        with st.spinner("Training model..."):
                            df = st.session_state.preprocessed_data.copy()
                            
                            # Prepare features and target
                            if 'Response' in df.columns:
                                X = df.drop('Response', axis=1)
                                y = df['Response']
                            else:
                                st.error("'Response' column not found in dataset")
                                st.stop()
                            
                            # Split data
                            X_train, X_test, y_train, y_test = train_test_split(
                                X, y, test_size=0.2, random_state=42, stratify=y
                            )
                            
                            # Scale features
                            scaler = StandardScaler()
                            X_train_scaled = scaler.fit_transform(X_train)
                            X_test_scaled = scaler.transform(X_test)
                            
                            # Train model
                            model = RandomForestClassifier(n_estimators=100, random_state=42)
                            model.fit(X_train_scaled, y_train)
                            
                            # Make predictions
                            y_pred_train = model.predict(X_train_scaled)
                            y_pred_test = model.predict(X_test_scaled)
                            
                            # Store in session
                            st.session_state.model = model
                            st.session_state.scaler = scaler
                            st.session_state.model_trained = True
                            
                            # Calculate metrics
                            metrics = {
                                'train_accuracy': accuracy_score(y_train, y_pred_train),
                                'test_accuracy': accuracy_score(y_test, y_pred_test),
                                'precision': precision_score(y_test, y_pred_test, average='weighted'),
                                'recall': recall_score(y_test, y_pred_test, average='weighted'),
                                'f1': f1_score(y_test, y_pred_test, average='weighted')
                            }
                            
                            st.session_state.metrics = metrics
                            st.session_state.confusion_mat = confusion_matrix(y_test, y_pred_test)
                            st.session_state.y_test = y_test
                            st.session_state.y_pred = y_pred_test
                            st.session_state.feature_names = X.columns.tolist()
                            
                            # Save metrics to database
                            save_model_metrics(metrics, model_name="RandomForest", data_rows=len(df))
                            
                            # Save model and scaler
                            save_model(model, scaler, metrics['test_accuracy'], model_name="RandomForest")
                            
                            st.success("Model trained successfully!")
                    
                    except Exception as e:
                        st.error(f"Error during model training: {str(e)}")


# ==================== TAB 3: RESULTS ====================
with tab3:
    st.markdown("<h2 class='section-header'>Model Performance Metrics</h2>", unsafe_allow_html=True)
    
    if not st.session_state.model_trained:
        st.info("Train the model first to see results!")
    else:
        # Metrics display
        st.subheader("Performance Metrics")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Accuracy", f"{st.session_state.metrics['test_accuracy']:.3f}")
        with col2:
            st.metric("Precision", f"{st.session_state.metrics['precision']:.3f}")
        with col3:
            st.metric("Recall", f"{st.session_state.metrics['recall']:.3f}")
        with col4:
            st.metric("F1-Score", f"{st.session_state.metrics['f1']:.3f}")
        with col5:
            st.metric("Train Acc.", f"{st.session_state.metrics['train_accuracy']:.3f}")
        
        # Confusion Matrix
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
        
        # Classification Report
        st.subheader("Detailed Classification Report")
        
        report = classification_report(st.session_state.y_test, st.session_state.y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.write(report_df)


# ==================== TAB 4: PREDICTION ====================
with tab4:
    st.markdown("<h2 class='section-header'>Customer Response Prediction</h2>", unsafe_allow_html=True)
    
    if not st.session_state.model_trained:
        st.warning("Please train the model first before making predictions!")
    else:
        st.write("Enter customer details below to predict marketing campaign response:")
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.slider("Age", min_value=18, max_value=100, value=35, step=1)
            income = st.number_input("Annual Income ($)", min_value=0, value=50000, step=1000)
            previous_purchases = st.slider("Previous Purchases", min_value=0, max_value=100, value=5, step=1)
        
        with col2:
            email_opened = st.radio("Email Opened?", ["Yes", "No"])
            website_visits = st.slider("Website Visits (Last 30 Days)", min_value=0, max_value=100, value=10, step=1)
        
        st.markdown("---")
        
        if st.button("Make Prediction", key="predict"):
            try:
                with st.spinner("Making prediction..."):
                    # Prepare input
                    email_encoded = 1 if email_opened == "Yes" else 0
                    
                    input_data = np.array([[
                        age,
                        income,
                        previous_purchases,
                        email_encoded,
                        website_visits
                    ]])
                    
                    # Scale input
                    input_scaled = st.session_state.scaler.transform(input_data)
                    
                    # Make prediction
                    prediction = st.session_state.model.predict(input_scaled)[0]
                    probability = st.session_state.model.predict_proba(input_scaled)[0]
                    
                    # Save prediction to database
                    save_prediction(age, income, previous_purchases, email_opened, website_visits, prediction, probability)
                    
                    # Display results
                    st.markdown("---")
                    st.subheader("Prediction Result")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if prediction == 1:
                            st.success("Customer WILL respond to the campaign")
                        else:
                            st.error("Customer will NOT respond to the campaign")
                    
                    with col2:
                        response_prob = probability[1] * 100
                        no_response_prob = probability[0] * 100
                        st.metric("Response Probability", f"{response_prob:.1f}%")
                    
                    # Probability chart
                    st.subheader("Confidence Breakdown")
                    
                    prob_data = {
                        'Outcome': ['Will Respond', 'Will Not Respond'],
                        'Probability': [response_prob, no_response_prob]
                    }
                    prob_df = pd.DataFrame(prob_data)
                    
                    fig = px.bar(prob_df, x='Outcome', y='Probability', color='Outcome',
                                color_discrete_map={'Will Respond': '#00CC96', 'Will Not Respond': '#EF553B'})
                    fig.update_layout(height=400, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Input summary
                    st.subheader("Input Summary")
                    summary_df = pd.DataFrame({
                        'Feature': ['Age', 'Income', 'Previous Purchases', 'Email Opened', 'Website Visits'],
                        'Value': [age, f"${income:,}", previous_purchases, email_opened, website_visits]
                    })
                    st.write(summary_df)
            
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")


# ==================== TAB 5: HISTORY & DATABASE ====================
with tab5:
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