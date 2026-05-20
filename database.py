import sqlite3
import pandas as pd
from datetime import datetime
import os
import json
import pickle

DB_PATH = "predictions.db"

def init_database():
    """Initialize the SQLite database with required tables"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create predictions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            age INTEGER NOT NULL,
            income INTEGER NOT NULL,
            previous_purchases INTEGER NOT NULL,
            email_opened TEXT NOT NULL,
            website_visits INTEGER NOT NULL,
            prediction INTEGER NOT NULL,
            probability_no_response REAL NOT NULL,
            probability_response REAL NOT NULL
        )
    ''')
    
    # Create model metrics table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS model_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            train_accuracy REAL NOT NULL,
            test_accuracy REAL NOT NULL,
            precision REAL NOT NULL,
            recall REAL NOT NULL,
            f1_score REAL NOT NULL,
            model_name TEXT,
            data_rows INTEGER
        )
    ''')
    
    # Create model history table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS model_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            model_filename TEXT NOT NULL,
            scaler_filename TEXT NOT NULL,
            accuracy REAL NOT NULL,
            model_name TEXT
        )
    ''')
    
    conn.commit()
    conn.close()

def save_prediction(age, income, previous_purchases, email_opened, website_visits, prediction, probability):
    """Save a prediction to the database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    prob_no_response = probability[0]
    prob_response = probability[1]
    
    cursor.execute('''
        INSERT INTO predictions 
        (timestamp, age, income, previous_purchases, email_opened, website_visits, prediction, probability_no_response, probability_response)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (timestamp, age, income, previous_purchases, email_opened, website_visits, prediction, prob_no_response, prob_response))
    
    conn.commit()
    conn.close()

def get_prediction_history():
    """Retrieve all predictions from the database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM predictions ORDER BY timestamp DESC')
    rows = cursor.fetchall()
    
    conn.close()
    
    if not rows:
        return pd.DataFrame()
    
    columns = ['ID', 'Timestamp', 'Age', 'Income', 'Prev. Purchases', 'Email Opened', 'Website Visits', 'Prediction', 'Prob. No Response', 'Prob. Response']
    df = pd.DataFrame(rows, columns=columns)
    
    # Format prediction
    df['Prediction'] = df['Prediction'].apply(lambda x: 'Yes' if x == 1 else 'No')
    
    # Format probabilities as percentages
    df['Prob. Response'] = df['Prob. Response'].apply(lambda x: f"{x*100:.1f}%")
    df['Prob. No Response'] = df['Prob. No Response'].apply(lambda x: f"{x*100:.1f}%")
    
    return df

def save_model_metrics(metrics, model_name="RandomForest", data_rows=0):
    """Save model metrics to the database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    cursor.execute('''
        INSERT INTO model_metrics 
        (timestamp, train_accuracy, test_accuracy, precision, recall, f1_score, model_name, data_rows)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (timestamp, metrics['train_accuracy'], metrics['test_accuracy'], metrics['precision'], 
          metrics['recall'], metrics['f1'], model_name, data_rows))
    
    conn.commit()
    conn.close()

def get_model_metrics_history():
    """Retrieve all model metrics from the database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM model_metrics ORDER BY timestamp DESC')
    rows = cursor.fetchall()
    
    conn.close()
    
    if not rows:
        return pd.DataFrame()
    
    columns = ['ID', 'Timestamp', 'Train Accuracy', 'Test Accuracy', 'Precision', 'Recall', 'F1-Score', 'Model', 'Data Rows']
    df = pd.DataFrame(rows, columns=columns)
    
    # Format as percentages
    for col in ['Train Accuracy', 'Test Accuracy', 'Precision', 'Recall', 'F1-Score']:
        df[col] = df[col].apply(lambda x: f"{x*100:.2f}%")
    
    return df

def save_model(model, scaler, accuracy, model_name="model"):
    """Save trained model and scaler to disk"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"models/model_{model_name}_{timestamp}.pkl"
    scaler_filename = f"models/scaler_{model_name}_{timestamp}.pkl"
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Save model and scaler
    with open(model_filename, 'wb') as f:
        pickle.dump(model, f)
    
    with open(scaler_filename, 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save to database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    timestamp_db = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute('''
        INSERT INTO model_history 
        (timestamp, model_filename, scaler_filename, accuracy, model_name)
        VALUES (?, ?, ?, ?, ?)
    ''', (timestamp_db, model_filename, scaler_filename, accuracy, model_name))
    
    conn.commit()
    conn.close()
    
    return model_filename, scaler_filename

def load_model(model_filename, scaler_filename):
    """Load a saved model and scaler from disk"""
    with open(model_filename, 'rb') as f:
        model = pickle.load(f)
    
    with open(scaler_filename, 'rb') as f:
        scaler = pickle.load(f)
    
    return model, scaler

def get_model_history():
    """Retrieve all saved models from the database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM model_history ORDER BY timestamp DESC')
    rows = cursor.fetchall()
    
    conn.close()
    
    if not rows:
        return pd.DataFrame()
    
    columns = ['ID', 'Timestamp', 'Model File', 'Scaler File', 'Accuracy', 'Model Name']
    df = pd.DataFrame(rows, columns=columns)
    
    # Format accuracy as percentage
    df['Accuracy'] = df['Accuracy'].apply(lambda x: f"{x*100:.2f}%")
    
    return df

def get_statistics():
    """Get database statistics"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('SELECT COUNT(*) FROM predictions')
    total_predictions = cursor.fetchone()[0]
    
    cursor.execute('SELECT COUNT(*) FROM model_metrics')
    total_trainings = cursor.fetchone()[0]
    
    if total_predictions > 0:
        cursor.execute('SELECT SUM(prediction) FROM predictions')
        positive_predictions = cursor.fetchone()[0] or 0
        response_rate = (positive_predictions / total_predictions) * 100
    else:
        response_rate = 0
    
    conn.close()
    
    return {
        'total_predictions': total_predictions,
        'total_trainings': total_trainings,
        'response_rate': response_rate
    }

def clear_all_data():
    """Clear all data from the database (for reset)"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('DELETE FROM predictions')
    cursor.execute('DELETE FROM model_metrics')
    cursor.execute('DELETE FROM model_history')
    
    conn.commit()
    conn.close()

def export_predictions_csv(filename="predictions_export.csv"):
    """Export all predictions to CSV"""
    df = get_prediction_history()
    if not df.empty:
        df.to_csv(filename, index=False)
        return filename
    return None

def export_metrics_csv(filename="metrics_export.csv"):
    """Export all model metrics to CSV"""
    df = get_model_metrics_history()
    if not df.empty:
        df.to_csv(filename, index=False)
        return filename
    return None
