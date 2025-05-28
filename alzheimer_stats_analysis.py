import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import joblib
import warnings
warnings.filterwarnings('ignore')

# Define directories
BASE_DIR = "Fastsurfer_stats_Alz"
DATA_DIR = "Fastsurfer_stats"
PROCESSED_DIR = os.path.join(BASE_DIR, "processed_data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
CM_DIR = os.path.join(RESULTS_DIR, "confusion_matrices")

# Create directories if they don't exist
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CM_DIR, exist_ok=True)

# 1. Extract and Convert Stats to CSV
def parse_stats_file(file_path):
    """Parse a single .stats file and return a dictionary of features."""
    data = {}
    subject_id = os.path.basename(file_path).replace('.stats', '')
    data['subject_id'] = subject_id
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Find where the data starts (after the headers)
    data_start = 0
    for i, line in enumerate(lines):
        if line.startswith('# ColHeaders'):
            data_start = i + 1
            break
    
    # Extract the data
    for line in lines[data_start:]:
        if line.strip() and not line.startswith('#'):
            parts = line.strip().split()
            if len(parts) >= 5:  # Ensure we have enough columns
                index = int(parts[0])
                struct_name = parts[4].replace('-', '_')  # Clean up structure names
                
                # Extract volume, mean intensity, and std dev if available
                if len(parts) >= 4:
                    volume = float(parts[3])
                    data[f"{struct_name}_volume"] = volume
                
                if len(parts) >= 6:
                    norm_mean = float(parts[5])
                    data[f"{struct_name}_mean"] = norm_mean
                
                if len(parts) >= 7:
                    norm_std = float(parts[6])
                    data[f"{struct_name}_std"] = norm_std
    
    return data

def process_class_directory(class_dir, label):
    """Process all .stats files in a class directory."""
    stats_files = glob.glob(os.path.join(class_dir, "*.stats"))
    all_data = []
    
    print(f"Processing {len(stats_files)} files in {class_dir}...")
    
    for file_path in stats_files:
        try:
            data = parse_stats_file(file_path)
            data['label'] = label
            all_data.append(data)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    if all_data:
        df = pd.DataFrame(all_data)
        return df
    else:
        return pd.DataFrame()

# 2. Combine All CSVs and Add Labels
def combine_all_data():
    """Combine data from all class directories and add labels."""
    base_dir = DATA_DIR
    
    # Process each class directory
    ad_data = process_class_directory(os.path.join(base_dir, "AD"), "AD")
    cn_data = process_class_directory(os.path.join(base_dir, "CN"), "CN")
    mci_data = process_class_directory(os.path.join(base_dir, "MCI"), "MCI")
    
    # Combine all data
    combined_data = pd.concat([ad_data, cn_data, mci_data], ignore_index=True)
    
    # Fill missing values with 0
    combined_data = combined_data.fillna(0)
    
    # Save the combined raw data
    combined_data.to_csv(os.path.join(PROCESSED_DIR, "combined_raw.csv"), index=False)
    
    print(f"Combined data shape: {combined_data.shape}")
    print(f"Class distribution: {combined_data['label'].value_counts()}")
    
    return combined_data

# 3. Split the Data
def prepare_data(data, random_state=42):
    """Prepare data for model training by splitting and normalizing."""
    # Make a copy to avoid modifying the original DataFrame
    df = data.copy()
    
    # Ensure label column exists
    if 'label' not in df.columns:
        raise ValueError("Label column 'label' not found in the data")
    
    # Separate features and target
    X = df.drop(['label', 'subject_id'], axis=1, errors='ignore')
    y = df['label']
    
    # Keep track of feature names for later use
    feature_names = X.columns.tolist()
    
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data into train, validation, and test sets (70%, 15%, 15%)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_scaled, y, test_size=0.15, random_state=random_state, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.1765, random_state=random_state, stratify=y_train_val
    )  # 0.1765 * 0.85 = 0.15
    
    # Create DataFrames for saving to CSV
    train_df = pd.DataFrame(X_train, columns=feature_names)
    train_df['label'] = y_train.values
    
    val_df = pd.DataFrame(X_val, columns=feature_names)
    val_df['label'] = y_val.values
    
    test_df = pd.DataFrame(X_test, columns=feature_names)
    test_df['label'] = y_test.values
    
    # Save to CSV
    train_df.to_csv(os.path.join(PROCESSED_DIR, "train.csv"), index=False)
    val_df.to_csv(os.path.join(PROCESSED_DIR, "val.csv"), index=False)
    test_df.to_csv(os.path.join(PROCESSED_DIR, "test.csv"), index=False)
    
    print("Data split and saved to processed_data/")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, scaler, feature_names

# 4. Define Tabular CNN Model
class TabularCNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(TabularCNN, self).__init__()
        
        # Reshape input to a 2D grid for CNNs
        self.reshape_size = int(np.sqrt(input_size)) + 1
        self.pad_size = self.reshape_size**2 - input_size
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        # Pad the input to make it a perfect square
        x = torch.nn.functional.pad(x, (0, self.pad_size))
        
        # Reshape to 2D image
        x = x.view(-1, 1, self.reshape_size, self.reshape_size)
        
        # Apply convolutions
        x = self.conv_layers(x)
        
        # Apply fully connected layers
        x = self.fc_layers(x)
        
        return x

# Train CNN model
def train_cnn_model(X_train, y_train, X_val, y_val, num_epochs=50, batch_size=32):
    """Train a CNN model for tabular data."""
    # Convert string labels to numerical
    label_encoder = LabelEncoder()
    y_train_numeric = label_encoder.fit_transform(y_train)
    y_val_numeric = label_encoder.transform(y_val)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train_numeric)
    
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.LongTensor(y_val_numeric)
    
    # Create DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    input_size = X_train.shape[1]
    num_classes = len(label_encoder.classes_)
    model = TabularCNN(input_size, num_classes)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Track best model
    best_val_acc = 0
    best_model_state = None
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_preds = []
        val_true = []
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X)
                _, predicted = torch.max(outputs, 1)
                val_preds.extend(predicted.cpu().numpy())
                val_true.extend(batch_y.cpu().numpy())
        
        val_acc = accuracy_score(val_true, val_preds)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss/len(train_loader):.4f}, Val Acc: {val_acc:.4f}')
        
        # Save the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()
    
    # Save the best model
    model_path = os.path.join(MODELS_DIR, 'cnn_model.pt')
    torch.save({
        'model_state_dict': best_model_state,
        'label_encoder': label_encoder,
        'input_size': input_size,
        'num_classes': num_classes,
        'val_acc': best_val_acc
    }, model_path)
    
    # Load the best model state for final evaluation
    model.load_state_dict(best_model_state)
    
    # Final predictions on validation set
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_tensor)
        _, val_preds = torch.max(val_outputs, 1)
        val_preds = val_preds.cpu().numpy()
    
    # Convert numeric predictions back to original labels
    val_preds_labels = label_encoder.inverse_transform(val_preds)
    
    print(f"CNN model trained and saved to {model_path}")
    return model, val_preds_labels, label_encoder, best_val_acc

# Train and evaluate traditional models
def train_traditional_models(X_train, y_train, X_val, y_val, X_test, y_test, feature_names):
    """Train and evaluate traditional machine learning models."""
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='mlogloss'),
        'SVM': SVC(probability=True, random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
    }
    
    results = []
    val_predictions = {}
    
    # Encode labels for traditional models if they're strings
    le = LabelEncoder()
    if isinstance(y_train.iloc[0], str):
        y_train_enc = le.fit_transform(y_train)
        y_val_enc = le.transform(y_val)
        y_test_enc = le.transform(y_test)
        class_names = le.classes_
    else:
        y_train_enc = y_train
        y_val_enc = y_val
        y_test_enc = y_test
        class_names = np.unique(y_train)
    
    # Train and evaluate each model
    for name, model in models.items():
        print(f"Training {name}...")
        
        try:
            # Train the model
            model.fit(X_train, y_train_enc)
            
            # Make predictions
            if hasattr(model, 'predict_proba'):
                y_val_proba = model.predict_proba(X_val)
            
            y_val_pred_enc = model.predict(X_val)
            
            # Convert encoded predictions back to original labels
            if isinstance(y_train.iloc[0], str):
                y_val_pred = le.inverse_transform(y_val_pred_enc)
            else:
                y_val_pred = y_val_pred_enc
            
            val_predictions[name] = y_val_pred
            
            # Compute metrics
            accuracy = accuracy_score(y_val, y_val_pred)
            precision = precision_score(y_val, y_val_pred, average='weighted')
            recall = recall_score(y_val, y_val_pred, average='weighted')
            f1 = f1_score(y_val, y_val_pred, average='weighted')
            
            results.append({
                'Model': name,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1 Score': f1
            })
            
            # Save model
            model_path = os.path.join(MODELS_DIR, f"{name.replace(' ', '_')}.joblib")
            joblib.dump(model, model_path)
            print(f"  Saved model to {model_path}")
            
            # Create and save confusion matrix
            cm = confusion_matrix(y_val, y_val_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=class_names, yticklabels=class_names)
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title(f'Confusion Matrix - {name}')
            plt.tight_layout()
            cm_path = os.path.join(CM_DIR, f"{name.replace(' ', '_')}_cm.png")
            plt.savefig(cm_path)
            plt.close()
            print(f"  Saved confusion matrix to {cm_path}")
            
            # Feature importance (for models that support it)
            if name == 'Random Forest':
                feature_importance = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': model.feature_importances_
                })
                feature_importance = feature_importance.sort_values('Importance', ascending=False).head(20)
                
                plt.figure(figsize=(10, 8))
                sns.barplot(x='Importance', y='Feature', data=feature_importance)
                plt.title('Top 20 Feature Importances (Random Forest)')
                plt.tight_layout()
                importance_path = os.path.join(RESULTS_DIR, 'feature_importance.png')
                plt.savefig(importance_path)
                plt.close()
                print(f"  Saved feature importance plot to {importance_path}")
        
        except Exception as e:
            print(f"  Error training {name}: {e}")
    
    # Save metrics to CSV
    results_df = pd.DataFrame(results)
    metrics_path = os.path.join(RESULTS_DIR, 'traditional_models_metrics.csv')
    results_df.to_csv(metrics_path, index=False)
    print(f"Saved traditional model metrics to {metrics_path}")
    
    return results_df, val_predictions, models, le

# Combine CNN with traditional models and analyze performance
def analyze_all_models(trad_results_df, cnn_val_acc, cnn_val_preds, y_val, val_predictions, X_val, class_names):
    """Combine CNN results with traditional models and analyze overall performance."""
    print("Analyzing CNN performance...")
    # Add CNN to results
    cnn_accuracy = accuracy_score(y_val, cnn_val_preds)
    cnn_precision = precision_score(y_val, cnn_val_preds, average='weighted')
    cnn_recall = recall_score(y_val, cnn_val_preds, average='weighted')
    cnn_f1 = f1_score(y_val, cnn_val_preds, average='weighted')
    
    cnn_results = pd.DataFrame([{
        'Model': 'CNN',
        'Accuracy': cnn_accuracy,
        'Precision': cnn_precision,
        'Recall': cnn_recall,
        'F1 Score': cnn_f1
    }])
    
    # Print CNN metrics
    print(f"  CNN Accuracy: {cnn_accuracy:.4f}")
    print(f"  CNN Precision: {cnn_precision:.4f}")
    print(f"  CNN Recall: {cnn_recall:.4f}")
    print(f"  CNN F1 Score: {cnn_f1:.4f}")
    
    # Combine results using concat instead of append
    all_results_df = pd.concat([trad_results_df, cnn_results], ignore_index=True)
    metrics_path = os.path.join(RESULTS_DIR, 'all_models_metrics.csv')
    all_results_df.to_csv(metrics_path, index=False)
    print(f"Saved combined metrics to {metrics_path}")
    
    # Add CNN predictions to val_predictions
    val_predictions['CNN'] = cnn_val_preds
    
    # Create CNN confusion matrix
    try:
        cm = confusion_matrix(y_val, cnn_val_preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix - CNN')
        plt.tight_layout()
        cm_path = os.path.join(CM_DIR, 'CNN_cm.png')
        plt.savefig(cm_path)
        plt.close()
        print(f"  Saved CNN confusion matrix to {cm_path}")
    except Exception as e:
        print(f"  Error creating CNN confusion matrix: {e}")
    
    # Create comparison plots
    print("Creating model comparison visualizations...")
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    
    for metric in metrics:
        try:
            plt.figure(figsize=(12, 6))
            sns.barplot(x='Model', y=metric, data=all_results_df)
            plt.title(f'{metric} Comparison')
            plt.xticks(rotation=45)
            plt.tight_layout()
            metric_path = os.path.join(RESULTS_DIR, f"{metric.lower().replace(' ', '_')}_comparison.png")
            plt.savefig(metric_path)
            plt.close()
            print(f"  Saved {metric} comparison to {metric_path}")
        except Exception as e:
            print(f"  Error creating {metric} comparison plot: {e}")
    
    # Combined metrics plot
    try:
        plt.figure(figsize=(14, 8))
        results_melted = pd.melt(all_results_df, id_vars=['Model'], value_vars=metrics, 
                              var_name='Metric', value_name='Score')
        sns.barplot(x='Model', y='Score', hue='Metric', data=results_melted)
        plt.title('Model Performance Comparison')
        plt.xticks(rotation=45)
        plt.tight_layout()
        comparison_path = os.path.join(RESULTS_DIR, 'model_performance_comparison.png')
        plt.savefig(comparison_path)
        plt.close()
        print(f"  Saved overall model comparison to {comparison_path}")
    except Exception as e:
        print(f"  Error creating overall model comparison plot: {e}")
    
    return all_results_df

# Save the best model
def save_best_model(all_results_df, models, cnn_model, label_encoder, X_test, y_test, scaler):
    """Find and save the best model based on F1 score."""
    # Get the best model based on F1 score
    best_row = all_results_df.loc[all_results_df['F1 Score'].idxmax()]
    best_model_name = best_row['Model']
    
    print(f"Best model: {best_model_name} with F1 Score: {best_row['F1 Score']:.4f}")
    
    # Save best model info
    best_model_info = {
        'model_name': best_model_name,
        'metrics': {
            'accuracy': best_row['Accuracy'],
            'precision': best_row['Precision'],
            'recall': best_row['Recall'],
            'f1_score': best_row['F1 Score']
        }
    }
    
    # Save best model and metadata
    if best_model_name == 'CNN':
        # CNN model is already saved during training
        # Just update the metadata to indicate it's the best model
        cnn_data = torch.load(os.path.join(MODELS_DIR, 'cnn_model.pt'))
        cnn_data['is_best_model'] = True
        cnn_data['best_metrics'] = best_model_info['metrics']
        torch.save(cnn_data, os.path.join(MODELS_DIR, 'best_model.pt'))
        
        # Final evaluation on test set
        input_size = cnn_data['input_size']
        num_classes = cnn_data['num_classes']
        model = TabularCNN(input_size, num_classes)
        model.load_state_dict(cnn_data['model_state_dict'])
        
        # Make predictions on test set
        X_test_tensor = torch.FloatTensor(X_test)
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_tensor)
            _, test_preds = torch.max(test_outputs, 1)
            test_preds = test_preds.cpu().numpy()
        
        test_preds_labels = label_encoder.inverse_transform(test_preds)
    else:
        # Load traditional model
        model = joblib.load(os.path.join(MODELS_DIR, f"{best_model_name.replace(' ', '_')}.joblib"))
        
        # Make predictions on test set
        test_preds = model.predict(X_test)
        
        # Convert to original labels if needed
        if hasattr(label_encoder, 'inverse_transform'):
            test_preds_labels = label_encoder.inverse_transform(test_preds)
        else:
            test_preds_labels = test_preds
        
        # Save as best model with metadata
        joblib.dump({
            'model': model,
            'scaler': scaler,
            'label_encoder': label_encoder if hasattr(label_encoder, 'inverse_transform') else None,
            'is_best_model': True,
            'metrics': best_model_info['metrics']
        }, os.path.join(MODELS_DIR, 'best_model.joblib'))
    
    # Final test metrics
    test_accuracy = accuracy_score(y_test, test_preds_labels)
    test_precision = precision_score(y_test, test_preds_labels, average='weighted')
    test_recall = recall_score(y_test, test_preds_labels, average='weighted')
    test_f1 = f1_score(y_test, test_preds_labels, average='weighted')
    
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test F1 Score: {test_f1:.4f}")
    
    # Save test metrics
    test_report = classification_report(y_test, test_preds_labels, output_dict=True)
    test_metrics_df = pd.DataFrame(test_report).transpose()
    test_metrics_df.to_csv(os.path.join(RESULTS_DIR, 'test_metrics.csv'))
    
    # Save test confusion matrix
    class_names = np.unique(y_test)
    cm = confusion_matrix(y_test, test_preds_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
               xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix - Test Set')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'test_confusion_matrix.png'))
    plt.close()
    
    return best_model_name, test_accuracy, test_f1

# Main function
def main():
    print("Starting Alzheimer's Disease Detection Pipeline...")
    
    # 1 & 2: Extract, Convert, and Combine Data
    print("\n(1/5) Extracting and combining data...")
    combined_data = combine_all_data()
    
    # 3: Split the Data
    print("\n(2/5) Splitting and preparing data...")
    X_train, X_val, X_test, y_train, y_val, y_test, scaler, feature_names = prepare_data(combined_data)
    
    # 4: Train CNN Model
    print("\n(3/5) Training CNN model...")
    cnn_model, cnn_val_preds, label_encoder, cnn_val_acc = train_cnn_model(X_train, y_train, X_val, y_val)
    
    # 4: Train Traditional Models
    print("\n(3/5) Training traditional models...")
    trad_results_df, val_predictions, models, trad_le = train_traditional_models(
        X_train, y_train, X_val, y_val, X_test, y_test, feature_names)
    
    # 5: Analyze and Visualize Performance
    print("\n(4/5) Analyzing model performance...")
    all_results_df = analyze_all_models(
        trad_results_df, cnn_val_acc, cnn_val_preds, y_val, 
        val_predictions, X_val, np.unique(y_val))
    
    # 6: Save the Best Model
    print("\n(5/5) Saving best model...")
    best_model_name, test_accuracy, test_f1 = save_best_model(
        all_results_df, models, cnn_model, label_encoder, X_test, y_test, scaler)
    
    print("\nPipeline completed successfully!")
    print(f"Best model: {best_model_name} with test F1 score: {test_f1:.4f}")
    print(f"Results saved to {RESULTS_DIR}")
    print(f"Models saved to {MODELS_DIR}")

if __name__ == "__main__":
    main() 