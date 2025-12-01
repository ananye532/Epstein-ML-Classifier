"""
Epstein Files Document Classification System
==============================================
A machine learning pipeline for analyzing and classifying documents from the 
U.S. House Oversight Committee's public release of Epstein estate documents.

Author: [Your Name]
Date: December 2025
Dataset: House Oversight Committee Release (Nov 2025)
Source: https://huggingface.co/datasets/tensonaut/EPSTEIN_FILES_20K
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, List

# NLP and ML libraries
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import kagglehub

# Text preprocessing
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')


class EpsteinFilesClassifier:
    """
    A comprehensive document classification system for analyzing Epstein estate files.
    
    This classifier performs:
    1. Data loading and preprocessing
    2. Feature extraction using TF-IDF
    3. Multi-model classification
    4. Performance evaluation and visualization
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the classifier with configuration parameters.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.vectorizer = None
        self.models = {}
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_data(self, use_kaggle: bool = True) -> pd.DataFrame:
        """
        Load the Epstein files dataset.
        
        Args:
            use_kaggle: If True, download from Kaggle; else load from local path
            
        Returns:
            DataFrame containing the documents
        """
        print("📂 Loading dataset...")
        
        if use_kaggle:
            # Download from Kaggle
            path = kagglehub.dataset_download("jazivxt/the-epstein-files")
            print(f"✅ Dataset downloaded to: {path}")
            
            # Find CSV file in the downloaded path
            csv_files = list(Path(path).rglob("*.csv"))
            if not csv_files:
                raise FileNotFoundError("No CSV files found in dataset")
            
            df = pd.read_csv(csv_files[0])
        else:
            # Load from local path (if already downloaded)
            df = pd.read_csv("path_to_local_csv.csv")
        
        print(f"✅ Loaded {len(df)} documents")
        self.data = df
        return df
    
    def explore_data(self) -> None:
        """
        Perform exploratory data analysis and print insights.
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        print("\n" + "="*60)
        print("📊 EXPLORATORY DATA ANALYSIS")
        print("="*60)
        
        print(f"\nDataset Shape: {self.data.shape}")
        print(f"\nColumns: {list(self.data.columns)}")
        print(f"\nData Types:\n{self.data.dtypes}")
        print(f"\nMissing Values:\n{self.data.isnull().sum()}")
        
        # Display sample
        print("\n📄 Sample Documents:")
        print(self.data.head(3))
    
    def preprocess_text(self, text: str) -> str:
        """
        Clean and preprocess a single text document.
        
        Args:
            text: Raw text string
            
        Returns:
            Cleaned and preprocessed text
        """
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Tokenization and stopword removal
        stop_words = set(stopwords.words('english'))
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
        
        return ' '.join(tokens)
    
    def create_document_categories(self) -> pd.Series:
        """
        Create document categories based on file paths or content patterns.
        This is a simplified categorization for demonstration purposes.
        
        Returns:
            Series with document categories
        """
        print("\n🏷️  Creating document categories...")
        
        # Example categorization based on file paths
        # Adjust this based on actual dataset structure
        categories = []
        
        for idx, row in self.data.iterrows():
            # Get the file path/source if available
            source = str(row.get('source_file', row.get('file_path', '')))
            text = str(row.get('text', ''))
            
            # Categorize based on keywords and patterns
            if 'email' in source.lower() or '@' in text[:200]:
                categories.append('Email')
            elif 'image' in source.lower() or 'jpg' in source.lower():
                categories.append('Scanned_Document')
            elif 'legal' in source.lower() or 'affidavit' in text.lower():
                categories.append('Legal_Document')
            elif 'financial' in text.lower() or 'invoice' in text.lower():
                categories.append('Financial_Record')
            else:
                categories.append('Other')
        
        category_counts = pd.Series(categories).value_counts()
        print(f"\n📋 Document Categories:\n{category_counts}")
        
        return pd.Series(categories)
    
    def prepare_features(self, max_features: int = 5000) -> Tuple:
        """
        Prepare features using TF-IDF vectorization.
        
        Args:
            max_features: Maximum number of features for TF-IDF
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        print("\n🔧 Preparing features...")
        
        # Clean text data
        text_column = 'text' if 'text' in self.data.columns else self.data.columns[0]
        print(f"Using column: {text_column}")
        
        self.data['cleaned_text'] = self.data[text_column].apply(self.preprocess_text)
        
        # Create labels
        self.data['category'] = self.create_document_categories()
        
        # Remove documents with empty text
        self.data = self.data[self.data['cleaned_text'].str.len() > 0]
        
        # TF-IDF Vectorization
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )
        
        X = self.vectorizer.fit_transform(self.data['cleaned_text'])
        y = self.data['category']
        
        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        print(f"✅ Training set: {self.X_train.shape[0]} documents")
        print(f"✅ Test set: {self.X_test.shape[0]} documents")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_models(self) -> Dict:
        """
        Train multiple classification models for comparison.
        
        Returns:
            Dictionary of trained models with their performance
        """
        print("\n🤖 Training classification models...")
        
        # Define models
        models = {
            'Naive_Bayes': MultinomialNB(),
            'Logistic_Regression': LogisticRegression(max_iter=1000, random_state=self.random_state),
            'Random_Forest': RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\n📈 Training {name}...")
            
            # Train
            model.fit(self.X_train, self.y_train)
            
            # Predict
            y_pred = model.predict(self.X_test)
            
            # Evaluate
            accuracy = accuracy_score(self.y_test, y_pred)
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'predictions': y_pred
            }
            
            print(f"✅ {name} Accuracy: {accuracy:.4f}")
        
        self.models = results
        return results
    
    def evaluate_model(self, model_name: str = 'Logistic_Regression') -> None:
        """
        Generate detailed evaluation metrics for a specific model.
        
        Args:
            model_name: Name of the model to evaluate
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Available: {list(self.models.keys())}")
        
        print(f"\n" + "="*60)
        print(f"📊 DETAILED EVALUATION: {model_name}")
        print("="*60)
        
        y_pred = self.models[model_name]['predictions']
        
        # Classification report
        print("\n📈 Classification Report:")
        print(classification_report(self.y_test, y_pred))
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=sorted(self.y_test.unique()),
                    yticklabels=sorted(self.y_test.unique()))
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'confusion_matrix_{model_name}.png', dpi=300)
        print(f"\n✅ Confusion matrix saved as 'confusion_matrix_{model_name}.png'")
    
    def plot_model_comparison(self) -> None:
        """
        Create a bar plot comparing all model accuracies.
        """
        if not self.models:
            raise ValueError("No models trained yet. Call train_models() first.")
        
        accuracies = {name: results['accuracy'] for name, results in self.models.items()}
        
        plt.figure(figsize=(10, 6))
        plt.bar(accuracies.keys(), accuracies.values(), color='steelblue', alpha=0.8)
        plt.title('Model Performance Comparison', fontsize=16, fontweight='bold')
        plt.ylabel('Accuracy', fontsize=12)
        plt.xlabel('Model', fontsize=12)
        plt.ylim([0, 1])
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, (model, acc) in enumerate(accuracies.items()):
            plt.text(i, acc + 0.02, f'{acc:.3f}', ha='center', fontsize=11)
        
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300)
        print("\n✅ Model comparison plot saved as 'model_comparison.png'")
    
    def predict_new_document(self, text: str, model_name: str = 'Logistic_Regression') -> str:
        """
        Classify a new document.
        
        Args:
            text: Document text to classify
            model_name: Name of model to use for prediction
            
        Returns:
            Predicted category
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        # Preprocess
        cleaned = self.preprocess_text(text)
        
        # Vectorize
        features = self.vectorizer.transform([cleaned])
        
        # Predict
        model = self.models[model_name]['model']
        prediction = model.predict(features)[0]
        
        # Get probability
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(features)[0]
            confidence = max(proba)
            print(f"\n🎯 Prediction: {prediction} (Confidence: {confidence:.2%})")
        else:
            print(f"\n🎯 Prediction: {prediction}")
        
        return prediction


def main():
    """
    Main execution pipeline.
    """
    print("="*60)
    print("EPSTEIN FILES DOCUMENT CLASSIFICATION SYSTEM")
    print("="*60)
    
    # Initialize classifier
    classifier = EpsteinFilesClassifier(random_state=42)
    
    # Load data
    df = classifier.load_data(use_kaggle=True)
    
    # Explore data
    classifier.explore_data()
    
    # Prepare features
    classifier.prepare_features(max_features=3000)
    
    # Train models
    classifier.train_models()
    
    # Evaluate best model
    classifier.evaluate_model('Logistic_Regression')
    
    # Compare models
    classifier.plot_model_comparison()
    
    print("\n" + "="*60)
    print("✅ PIPELINE COMPLETED SUCCESSFULLY")
    print("="*60)
    print("\nGenerated files:")
    print("  - confusion_matrix_Logistic_Regression.png")
    print("  - model_comparison.png")
    print("\nModel ready for predictions!")


if __name__ == "__main__":
    main()
