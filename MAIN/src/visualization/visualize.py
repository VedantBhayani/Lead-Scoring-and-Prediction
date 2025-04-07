import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import shap

class LeadScoringVisualizer:
    def __init__(self):
        """Initialize the visualizer."""
        plt.style.use('seaborn')
        self.colors = sns.color_palette("husl", 8)
    
    def plot_feature_importance(self, feature_importance: pd.DataFrame, 
                              top_n: int = 10) -> None:
        """Plot feature importance scores."""
        plt.figure(figsize=(12, 6))
        top_features = feature_importance.head(top_n)
        
        sns.barplot(x='importance', y='feature', data=top_features, palette=self.colors)
        plt.title(f'Top {top_n} Most Important Features')
        plt.xlabel('Importance Score')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.show()
    
    def plot_lead_score_distribution(self, lead_scores: List[Dict[str, Any]]) -> None:
        """Plot distribution of lead scores."""
        scores = [score['lead_score'] for score in lead_scores]
        categories = [score['category'] for score in lead_scores]
        
        plt.figure(figsize=(12, 6))
        
        # Create subplot for score distribution
        plt.subplot(1, 2, 1)
        sns.histplot(scores, bins=20, color=self.colors[0])
        plt.title('Lead Score Distribution')
        plt.xlabel('Lead Score')
        plt.ylabel('Count')
        
        # Create subplot for category distribution
        plt.subplot(1, 2, 2)
        category_counts = pd.Series(categories).value_counts()
        plt.pie(category_counts, labels=category_counts.index, colors=self.colors)
        plt.title('Lead Category Distribution')
        
        plt.tight_layout()
        plt.show()
    
    def plot_roc_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> None:
        """Plot ROC curve."""
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = np.trapz(tpr, fpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color=self.colors[0], lw=2,
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.show()
    
    def plot_precision_recall_curve(self, y_true: np.ndarray, 
                                  y_pred_proba: np.ndarray) -> None:
        """Plot precision-recall curve."""
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color=self.colors[0], lw=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.grid(True)
        plt.show()
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """Plot confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()
    
    def create_lead_score_dashboard(self, lead_scores: List[Dict[str, Any]]) -> None:
        """Create an interactive dashboard using Plotly."""
        df = pd.DataFrame(lead_scores)
        
        # Create lead score distribution
        fig1 = px.histogram(df, x='lead_score', nbins=20,
                          title='Lead Score Distribution')
        
        # Create category distribution
        fig2 = px.pie(df, names='category', title='Lead Category Distribution')
        
        # Create feature importance heatmap
        feature_importance = pd.DataFrame([
            score['feature_importance'] for score in lead_scores
        ])
        fig3 = px.imshow(feature_importance, title='Feature Importance Heatmap')
        
        # Show all plots
        fig1.show()
        fig2.show()
        fig3.show()
    
    def plot_lead_trends(self, lead_scores: List[Dict[str, Any]], 
                        time_column: str = 'timestamp') -> None:
        """Plot lead scoring trends over time."""
        df = pd.DataFrame(lead_scores)
        df[time_column] = pd.to_datetime(df[time_column])
        
        # Daily average lead scores
        daily_scores = df.groupby(df[time_column].dt.date)['lead_score'].mean()
        
        plt.figure(figsize=(12, 6))
        daily_scores.plot(kind='line', color=self.colors[0])
        plt.title('Daily Average Lead Scores')
        plt.xlabel('Date')
        plt.ylabel('Average Lead Score')
        plt.grid(True)
        plt.show()
        
        # Category distribution over time
        category_trends = pd.crosstab(
            df[time_column].dt.date, 
            df['category']
        )
        
        plt.figure(figsize=(12, 6))
        category_trends.plot(kind='bar', stacked=True, colormap='Set3')
        plt.title('Lead Category Distribution Over Time')
        plt.xlabel('Date')
        plt.ylabel('Count')
        plt.legend(title='Category')
        plt.tight_layout()
        plt.show()

class LeadVisualizer:
    def __init__(self):
        """Initialize the visualizer."""
        plt.style.use('seaborn')
        sns.set_palette("husl")
    
    def plot_feature_importance(self, feature_importance: pd.DataFrame, 
                              top_n: int = 10) -> None:
        """Plot feature importance scores."""
        plt.figure(figsize=(10, 6))
        sns.barplot(data=feature_importance.head(top_n), 
                   x='importance', y='feature')
        plt.title('Top Feature Importance')
        plt.xlabel('Importance Score')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.show()
    
    def plot_roc_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> None:
        """Plot ROC curve."""
        from sklearn.metrics import roc_curve, auc
        
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.show()
    
    def plot_lead_score_distribution(self, lead_scores: List[Dict[str, Any]]) -> None:
        """Plot distribution of lead scores."""
        scores = [lead['lead_score'] for lead in lead_scores]
        categories = [lead['category'] for lead in lead_scores]
        
        plt.figure(figsize=(10, 6))
        sns.histplot(data=pd.DataFrame({
            'Lead Score': scores,
            'Category': categories
        }), x='Lead Score', hue='Category', multiple="stack")
        plt.title('Lead Score Distribution by Category')
        plt.xlabel('Lead Score')
        plt.ylabel('Count')
        plt.show()
    
    def plot_feature_correlations(self, df: pd.DataFrame) -> None:
        """Plot feature correlation matrix."""
        plt.figure(figsize=(12, 8))
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm', center=0)
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.show()
    
    def plot_shap_values(self, model: Any, X: np.ndarray, 
                        feature_names: List[str]) -> None:
        """Plot SHAP values for model interpretability."""
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X, feature_names=feature_names)
        plt.title('SHAP Values Summary')
        plt.tight_layout()
        plt.show()
    
    def plot_learning_curve(self, train_sizes: np.ndarray, 
                          train_scores: np.ndarray,
                          val_scores: np.ndarray) -> None:
        """Plot learning curve."""
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_scores.mean(axis=1), label='Training Score')
        plt.plot(train_sizes, val_scores.mean(axis=1), label='Validation Score')
        plt.fill_between(train_sizes, train_scores.mean(axis=1) - train_scores.std(axis=1),
                        train_scores.mean(axis=1) + train_scores.std(axis=1), alpha=0.1)
        plt.fill_between(train_sizes, val_scores.mean(axis=1) - val_scores.std(axis=1),
                        val_scores.mean(axis=1) + val_scores.std(axis=1), alpha=0.1)
        plt.xlabel('Training Examples')
        plt.ylabel('Score')
        plt.title('Learning Curve')
        plt.legend(loc='best')
        plt.grid(True)
        plt.show()

def main():
    # Example usage
    visualizer = LeadVisualizer()
    
    # Sample data
    feature_importance = pd.DataFrame({
        'feature': ['website_visits', 'email_opens', 'form_submissions', 
                   'company_size', 'lead_source'],
        'importance': [0.3, 0.25, 0.2, 0.15, 0.1]
    })
    
    # Sample lead scores
    lead_scores = [
        {
            'lead_score': 85,
            'category': 'Hot',
            'feature_importance': {'website_visits': 0.3, 'email_opens': 0.25}
        },
        {
            'lead_score': 65,
            'category': 'Warm',
            'feature_importance': {'website_visits': 0.3, 'email_opens': 0.25}
        },
        {
            'lead_score': 45,
            'category': 'Cold',
            'feature_importance': {'website_visits': 0.3, 'email_opens': 0.25}
        }
    ]
    
    # Generate visualizations
    visualizer.plot_feature_importance(feature_importance)
    visualizer.plot_lead_score_distribution(lead_scores)

if __name__ == "__main__":
    main() 