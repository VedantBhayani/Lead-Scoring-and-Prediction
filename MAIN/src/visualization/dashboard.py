import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, List
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json

class LeadDashboard:
    def __init__(self):
        """Initialize the dashboard."""
        st.set_page_config(page_title="Lead Scoring Dashboard", layout="wide")
    
    def display_lead_score(self, lead_data: Dict[str, Any]) -> None:
        """Display lead score and related information."""
        st.header("Lead Score")
        
        # Create three columns for score display
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Lead Score", f"{lead_data['lead_score']}")
        
        with col2:
            st.metric("Conversion Probability", 
                     f"{lead_data['conversion_probability']:.2%}")
        
        with col3:
            st.metric("Category", lead_data['category'])
        
        # Display feature importance
        st.subheader("Feature Importance")
        feature_importance = pd.DataFrame(
            list(lead_data['feature_importance'].items()),
            columns=['Feature', 'Importance']
        )
        fig = px.bar(feature_importance, x='Importance', y='Feature',
                    title='Feature Importance for this Lead')
        st.plotly_chart(fig)
    
    def display_batch_results(self, predictions: List[Dict[str, Any]]) -> None:
        """Display batch prediction results."""
        st.header("Batch Prediction Results")
        
        # Convert predictions to DataFrame
        df = pd.DataFrame(predictions)
        
        # Display summary metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Average Lead Score", 
                     f"{df['lead_score'].mean():.1f}")
        
        with col2:
            st.metric("Hot Leads", 
                     f"{len(df[df['category'] == 'Hot'])}")
        
        with col3:
            st.metric("Conversion Rate", 
                     f"{df['predicted_conversion'].mean():.1%}")
        
        # Display lead score distribution
        st.subheader("Lead Score Distribution")
        fig = px.histogram(df, x='lead_score', color='category',
                          title='Lead Score Distribution by Category')
        st.plotly_chart(fig)
        
        # Display detailed results table
        st.subheader("Detailed Results")
        st.dataframe(df)
    
    def display_model_metrics(self, metrics: Dict[str, float]) -> None:
        """Display model performance metrics."""
        st.header("Model Performance")
        
        # Display ROC curve
        st.subheader("ROC Curve")
        fpr, tpr, _ = metrics['precision_recall_curve']
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines',
                               name=f'ROC (AUC = {metrics["roc_auc"]:.3f})'))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                               name='Random Classifier', line=dict(dash='dash')))
        fig.update_layout(title='ROC Curve', xaxis_title='False Positive Rate',
                         yaxis_title='True Positive Rate')
        st.plotly_chart(fig)
        
        # Display other metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("ROC AUC Score", f"{metrics['roc_auc']:.3f}")
        
        with col2:
            st.metric("F1 Score", f"{metrics['f1']:.3f}")
    
    def run(self):
        """Run the dashboard."""
        st.title("Lead Scoring Dashboard")
        
        # Sidebar for file upload
        with st.sidebar:
            st.header("Input")
            uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
            
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
                st.success("File uploaded successfully!")
                
                # Display sample of uploaded data
                st.subheader("Sample Data")
                st.dataframe(df.head())
        
        # Main content
        if uploaded_file is not None:
            # Add tabs for different views
            tab1, tab2, tab3 = st.tabs(["Lead Scores", "Model Performance", "Data Analysis"])
            
            with tab1:
                # Display lead scores
                if 'lead_score' in df.columns:
                    self.display_batch_results(df.to_dict('records'))
                else:
                    st.warning("No lead scores found in the data.")
            
            with tab2:
                # Display model metrics if available
                if 'metrics' in st.session_state:
                    self.display_model_metrics(st.session_state.metrics)
                else:
                    st.info("Train the model to see performance metrics.")
            
            with tab3:
                # Display data analysis
                st.subheader("Data Overview")
                st.write(f"Total Records: {len(df)}")
                st.write(f"Features: {', '.join(df.columns)}")
                
                # Display feature distributions
                for col in df.select_dtypes(include=['number']).columns:
                    fig = px.histogram(df, x=col, title=f'{col} Distribution')
                    st.plotly_chart(fig)

def main():
    dashboard = LeadDashboard()
    dashboard.run()

if __name__ == "__main__":
    main() 