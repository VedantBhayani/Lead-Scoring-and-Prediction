import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, List
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import os

class LeadDashboard:
    def __init__(self):
        """Initialize the dashboard."""
        st.set_page_config(
            page_title="LeadScore",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Apply custom CSS
        self.apply_custom_css()
    
    def apply_custom_css(self):
        """Apply custom CSS styling."""
        st.markdown("""
            <style>
                .stApp {
                    background-color: #f8f9fa;
                }
                .status-high {
                    background-color: #d4edda;
                    color: #155724;
                    padding: 4px 8px;
                    border-radius: 4px;
                }
                .status-medium {
                    background-color: #fff3cd;
                    color: #856404;
                    padding: 4px 8px;
                    border-radius: 4px;
                }
                .status-low {
                    background-color: #f8d7da;
                    color: #721c24;
                    padding: 4px 8px;
                    border-radius: 4px;
                }
            </style>
        """, unsafe_allow_html=True)
    
    def load_predictions(self) -> pd.DataFrame:
        """Load the most recent predictions."""
        output_dir = os.path.join(os.getcwd(), 'output')
        prediction_files = [f for f in os.listdir(output_dir) if f.startswith('predictions_')]
        if not prediction_files:
            return None
        
        latest_file = max(prediction_files)
        predictions_path = os.path.join(output_dir, latest_file)
        return pd.read_csv(predictions_path)
    
    def display_lead_score_distribution(self, scores: List[float]):
        """Display the lead score distribution chart."""
        st.subheader("Lead Score Distribution")
        
        # Create bins for scores
        bins = [0, 20, 40, 60, 80, 100]
        labels = ['0-20', '21-40', '41-60', '61-80', '81-100']
        score_bins = pd.cut(scores, bins=bins, labels=labels)
        counts = pd.value_counts(score_bins, sort=False)
        
        # Create bar chart
        fig = go.Figure(data=[
            go.Bar(
                x=labels,
                y=counts.values,
                marker_color='#3B82F6'
            )
        ])
        
        fig.update_layout(
            plot_bgcolor='white',
            showlegend=False,
            margin=dict(t=0, b=0, l=0, r=0),
            xaxis=dict(
                showgrid=True,
                gridcolor='#E5E7EB',
                title=None
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='#E5E7EB',
                title=None
            ),
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def display_lead_predictions_table(self, df: pd.DataFrame):
        """Display the lead predictions table."""
        st.subheader("Lead Score Predictions")
        
        # Create the table
        table_data = []
        for _, row in df.iterrows():
            # Determine status class based on lead score
            score = row['lead_score']
            if score >= 80:
                status = "High"
                status_class = "status-high"
            elif score >= 50:
                status = "Medium"
                status_class = "status-medium"
            else:
                status = "Low"
                status_class = "status-low"
            
            # Get key factors
            factors = []
            if row['website_visits'] > 20:
                factors.append("Website Visit")
            if row['email_opens'] > 10:
                factors.append("Email Open")
            if row['form_submissions'] > 2:
                factors.append("Form Submission")
            
            # Add row to table
            table_data.append({
                "Name": row['lead_name'],
                "Score": f"{int(row['lead_score'])}/100",
                "Probability": f"{row['conversion_probability']:.0%}",
                "Key Factors": "• " + "\n• ".join(factors),
                "Status": f'<span class="{status_class}">{status}</span>'
            })
        
        # Convert to DataFrame for display
        table_df = pd.DataFrame(table_data)
        st.write(table_df.to_html(escape=False, index=False), unsafe_allow_html=True)
    
    def run(self):
        """Run the dashboard."""
        # Sidebar
        with st.sidebar:
            st.title("LeadScore")
            st.markdown("---")
            menu_selection = st.radio(
                "Menu",
                ["Dashboard", "Leads", "Predictions"]
            )
        
        # Main content
        if menu_selection == "Dashboard":
            st.title("Prediction Results")
            
            # Load predictions
            df = self.load_predictions()
            if df is not None:
                # Display lead score distribution
                self.display_lead_score_distribution(df['lead_score'])
                
                # Display predictions table
                self.display_lead_predictions_table(df)
            else:
                st.info("No predictions available. Please run the prediction model first.")

def main():
    dashboard = LeadDashboard()
    dashboard.run()

if __name__ == "__main__":
    main() 