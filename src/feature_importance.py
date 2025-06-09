import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

def simple_shap_analysis(model, data_path: str):

    try:
        # Load data
        data = pd.read_csv(data_path)
        feature_names = [col for col in data.columns if col not in ['customer_id', 'purchase']]
        X = data[feature_names].values
        
        model.eval()
        
        # Create SHAP explainer
        background_data = torch.tensor(X[:50], dtype=torch.float32)
        explainer = shap.DeepExplainer(model, background_data)
        
        # Calculate SHAP values
        sample_data = torch.tensor(X[:100], dtype=torch.float32)
        shap_values = explainer.shap_values(sample_data)
        
        # Calculate feature importance
        importance_scores = np.abs(shap_values).mean(axis=0)
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_scores
        }).sort_values('importance', ascending=False)
        
        # Create bar plot
        plt.figure(figsize=(10, 6))
        plt.barh(importance_df['feature'], importance_df['importance'], color='skyblue')
        plt.xlabel('SHAP Importance Score')
        plt.title('Feature Importance (SHAP Analysis)')
        plt.gca().invert_yaxis()
        
        # Add values on bars
        for i, v in enumerate(importance_df['importance']):
            plt.text(v + 0.001, i, f'{v:.3f}', va='center')
        
        plt.tight_layout()
        plt.savefig('outputs/validation_img/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return importance_df
        
    except:
        return None

def simple_shap_analysis_with_path(model_path: str, data_path: str):
    """
    Version with model path - for backward compatibility
    """
    try:
        from .model import PurchaseModel
        
        data = pd.read_csv(data_path)
        feature_names = [col for col in data.columns if col not in ['customer_id', 'purchase']]
        
        model = PurchaseModel(input_size=len(feature_names))
        model.load_state_dict(torch.load(model_path))
        
        return simple_shap_analysis(model, data_path)
    except:
        return None

def quick_shap_summary(model, data_path: str):
    """
    Quick SHAP summary plot
    """
    try:
        import shap
    except ImportError:
        return None
    
    try:
        data = pd.read_csv(data_path)
        feature_names = [col for col in data.columns if col not in ['customer_id', 'purchase']]
        X = data[feature_names].values
        
        model.eval()
        
        background = torch.tensor(X[:30], dtype=torch.float32)
        explainer = shap.DeepExplainer(model, background)
        
        sample = torch.tensor(X[:50], dtype=torch.float32)
        shap_values = explainer.shap_values(sample)
        
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X[:50], feature_names=feature_names, show=False)
        plt.title('SHAP Summary - How Features Affect Predictions')
        plt.tight_layout()
        plt.savefig('outputs/validation_img/shap_summary.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    except:
        return None 