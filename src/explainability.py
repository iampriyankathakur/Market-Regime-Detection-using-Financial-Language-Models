import shap
import torch
import numpy as np

def shap_explain(model, X_sample):

    def model_predict(data):
        data = torch.tensor(data).float()
        with torch.no_grad():
            return model(data).numpy()

    explainer = shap.KernelExplainer(model_predict, X_sample[:50])
    shap_values = explainer.shap_values(X_sample[:50])
    shap.summary_plot(shap_values, X_sample[:50])
