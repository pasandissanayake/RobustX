from robustx.lib.models.BaseModel import BaseModel
from robustx.lib.models.pytorch_models.SimpleNNModel import SimpleNNModel
import torch
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import pandas as pd


class EnsembleModelWrapper(BaseModel):
    def __init__(self, model_ensemble: list[SimpleNNModel], aggregation_method: str = 'majority_vote'):
        super().__init__(EnsembleModelWrapper)
        self.model_ensemble = model_ensemble
        self.pt_model_ensemble = [model._model for model in model_ensemble]
        self.aggregation_method = aggregation_method
    
    def train(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        print("Training should be done on individual models in the ensemble.")
    
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        device = next(self.pt_model_ensemble[0].parameters()).device
        X_tensor = torch.Tensor(X.to_numpy()).to(device)
        preds_ensemble = []
        for model in self.pt_model_ensemble:
            model.eval()
            with torch.no_grad():
                outputs = model(X_tensor).cpu().numpy()
                preds_ensemble.append(outputs)
        preds_ensemble = np.array(preds_ensemble)  # Shape: (n_models, n_samples, n_classes)
        if self.aggregation_method == 'majority_vote':
            final_preds = np.round(np.mean(preds_ensemble, axis=0)) # Shape: (n_samples, n_classes)
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")
        predictions = final_preds.astype(int)
        
        return pd.DataFrame(predictions, columns=['prediction'], index=X.index)
    
    def predict_single(self, X: pd.DataFrame) -> int:
        return self.predict(X).values.item()
    
    def predict_ensemble_proba_tensor(self, X: torch.Tensor) -> torch.Tensor:
        device = next(self.pt_model_ensemble[0].parameters()).device
        X = X.to(device)
        probs_ensemble = []
        for model in self.pt_model_ensemble:
            model.eval()
            outputs = model(X)
            probs_ensemble.append(outputs)
        probs_ensemble = torch.stack(probs_ensemble, dim=0)  # Shape: (n_models, n_samples, n_classes)
        return probs_ensemble
    
    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        X_tensor = torch.Tensor(X.to_numpy())
        probs_ensemble = self.predict_ensemble_proba_tensor(X_tensor).numpy()  # Shape: (n_models, n_samples, n_classes)
        if self.aggregation_method == 'majority_vote':
            aggregated_probs = np.mean(probs_ensemble, axis=0)
        return pd.DataFrame(aggregated_probs, columns=[f'class_{i}' for i in range(aggregated_probs.shape[1])], index=X.index)
        
    def predict_proba_tensor(self, X: torch.Tensor) -> torch.Tensor:
        X_numpy = X.numpy()
        probabilities = self.predict_proba(X_numpy)
        return torch.tensor(probabilities)
    
    def evaluate(self, X: pd.DataFrame, y: pd.DataFrame):
        y_pred = self.predict(X)
        accuracy = accuracy_score(y, y_pred)
        report = classification_report(y, y_pred)
        return {
            'accuracy': accuracy,
            'classification_report': report
        }
    
    def compute_accuracy(self, X_test, y_test):            
        return self.evaluate(X_test, y_test)['accuracy']