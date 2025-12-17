from robustx.lib.models.BaseModel import BaseModel
from robustx.lib.models.pytorch_models.CustomPyTorchModel import CustomPyTorchModel
import torch
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import pandas as pd
import torch

from typing import Union, Dict, Any, List


class EnsembleModelWrapper(BaseModel):
    ENSEMBLE_SINGLE_OUTPUT_SINGLE_TARGET_LIST = "single_target_list"
    ENSEMBLE_SINGLE_OUTPUT_MULTI_TARGET_DICT = "sinlge_output_multi_target_dict"
    ENSEMBLE_MULTI_OUTPUT_MULTI_TARGET_DICT = "multi_output_multi_target_dict"

    AGGREGATION_MAJORITY_VOTE = "majority_vote"

    def __init__(self, 
                 model_ensemble: Union[List[torch.nn.Module], Dict[Any, torch.nn.Module], torch.nn.Module],
                 device: str,
                 ensemble_type: str,
                 apply_softmax: bool,
                 aggregation_method: str
                ):
        super().__init__(EnsembleModelWrapper)
        
        self.model_ensemble = model_ensemble
        self.ensemble_type = ensemble_type
        self.aggregation_method = aggregation_method
        self.device = device
        self.apply_softmax = apply_softmax
    
    def train(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        print("Training should be done on individual models in the ensemble.")

    def aggregate_preds(self, X: torch.Tensor, aggregation_method, **kwargs):
        """
        Aggregate the probability predictions of an ensemble of models
        
        :param X: a tensor of predictions by the model ensemble with shape (n_models, batch_size, n_classes)
        :param aggregation_method: aggregation method
        :return: a tensor of aggregated predictions with shape (batch_size, n_classes)
        """
        if aggregation_method == EnsembleModelWrapper.AGGREGATION_MAJORITY_VOTE:
            individual_labels = torch.argmax(X, dim=2) # shape: (n_models, batch_size)
            final_labels, _ = torch.mode(individual_labels, dim=0) # shape: (batch_size)
            return final_labels
        else:
            print(f"Invalid aggregation method: {aggregation_method}")


    def predict_tensor(self, X: torch.Tensor) -> Union[torch.Tensor, Dict[Any, torch.Tensor]]:
        """
        Predict all probabilities for an input feature tensor
        
        :param X: tensor with shape (batch_size, n_features)
        :return: tensor with shape (n_models, batch_size, n_classes) if model ensemble is a list
                 a dictionary of the form {id: probs} where probs is a tensor with shape (batch_size, n_classes)
        """
        X.to(self.device)

        # If the ensemble is a list of models predicting the same target
        if self.ensemble_type == EnsembleModelWrapper.ENSEMBLE_SINGLE_OUTPUT_SINGLE_TARGET_LIST:
            assert isinstance(self.model_ensemble, List), f"Incorrect model ensemble type {self.ensemble_type}, where actual type is {type(self.model_ensemble)}"
            prob_ensemble = []
            for model in self.model_ensemble:
                model.to(self.device)
                model.eval()
                outputs = model(X)
                if self.apply_softmax:
                    outputs = torch.nn.functional.softmax(outputs, dim=1)
                prob_ensemble.append(outputs.unsqueeze(dim=0))
            prob_ensemble = torch.stack(prob_ensemble, dim=0)  # Shape: (n_models, n_samples, n_classes)
            return prob_ensemble
        
        # If the ensemble is a dict of single-output models predicting multiple targets
        elif self.ensemble_type == EnsembleModelWrapper.ENSEMBLE_SINGLE_OUTPUT_MULTI_TARGET_DICT:
            assert isinstance(self.model_ensemble, Dict), f"Incorrect model ensemble type {self.ensemble_type}, where actual type is {type(self.model_ensemble)}"
            prob_ensemble = {}
            for key in self.model_ensemble.keys():
                model = self.model_ensemble[key]
                model.to(self.device)
                model.eval()
                outputs = model(X)
                if self.apply_softmax:
                    outputs = torch.nn.functional.softmax(outputs, dim=1)
                prob_ensemble[key] = outputs
            return prob_ensemble
        
        # If the ensemble is a dict of multi-output models predicting multiple targets
        elif self.ensemble_type == EnsembleModelWrapper.ENSEMBLE_MULTI_OUTPUT_MULTI_TARGET_DICT:
            assert isinstance(self.model_ensemble, Dict), f"Incorrect model ensemble type {self.ensemble_type}, where actual type is {type(self.model_ensemble)}"
            prob_ensemble = {}
            for key in self.model_ensemble.keys():
                model = self.model_ensemble[key]
                model.to(self.device)
                model.eval()
                outputs = model(X)
                for k, v in outputs.items():
                    if self.apply_softmax:
                        v = torch.nn.functional.softmax(outputs, dim=1)
                    prob_ensemble[f"{key}_{k}"] = v
            return prob_ensemble
        
        else:
            print(f"Invalid model ensemble type {self.ensemble_type}, where actual type is {type(self.model_ensemble)}")
            return None
                    
    
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        X_tensor = torch.Tensor(X.to_numpy()).to(self.device)
        prob_ensemble = self.predict_tensor(X_tensor)

        # If the ensemble is a list of models predicting the same target
        if self.ensemble_type == EnsembleModelWrapper.ENSEMBLE_SINGLE_OUTPUT_SINGLE_TARGET_LIST:
            
        
        # If the ensemble is a dict of single-output models predicting multiple targets
        elif self.ensemble_type == EnsembleModelWrapper.ENSEMBLE_SINGLE_OUTPUT_MULTI_TARGET_DICT:
            predictions = {}
            for key in self.model_ensemble.keys():
                model = self.model_ensemble[key]
                model.to(self.device)
                model.eval()
                outputs = model(X)
                if self.apply_softmax:
                    outputs = torch.nn.functional.softmax(outputs, dim=1)
                prob_ensemble[key] = outputs
            return prob_ensemble
        
        # If the ensemble is a dict of multi-output models predicting multiple targets
        elif self.ensemble_type == EnsembleModelWrapper.ENSEMBLE_MULTI_OUTPUT_MULTI_TARGET_DICT:
            assert isinstance(self.model_ensemble, Dict), f"Incorrect model ensemble type {self.ensemble_type}, where actual type is {type(self.model_ensemble)}"
            prob_ensemble = {}
            for key in self.model_ensemble.keys():
                model = self.model_ensemble[key]
                model.to(self.device)
                model.eval()
                outputs = model(X)
                for k, v in outputs.items():
                    if self.apply_softmax:
                        v = torch.nn.functional.softmax(outputs, dim=1)
                    prob_ensemble[f"{key}_{k}"] = v
            return prob_ensemble
        
        else:
            print(f"Invalid model ensemble type {self.ensemble_type}, where actual type is {type(self.model_ensemble)}")
            return None





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