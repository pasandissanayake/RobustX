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
    ENSEMBLE_SINGLE_OUTPUT_MULTI_TARGET_DICT = "single_output_multi_target_dict"
    ENSEMBLE_MULTI_OUTPUT_MULTI_TARGET_LIST = "multi_output_multi_target_list"

    AGGREGATION_MAJORITY_VOTE = "majority_vote"

    def __init__(self, 
                 model_ensemble: Union[List[torch.nn.Module], Dict[Any, List[torch.nn.Module]]],
                 device: str,
                 ensemble_type: str,
                 aggregation_method: str
                ):
        super().__init__(EnsembleModelWrapper)
        
        self.model_ensemble = model_ensemble
        self.ensemble_type = ensemble_type
        self.aggregation_method = aggregation_method
        self.device = device
    
    def train(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        print("Training should be done on individual models in the ensemble.")


    def aggregate_preds(self, X: torch.Tensor, aggregation_method, **kwargs) -> torch.Tensor:
        """
        Aggregate the probability predictions of an ensemble of models
        
        :param X: Tensor of predictions by the model ensemble with shape (n_models, batch_size, n_classes)
        :param aggregation_method: Aggregation method
        :return: Tensor of aggregated predictions with shape (batch_size,)
        """
        if aggregation_method == EnsembleModelWrapper.AGGREGATION_MAJORITY_VOTE:
            individual_labels = torch.argmax(X, dim=2) # shape: (n_models, batch_size)
            final_labels, _ = torch.mode(individual_labels, dim=0) # shape: (batch_size)
            return final_labels
        else:
            print(f"Invalid aggregation method: {aggregation_method}")
            return torch.Tensor()


    def predict_tensor(self, X: torch.Tensor, apply_softmax: bool) -> Union[torch.Tensor, Dict[Any, torch.Tensor]]:
        """
        Predict all probabilities for an input feature tensor
        
        :param X: Tensor with shape (batch_size, n_features)
        :param apply_softmax: If True, apply softmax to model outputs to convert them into class-wise probabilities
        :return: In case of ENSEMBLE_SINGLE_OUTPUT_SINGLE_TARGET_LIST, tensor with shape (n_models, batch_size, n_classes)
                 In case of ENSEMBLE_SINGLE_OUTPUT_MULTI_TARGET_DICT or ENSEMBLE_MULTI_OUTPUT_MULTI_TARGET_LIST, 
                 a dict of the form {target: preds} where preds is a tensor with shape (n_models, batch_size, n_classes)                
        """
        X = X.to(self.device)

        # If the ensemble is a list of models predicting the same target, output: tensor (n_models, batch_size, n_classes)
        if self.ensemble_type == EnsembleModelWrapper.ENSEMBLE_SINGLE_OUTPUT_SINGLE_TARGET_LIST:
            assert isinstance(self.model_ensemble, List), f"Incorrect model ensemble type {self.ensemble_type}, where actual type is {type(self.model_ensemble)}"
            prob_ensemble = []
            for model in self.model_ensemble:
                model = model.to(self.device)
                model.eval()
                outputs = model(X) # shape: (batch_size, n_classes)
                if apply_softmax:
                    outputs = torch.nn.functional.softmax(outputs, dim=1)
                prob_ensemble.append(outputs) # shape: (1, batch_size, n_classes)
            prob_ensemble = torch.stack(prob_ensemble, dim=0)  # Shape: (n_models, batch_size, n_classes)
            return prob_ensemble
        
        # If the ensemble is a dict of single-output models predicting multiple targets
        elif self.ensemble_type == EnsembleModelWrapper.ENSEMBLE_SINGLE_OUTPUT_MULTI_TARGET_DICT:
            assert isinstance(self.model_ensemble, Dict), f"Incorrect model ensemble type {self.ensemble_type}, where actual type is {type(self.model_ensemble)}"
            prob_ensemble = {}
            for target_key in self.model_ensemble.keys():
                model_output_list = []
                for model in self.model_ensemble[target_key]:
                    model = model.to(self.device)
                    model.eval()
                    outputs = model(X)
                    if apply_softmax:
                        outputs = torch.nn.functional.softmax(outputs, dim=1) # shape: (batch_size, n_classes)
                    model_output_list.append(outputs)
                prob_ensemble[target_key] = torch.stack(model_output_list, dim=0) # shape: (n_models, batch_size, n_classes)
            return prob_ensemble
        
        # If the ensemble is a list of multi-output models predicting multiple targets
        elif self.ensemble_type == EnsembleModelWrapper.ENSEMBLE_MULTI_OUTPUT_MULTI_TARGET_LIST:
            assert isinstance(self.model_ensemble, Dict), f"Incorrect model ensemble type {self.ensemble_type}, where actual type is {type(self.model_ensemble)}"
            prob_ensemble = {}
            for model in self.model_ensemble:
                model = model.to(self.device)
                model.eval()
                outputs = model(X)
                for target_key, output in outputs.items():
                    if apply_softmax:
                        output = torch.nn.functional.softmax(output, dim=1) # shape: (batch_size, n_classes)
                    if target_key in prob_ensemble:
                        prob_ensemble[target_key].append(output)
                    else:
                        prob_ensemble[target_key] = [output]

            prob_ensemble = {
                key: torch.stack(val, dim=0) for key, val in prob_ensemble # shape: (n_models, batch_size, n_classes)
            }
            return prob_ensemble
        
        else:
            print(f"Invalid model ensemble type {self.ensemble_type}, where actual type is {type(self.model_ensemble)}")
            return {}
                    
    
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        X_tensor = torch.Tensor(X.to_numpy()).to(self.device)
        prob_ensemble = self.predict_tensor(X_tensor, apply_softmax=True)
        
        # If the ensemble is a list of models predicting the same target
        if self.ensemble_type == EnsembleModelWrapper.ENSEMBLE_SINGLE_OUTPUT_SINGLE_TARGET_LIST:
            aggregated_labels = self.aggregate_preds(prob_ensemble, self.aggregation_method)
            return pd.DataFrame(aggregated_labels.cpu().detach().numpy(), columns=['prediction'])
                           
        # If the ensemble is a dict of single-output models predicting multiple targets
        elif self.ensemble_type == EnsembleModelWrapper.ENSEMBLE_SINGLE_OUTPUT_MULTI_TARGET_DICT \
            or self.ensemble_type == EnsembleModelWrapper.ENSEMBLE_MULTI_OUTPUT_MULTI_TARGET_LIST:
            aggregated_labels = {}
            for target_key in prob_ensemble.keys():
                aggregated_labels[target_key] = self.aggregate_preds(prob_ensemble[target_key], self.aggregation_method).cpu().detach().numpy()
            return pd.DataFrame(aggregated_labels)
        
        else:
            print(f"Invalid model ensemble type {self.ensemble_type}, where actual type is {type(self.model_ensemble)}")
            return pd.DataFrame()
    
    def predict_single(self, X: pd.DataFrame) -> int:
        return self.predict(X).values.item()
    
    def predict_ensemble_proba_tensor(self, X: torch.Tensor) -> torch.Tensor:
        prob_ensemble = self.predict_tensor(X, apply_softmax=True)
        return prob_ensemble
    
    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        X_tensor = torch.Tensor(X.to_numpy())
        prob_ensemble = self.predict_ensemble_proba_tensor(X_tensor).numpy()
        return pd.DataFrame(prob_ensemble)
        
    def predict_proba_tensor(self, X: torch.Tensor) -> torch.Tensor:
        return self.predict_tensor(X, apply_softmax=True)
    
    def evaluate(self, X: pd.DataFrame, y: pd.DataFrame):
        y_pred = self.predict(X).to_dict(orient="list")
        results = {}
        for target, preds in y_pred.items():
            accuracy = accuracy_score(y[target], preds)
            report = classification_report(y[target], preds)
            results[target] = {'accuracy': accuracy, 'classification_report': report}
        return results
    
    def compute_accuracy(self, X_test:pd.DataFrame, y_test:pd.DataFrame):    
        results = self.evaluate(X_test, y_test)        
        return {
            target: results[target]['accuracy'] for target in y_test.columns
        }
            