import torch
import numpy as np
import pandas as pd
from robustx.lib.tasks.ClassificationTask import ClassificationTask
from robustx.generators.CEGenerator import CEGenerator
from robustx.lib.models.BaseModel import BaseModel
from typing import Union, Dict, Callable, Any
from collections.abc import Iterable


class EntropicRisk(torch.nn.Module):
    """
    Implements the entropic risk measure for counterfactual explanation generation.
    Given an ensemble of model prediction probabilities, it computes the entropic risk
    as defined by the formula:
        R_theta(m(x)) = (1/theta) * log( E[ exp(1 - theta * m_i(x)) ] )
    where the expectation is over the models in the ensemble.

    Args:
        :theta (float): Risk aversion parameter. Higher values indicate more risk-averse behavior.
    """
    def __init__(self, theta:float, loss_fn:Callable, weights:Union[Dict[Any, float], None]):
        """
        Initialize EntropicRisk loss
        
        :param theta: Risk aversion parameter
        :param loss_fn: Loss for computing the risk
        :param weights: Weights for considering importance in multi-target loss computation
        """
        super(EntropicRisk, self).__init__()
        self.theta = theta
        self.loss_fn = loss_fn
        self.weights = weights

    def forward(self, 
                preds:Union[torch.Tensor, Dict[Any, torch.Tensor]], 
            ):
        """
        Computes the entropic risk given the ensemble prediction probabilities.
        Args:
            :param preds: Ensemble prediction probabilities for a given class - a tensor with shape 
                          (n_models, batch_size) or a dictionary of the form {target: preds_ensemble} with 
                          preds_ensemble is a tensor with shape (n_models, batch_size)
            :return: Entropic risk values, shape: (batch_size)
        """
        
        if isinstance(preds, torch.Tensor):
            # Apply exponential and theta* loss_fn(m(x)) loss. Shape:(n_models, batch_size)
            exp_logits = torch.exp(self.theta * self.loss_fn(preds))
            # Average over models. Shape: (batch_size)     
            avg_exp_logits = torch.mean(exp_logits, dim=0)     
            # Compute risk. Add small constant for numerical stability. Shape: (batch_size)
            risk = (1 / self.theta) * torch.log(avg_exp_logits + 1e-10)
        else:
            target_keys = preds.keys()
            if self.weights is None:
                self.weights = {key: 1 for key in target_keys}

            avg_exp_logits_list = []
            for target_key in target_keys:
                # Apply exponential and theta*loss_fn(m(x)) loss, shape: (n_models, batch_size)
                # print(f"preds target: {preds[target_key].shape}")
                exp_logits = self.weights[target_key]*torch.exp(self.theta * self.loss_fn(preds[target_key]))
                # print(f"exp_logits: {exp_logits.shape}")
                # Average over models. Shape: (batch_size)
                avg_exp_logits = torch.mean(exp_logits, dim=0)
                # print(f"avg_exp_logits: {avg_exp_logits.shape}")    
                # Append to list for later computing weighted average. Shape: [(batch_size)]
                avg_exp_logits_list.append(avg_exp_logits)
                # print(f"avg_exp_logits_list: {avg_exp_logits_list}")    

            # Compute weighted average, shape: sum(n_targets, batch_size) --> (batch_size)
            wavg_exp_logits = torch.sum(torch.stack(avg_exp_logits_list, dim=0), dim=0) / sum(self.weights.values())
            # print(f"wavg_exp_logits: {wavg_exp_logits.shape}")    
            # Compute risk. Add small constant for numerical stability. Shape: (batch_size)
            risk = (1 / self.theta) * torch.log(wavg_exp_logits + 1e-10)  
        
        return risk


class EntropicRiskCE(CEGenerator):
    """
    Entropic risk measure based robust counterfactual explanation generator.


    Inherits from CEGenerator and implements the _generation_method to find counterfactuals
    using gradient descent.
    """

    def _generation_method(self, 
                           instance:pd.DataFrame,
                           target_class:int=1,
                           loss_fn: Callable=lambda x: 1-x,
                           target_weights: Dict[Any, float]=None,
                           project_to_range: bool=True,
                           permitted_ranges: Dict[Any, Iterable]=None,
                           immutable_features: Iterable[str]=[],
                           max_iter:int=10,
                           theta:float=1.0,
                           tau:float=0.5,
                           lr:float=0.01, 
                           device:str="cuda" if torch.cuda.is_available() else "cpu",
                           verbose:bool=False,
                           **kwargs):
        """
        Generates a counterfactual explanation for the given instance using entropic risk minimization.
        
        Args:
            :param instance: The input instance for which to generate a counterfactual.
            :param base_cf_gen_class: The base counterfactual generator class to generate the initial counterfactual.
            :param base_cf_gen_args: Arguments for the base counterfactual generator.
            :param target_class: The desired target class for the counterfactual.
            :param max_iter: Maximum number of optimization iterations.
            :param theta: Risk aversion parameter for entropic risk.
            :param tau: Risk threshold for valid counterfactual.
            :param lr: Learning rate for the optimizer.
            :param ref_model_idx: Index of the reference model in the ensemble. If None, a random model is selected.
            :param seed: Random seed for selecting a model from the ensemble of models, for reproducibility.
            :param device: Device to run the optimization on (CPU or GPU).
            :param verbose: If True, prints detailed optimization information.
            :return: A DataFrame representing the generated counterfactual explanation.
        """
        
        # Prepare a list of indices of the immutable features
        immutable_indices = [instance.index.get_loc(im_feat) for im_feat in immutable_features]
        
        # Prepare a dictionary of permitted ranges with feature location instead of feature name
        if permitted_ranges is not None:
            permitted_ranges_with_idx = {
                instance.index.get_loc(feature_name): val_range for feature_name, val_range in permitted_ranges.items()
            }
        
        # Initialize counterfactual <-- instance
        ref_ce = torch.tensor(instance.to_numpy(), dtype=torch.float, device=device)
        if len(ref_ce.shape) < 2:
            ref_ce = ref_ce.unsqueeze(dim=0) # Insert batch dimension if required

        ent_ce = ref_ce.detach().clone().requires_grad_(True)
               
        # Instantiate optimizer and entropic risk loss
        optimiser = torch.optim.Adam([ent_ce], lr, amsgrad=True)
        entropic_risk = EntropicRisk(
            theta=theta,
            loss_fn=loss_fn,
            weights=target_weights
        ).to(device)

        # Optimization loop
        iterations = 0
        cf_is_valid = False
        while not cf_is_valid and iterations <= max_iter:
            optimiser.zero_grad()

            # For multiclass classification, predict_ensemble_proba_tensor returns shape: (n_models, batch_size[=1], n_classes) 
            # For binary classification, predict_ensemble_proba_tensor shape: (n_models, batch_size[=1], 1)
            # Either case, we need to extract the probabilities for the target class and the resultant
            # class_prob should have shape: (n_models, batch_size)                        
            preds = self.task.model.predict_ensemble_proba_tensor(ent_ce)

            if isinstance(preds, torch.Tensor):
                if preds.shape[2] >= 2:
                    class_prob = preds[:, :, target_class]  # Get probs for positive class, shape: (n_models, batch_size)
                else:
                    class_prob = preds  # Get probs for positive class, shape: (n_models, batch_size)
            elif isinstance(preds, Dict):
                class_prob = {}
                for key in preds.keys():
                    if preds[key].shape[2] >= 2:
                        class_prob[key] = preds[key][:, :, target_class]  # Get probs for positive class, shape: (n_model, batch_size)
                    else:
                        class_prob[key] = preds[key]  # Get probs for positive class, shape: (n_model, batch_size)

            risk = entropic_risk(class_prob) # class_prob shape: (n_models, batch_size) or {target: (n_model, batch_size)}
            risk.backward()

            # Zero-out gradients for immutable features:
            ent_ce.grad[:, immutable_indices] = 0

            # Take gradient step
            optimiser.step()

            # Project the updated counterfactual features to permitted ranges
            if project_to_range:
                with torch.no_grad():
                    for feat_idx, (fmin, fmax) in permitted_ranges_with_idx.items():
                        ent_ce[:, feat_idx].clamp_(min=fmin, max=fmax)

            # Break conditions
            if risk.item() < tau:
                cf_is_valid = True
            iterations += 1
            
            # Print verbose info
            if verbose:
                print(f"Iteration {iterations:03d}: Entropic risk = {risk.item():.08f}")
                # print(f"Current CE: {ent_ce.detach().cpu().numpy()}")
        
        # If risk threshold not met, print warning
        if not cf_is_valid:
            print("Warning: Entropic CE generation did not converge to a valid counterfactual within the max iterations.")

        # Return the counterfactual as a DataFrame
        res = pd.DataFrame(ent_ce.detach().cpu().numpy(), columns=instance.index)
        
        return res 