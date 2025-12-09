import torch
import numpy as np
import pandas as pd
from robustx.lib.tasks.ClassificationTask import ClassificationTask
from robustx.generators.CEGenerator import CEGenerator


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
    def __init__(self, theta):
        super(EntropicRisk, self).__init__()
        self.theta = theta

    def forward(self, ensemble_proba):
        """
        Computes the entropic risk given the ensemble prediction probabilities.
        Args:
            :param ensemble_proba: ensemble prediction probabilities, shape: (n_models, batch_size)
            :return: entropic risk values, shape: (batch_size)
        """
        # ensemble_proba contains prediction probs of each model in the ensemble
        # for the target class, shape: (n_models, batch_size)
        exp_logits = torch.exp(1 - self.theta * ensemble_proba)  # Apply exponential and 1-theta*m(x) loss
        avg_exp_logits = torch.mean(exp_logits, dim=0)  # Average over models, Shape: (batch_size)        
        risk = (1 / self.theta) * torch.log(avg_exp_logits + 1e-10)  # Add small constant for numerical stability
        return risk


class EntropicRiskCE(CEGenerator):
    """
    Entropic risk measure based robust counterfactual explanation generator.


    Inherits from CEGenerator and implements the _generation_method to find counterfactuals
    using gradient descent.
    """

    def _generation_method(self, 
                           instance:pd.DataFrame,
                           base_cf_gen_class:CEGenerator=None,
                           base_cf_gen_args:dict=None,
                           target_class:int=1,
                           max_iter:int=10,
                           theta:float=1.0,
                           tau:float=0.5,
                           lr:float=0.01, 
                           ref_model_idx:int=None,
                           seed:int=None,
                           device:str=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
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
        
        # Check that base generator class and args are provided
        assert base_cf_gen_class is not None, "You must provide a base_cf_gen_class for generating the initial counterfactual."
        assert base_cf_gen_args is not None, "You must provide base_cf_gen_args for the base counterfactual generator."

        # Get a reference model from the ensemble
        model_ensemble = self.task.model.model_ensemble
        n_models = len(model_ensemble)
        if ref_model_idx is None:
            np.random.seed(seed)
            ref_model_idx = np.random.randint(0, n_models)
        ref_model = model_ensemble[ref_model_idx]

        # Get an initial counterfactual from the base generator
        ref_task = ClassificationTask(ref_model, self.task.training_data)
        ce_gen = base_cf_gen_class(ref_task)
        ref_ce = ce_gen.generate_for_instance(instance, **base_cf_gen_args)


        ref_ce = torch.Tensor(ref_ce.to_numpy()).to(device)
        ent_ce = torch.autograd.Variable(ref_ce.clone(), requires_grad=True).to(device)

        optimiser = torch.optim.Adam([ent_ce], lr, amsgrad=True)
        entropic_risk = EntropicRisk(theta).to(device)

        # Optimization loop
        iterations = 0
        cf_is_valid = False
        while not cf_is_valid and iterations <= max_iter:
            optimiser.zero_grad()

            # For multiclass classification, predict_ensemble_proba_tensor returns shape: (n_models, batch_size[=1], n_classes) 
            # For binary classification, predict_ensemble_proba_tensor shape: (n_models, batch_size[=1], 1)
            # Either case, we need to extract the probabilities for the target class and the resultant
            # class_prob should have shape: (n_models, batch_size)
            class_prob = self.task.model.predict_ensemble_proba_tensor(ent_ce)
            if class_prob.shape[2] >= 2:
                class_prob = class_prob[:, 0, target_class].squeeze(dim=2)  # Get probs for positive class
            else:
                class_prob = class_prob.squeeze(dim=2)  # Get probs for positive class
            
            risk = entropic_risk(class_prob)
            risk.sum().backward()
            optimiser.step()

            # Break conditions
            if risk.item() < tau:
                cf_is_valid = True
            iterations += 1
            
            # Print verbose info
            if verbose:
                print(f"Iteration {iterations:02d}: Entropic risk = {risk.item():.4f}")
                print(f"Current CE: {ent_ce.detach().cpu().numpy()}")
        
        # If risk threshold not met, print warning
        if not cf_is_valid:
            print("Warning: Entropic CE generation did not converge to a valid counterfactual within the max iterations.")

        # Return the counterfactual as a DataFrame
        res = pd.DataFrame(ent_ce.detach().numpy())
        res.columns = instance.index
        
        return res 