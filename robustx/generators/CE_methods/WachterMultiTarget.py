import torch
import torch.nn as nn
import pandas as pd
from RobustX.robustx.lib.tasks.ClassificationTask import ClassificationTask
from RobustX.robustx.generators.CEGenerator import CEGenerator
from RobustX.robustx.lib.models.BaseModel import BaseModel
from typing import Union, Dict, Callable, Any
from collections.abc import Iterable


class CostLoss(nn.Module):
    """
    Custom loss function to calculate the absolute difference between two tensors.

    Inherits from nn.Module.
    """

    def __init__(self):
        """
        Initializes the CostLoss module.
        """
        super(CostLoss, self).__init__()

    def forward(self, x1, x2):
        """
        Computes the forward pass of the loss function.

        @param x1: The first tensor (e.g., the original instance).
        @param x2: The second tensor (e.g., the counterfactual instance).
        @return: The absolute difference between x1 and x2.
        """
        dist = torch.abs(x1 - x2)
        return dist


class WachterMultiTarget(CEGenerator):

    def _generation_method(
        self,
        instance: pd.DataFrame,
        target_classes: Dict[Any, int],   # {target_name: target_value}
        lamb: float = 0.1,
        lr: float = 0.02,
        max_iter: int = 1000,
        epsilon: float = 1e-3,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        immutable_features: Iterable[str] = [],
        project_to_range: bool = False,
        permitted_ranges: Dict[Any, Iterable] = None,
        verbose: bool = False,
        **kwargs
    ):

        # Immutable feature indices
        immutable_indices = [instance.index.get_loc(f) for f in immutable_features]

        if permitted_ranges is not None:
            permitted_ranges_with_idx = {
                instance.index.get_loc(k): v for k, v in permitted_ranges.items()
            }

        # Initial CF
        x = torch.tensor(instance.to_numpy(), dtype=torch.float, device=device)
        if len(x.shape) < 2:
            x = x.unsqueeze(0)

        wac = x.detach().clone().requires_grad_(True)

        optimiser = torch.optim.Adam([wac], lr, amsgrad=True)

        validity_loss = torch.nn.BCELoss()
        cost_loss = CostLoss()

        iterations = 0
        cf_is_valid = False

        while not cf_is_valid and iterations <= max_iter:

            optimiser.zero_grad()

            preds = self.task.model.predict_ensemble_proba_tensor(wac)

            total_validity_loss = 0.0
            satisfied = True

            # ----------- Handle ensemble outputs -----------
            if isinstance(preds, torch.Tensor):
                # single target ensemble
                target_val = list(target_classes.values())[0]

                if preds.shape[2] >= 2:
                    class_prob = preds[:, :, target_val]
                else:
                    class_prob = preds.squeeze(2)

                avg_prob = torch.mean(class_prob, dim=0)

                y_target = torch.tensor([target_val], device=device, dtype=torch.float)

                total_validity_loss += validity_loss(avg_prob, y_target)

                p = avg_prob.item()
                if not ((target_val == 1 and p >= 0.5 - epsilon) or
                        (target_val == 0 and p < 0.5 + epsilon)):
                    satisfied = False

            else:
                # multi-target ensemble dict
                for target_key, target_val in target_classes.items():

                    target_preds = preds[target_key]

                    if target_preds.shape[2] >= 2:
                        class_prob = target_preds[:, :, target_val]
                    else:
                        class_prob = target_preds.squeeze(2)

                    avg_prob = torch.mean(class_prob, dim=0)

                    y_target = torch.tensor([target_val], device=device, dtype=torch.float)

                    total_validity_loss += validity_loss(avg_prob, y_target)

                    p = avg_prob.item()
                    if not ((target_val == 1 and p >= 0.5 - epsilon) or
                            (target_val == 0 and p < 0.5 + epsilon)):
                        satisfied = False

            # ----------- Distance loss -----------
            dist_loss = cost_loss(x, wac).mean()

            total_loss = total_validity_loss + lamb * dist_loss

            total_loss.backward()

            # Freeze immutable features
            if len(immutable_indices) > 0:
                wac.grad[:, immutable_indices] = 0

            optimiser.step()

            # Project to permitted ranges
            if project_to_range:
                with torch.no_grad():
                    for feat_idx, (fmin, fmax) in permitted_ranges_with_idx.items():
                        wac[:, feat_idx].clamp_(min=fmin, max=fmax)

            if satisfied:
                cf_is_valid = True

            if verbose:
                print(f"Iter {iterations:04d} | Loss {total_loss.item():.6f}")

            iterations += 1

        if not cf_is_valid:
            print("Warning: Wachter multi-target did not converge.")

        res = pd.DataFrame(wac.detach().cpu().numpy(), columns=instance.index)
        return res
