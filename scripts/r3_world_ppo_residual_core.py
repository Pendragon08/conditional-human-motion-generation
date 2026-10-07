"""A fixed linear physics predictor plus a trainable causal GRU residual.

Only encode(history) is deployed to the actor. Future actions and ridge output
are supervised decoder inputs and never enter a current actor feature.
"""
import torch
from torch import nn
from r3_world_ppo_core import WorldModel, HORIZONS, CONTINUOUS, TARGET


class ResidualWorldModel(WorldModel):
    def __init__(self, coefficients):
        super().__init__()
        for horizon in HORIZONS:
            self.register_buffer('ridge_'+str(horizon),torch.as_tensor(coefficients[horizon],dtype=torch.float32))
            # Exactly the frozen ridge at initialization, rather than a random
            # residual that initially makes the established predictor worse.
            nn.init.zeros_(self.heads[str(horizon)][-1].weight)
            nn.init.zeros_(self.heads[str(horizon)][-1].bias)

    def predict(self, history, future_actions, state_only=False, baseline_override=None):
        latent=self.encode(history)
        normalized=(history-self.input_mean)/self.input_std
        predictions={}
        for k in HORIZONS:
            action=torch.zeros_like(future_actions[:,:k]) if state_only else future_actions[:,:k]
            features=torch.cat([normalized.flatten(1),action.flatten(1),
                                torch.ones(len(history),1,device=history.device)],-1)
            baseline=(features@getattr(self,'ridge_'+str(k))) if baseline_override is None else baseline_override[k]
            # Residual continuous dynamics, separate contact logits. Contacts
            # retain proper BCE supervision instead of shuffling logit deltas.
            baseline=torch.cat([(baseline[:,:CONTINUOUS]-self.delta_mean)/self.delta_std,
                                torch.zeros(len(history),TARGET-CONTINUOUS,device=history.device)],-1)
            residual=self.heads[str(k)](torch.cat([latent,action.flatten(1)],-1))
            predictions[k]=baseline+residual
        return predictions
