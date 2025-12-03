dependencies = ['torch']

import torch
import network

class SimpleArgs:
    def __init__(self, **kwargs):
        self.backbone = kwargs.pop('backbone', 'dinov2')
        self.aggregator = kwargs.pop('aggregator', None)
      
        self.num_learnable_aggregation_tokens = kwargs.pop('num_learnable_aggregation_tokens', 8)
        self.freeze_te = kwargs.pop('freeze_te', 8)

        self.resume = kwargs.pop("resume", True)
        self.foundation_model_path = kwargs.pop('foundation_model_path', None)
        
        for key, value in kwargs.items():
            setattr(self, key, value)

def ImAge(training_set="Merged", **kwargs):
    args = SimpleArgs(**kwargs)
    model = network.VPRmodel(args)
    model = torch.nn.DataParallel(model)
    if training_set == "Merged":
      model.load_state_dict(
          torch.hub.load_state_dict_from_url(f'https://github.com/Lu-Feng/ImAge/releases/download/v1.0.0/ImAge_Merged.pth', map_location=torch.device('cpu'))["model_state_dict"]
      )
    elif training_set == "GSV_Cities":
      model.load_state_dict(
          torch.hub.load_state_dict_from_url(f'https://github.com/Lu-Feng/ImAge/releases/download/v1.0.0/ImAge_GSV.pth', map_location=torch.device('cpu'))["model_state_dict"]
      )
    return model
