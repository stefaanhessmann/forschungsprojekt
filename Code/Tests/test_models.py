import torch
import pytest
from torch.nn import MSELoss
from torch.optim import Adam
from Code.Models.SchNet.schnet import SchNet
from Code.Models.DeepPotential.deep_potential import DeepPotential



def assert_params_changed(model, input, exclude=[]):
    """
    Check if all model-parameters are updated when training.

    Args:
        model (torch.nn.Module): model to test
        data (torch.utils.data.Dataset): input dataset
        exclude (list): layers that are not necessarily updated
    """
    # save state-dict
    torch.save(model.state_dict(), 'before')
    # do one training step
    optimizer = Adam(model.parameters())
    loss_fn = MSELoss()
    pred = model(input)
    loss = loss_fn(pred, torch.rand(pred.shape))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # check if all trainable parameters have changed
    after = model.state_dict()
    before = torch.load('before')
    for key in before.keys():
        if sum([key.startswith(exclude_layer) for exclude_layer in exclude]) != 0:
            continue
        assert (before[key] != after[key]).any(), 'Not all Parameters have been updated!'


def test_parameter_update_schnet():
    model = SchNet()
    in_data = torch.rand((4, 19, 19))
    assert_params_changed(model, in_data)

def test_parameter_updata_deeppotential():
    model = DeepPotential()
    in_data = torch.rand((4, 19, 72))
    assert_params_changed(model, in_data)
