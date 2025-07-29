import torch


class SimpleLoss:
    def __init__(self, model, params=None):
        if hasattr(model, 'module'):
            model = model.module
        self.device = next(model.parameters()).device

    def __call__(self, outputs, targets):
        x1, x2, x3 = outputs
        y1, y2, y3 = targets
        loss_fn = torch.nn.BCEWithLogitsLoss()
        loss = loss_fn(x1, y1) + loss_fn(x2, y2) + loss_fn(x3, y3)

        if not isinstance(loss, torch.Tensor):
            loss = torch.tensor(loss, device=self.device)
        if loss.ndim == 0:  # if loss is a scalar
            loss = loss.unsqueeze(0)  # add a batch dimension
        elif loss.ndim == 1:  # if loss is a vector
            loss = loss.unsqueeze(0)  # add a batch dimension
        elif loss.ndim > 1:  # if loss is a matrix or higher
            loss = loss.mean(dim=0, keepdim=True)  # average over the first dimension

        return loss