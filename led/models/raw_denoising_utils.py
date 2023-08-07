import torch
from torch import nn

class IlluminanceCorrect(nn.Module):
    def __init__(self):
        super(IlluminanceCorrect, self).__init__()

    # Illuminance Correction
    def forward(self, predict, source):
        if predict.shape[0] != 1:
            output = torch.zeros_like(predict)
            if source.shape[0] != 1:
                for i in range(predict.shape[0]):
                    output[i:i+1, ...] = self.correct(predict[i:i+1, ...], source[i:i+1, ...])
            else:
                for i in range(predict.shape[0]):
                    output[i:i+1, ...] = self.correct(predict[i:i+1, ...], source)
        else:
            output = self.correct(predict, source)
        return output

    def correct(self, predict, source):
        N, C, H, W = predict.shape
        predict = torch.clamp(predict, 0, 1)
        assert N == 1
        output = torch.zeros_like(predict, device=predict.device)
        pred_c = predict[source != 1]
        source_c = source[source != 1]

        num = torch.dot(pred_c, source_c)
        den = torch.dot(pred_c, pred_c)
        output = num / den * predict

        return output
