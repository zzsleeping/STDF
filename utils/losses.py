import torch
import torch.nn as nn
import torch.nn.functional as F


class GuidanceLoss(nn.Module):
    """STGF training objective (Eq. 6).

        L = MAE_B(S, Y) + lambda * MSE_{B_T}(S, sg(T)) + beta * MAE_{B_T}(S, sg(T))

    S: student forecast, Y: observed future, T: precomputed teacher forecast.
    The supervised term covers every window of the minibatch B; both guidance
    terms are averaged over the subset B_T whose forecast origin has enough
    history for the teacher, and vanish when B_T is empty.
    """

    def __init__(self, con_w=0.4, rob_w=0.4):
        super().__init__()
        self.con_w, self.rob_w = con_w, rob_w

    def forward(self, outputs, batch_y, teacher=None, valid=None):
        sup = F.l1_loss(outputs, batch_y)
        con = rob = outputs.new_zeros(())
        if teacher is not None and (self.con_w or self.rob_w) and bool(valid.any()):
            student, target = outputs[valid], teacher[valid].detach()
            con, rob = F.mse_loss(student, target), F.l1_loss(student, target)
        loss = sup + self.con_w * con + self.rob_w * rob
        return loss, {'sup': sup.detach(), 'con': con.detach(), 'rob': rob.detach()}
