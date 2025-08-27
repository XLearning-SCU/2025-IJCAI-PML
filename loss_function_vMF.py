import torch
import torch.nn.functional as F
import torch.nn as nn

def CCL(pred, targets, num_classes):
    logits = pred.softmax(1)
    loss = (logits.clamp_min(1e-7).log() * (1. - targets)).sum(1).mean()
    return loss

def get_loss(final_out, Logit, weight, target, num_classes):
    target = F.one_hot(target, num_classes=num_classes).float()
    
    crterion = nn.CrossEntropyLoss()
    clf_loss = crterion(final_out, target)
    criterion_r = nn.L1Loss()

    tcps = []
    for key in Logit.keys():
        pred = torch.nn.functional.softmax(Logit[key], dim=1)
        tcps.append((pred * pred.clamp_min(1e-7).log()).sum(-1, keepdim=True))
        clf_loss += crterion(Logit[key], target) / len(Logit)
    tcps = torch.concat(tcps, dim=-1).softmax(-1)
    tcp_pred_loss = 0
    for i in range(tcps.shape[-1]):
        tcp_pred_loss += criterion_r(weight[i], tcps[:, i: i + 1].detach())
    
    loss = clf_loss + tcp_pred_loss

    return loss