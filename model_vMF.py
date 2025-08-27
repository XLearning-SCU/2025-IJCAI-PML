import numpy as np
import torch
import torch.nn as nn
from vmf_loss import vMFLoss
from pytorch_pretrained_bert.modeling import BertModel
import torchvision
import torch.nn.functional as F

        
class AudioEncoder(nn.Module):
    def __init__(self, args):
        super(AudioEncoder, self).__init__()
        model = torchvision.models.resnet18(False)
        modules = list(model.children())[:-2]
        self.model = nn.Sequential(*modules)

        pool_func = (
            nn.AdaptiveAvgPool2d
            if args.img_embed_pool_type == "avg"
            else nn.AdaptiveMaxPool2d
        )

        if args.num_image_embeds in [1, 2, 3, 5, 7]:
            self.pool = pool_func((args.num_image_embeds, 1))
        elif args.num_image_embeds == 4:
            self.pool = pool_func((2, 2))
        elif args.num_image_embeds == 6:
            self.pool = pool_func((3, 2))
        elif args.num_image_embeds == 8:
            self.pool = pool_func((4, 2))
        elif args.num_image_embeds == 9:
            self.pool = pool_func((3, 3))
    
    def forward(self, x):
        x = x.unsqueeze(1).expand(-1, 3, -1, -1)
        a = self.pool(self.model(x))
        a = torch.flatten(a, 1)
        return a.squeeze()
    
class TextEncoder(nn.Module):
    def __init__(self):
        super(TextEncoder, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')

    def forward(self, inputs):
        txt, segment, mask = inputs
        _, out = self.bert(
            txt,
            token_type_ids=segment,
            attention_mask=mask,
            output_all_encoded_layers=False,
        )
        return out


class ImageEncoder(nn.Module):
    def __init__(self, args, pretrained=True):
        super(ImageEncoder, self).__init__()
        self.args = args
        model = torchvision.models.__dict__[args.arch](pretrained)
        modules = list(model.children())[:-2]
        self.model = nn.Sequential(*modules)

        pool_func = (
            nn.AdaptiveAvgPool2d
            if args.img_embed_pool_type == "avg"
            else nn.AdaptiveMaxPool2d
        )

        if args.num_image_embeds in [1, 2, 3, 5, 7]:
            self.pool = pool_func((args.num_image_embeds, 1))
        elif args.num_image_embeds == 4:
            self.pool = pool_func((2, 2))
        elif args.num_image_embeds == 6:
            self.pool = pool_func((3, 2))
        elif args.num_image_embeds == 8:
            self.pool = pool_func((4, 2))
        elif args.num_image_embeds == 9:
            self.pool = pool_func((3, 3))

    def forward(self, x):
        out = self.pool(self.model(x))
        out = torch.flatten(out, start_dim=2)
        out = out.squeeze()
        return out  # BxNx2048


class LogitCollector(nn.Module):
    def __init__(self, input_dim, mid_dim, beta, num_layer1=2, num_layer2=3):
        super(LogitCollector, self).__init__()
        
        if num_layer1 > 0:
            self.net = nn.Sequential(*([nn.Linear(input_dim, mid_dim),
                                        nn.BatchNorm1d(mid_dim),
                                        nn.ReLU(True),] + 
                                       [nn.Linear(mid_dim, mid_dim),
                                        nn.BatchNorm1d(mid_dim),
                                        nn.ReLU(True),] * (num_layer1 - 1)))
            input_dim = mid_dim
        
        if num_layer2 <= 1:
            self.kappa = nn.Sequential(nn.Linear(input_dim, 1), 
                                        nn.Softplus(),)
            self.mean = nn.Sequential(nn.Linear(input_dim, mid_dim), )
        elif num_layer2 == 2:
            self.kappa = nn.Sequential(nn.Linear(input_dim, mid_dim),
                                        nn.BatchNorm1d(mid_dim),
                                        nn.ReLU(True),
                                        nn.Linear(mid_dim, 1), 
                                        nn.Softplus(),)
            self.mean = nn.Sequential(nn.Linear(input_dim, mid_dim),
                                        nn.BatchNorm1d(mid_dim),
                                        nn.ReLU(True),
                                        nn.Linear(mid_dim, mid_dim), )
        else:
            self.kappa = nn.Sequential(*([nn.Linear(input_dim, mid_dim),
                                          nn.BatchNorm1d(mid_dim),
                                          nn.ReLU(True),] + 
                                         [nn.Dropout(0.5),
                                          nn.Linear(mid_dim, mid_dim),
                                          nn.BatchNorm1d(mid_dim),
                                          nn.ReLU(True),] * (num_layer2 - 2) +
                                         [nn.Linear(mid_dim, 1),
                                          nn.Softplus(),]))
            self.mean = nn.Sequential(*([nn.Linear(input_dim, mid_dim),
                                          nn.BatchNorm1d(mid_dim),
                                          nn.ReLU(True),] + 
                                         [nn.Dropout(0.5),
                                          nn.Linear(mid_dim, mid_dim),
                                          nn.BatchNorm1d(mid_dim),
                                          nn.ReLU(True),] * (num_layer2 - 2) +
                                         [nn.Linear(mid_dim, mid_dim),]))
        self.feature_dim = mid_dim
        self.beta = beta
        threshold = torch.tensor(20.)
        self.register_buffer('threshold', threshold)
        self.momentum = 1e-4
      
    def forward(self, x):
        if hasattr(self, 'net'):
            x = self.net(x)
        mean, kappa = self.mean(x), 1 / self.kappa(x)
        if not self.training:
            kappa[kappa > self.threshold] = (self.threshold * 1.5 - kappa[kappa > self.threshold]).clamp_min(1e-5)
            pass
        else:
            self.threshold = (1. - self.momentum) * self.threshold + self.momentum * kappa.max().item()
        return F.normalize(mean, 2, dim=-1), kappa #* self.beta

class vMFModel(nn.Module):
    def __init__(self, args):
        super(vMFModel, self).__init__()
        self.num_views = args.num_views
        self.num_classes = args.num_classes
        self.dims = np.reshape(args.dims, [-1])
        self.mid_dim = args.mid_dim
        self.beta = args.mid_dim * args.beta
        self.args = args
        if args.backbone:
            if args.dataset == 'CREMA-D' or args.dataset == 'avsbench':
                self.img_encoder = ImageEncoder(args, pretrained=False)
                self.img_logit = nn.Sequential(self.img_encoder, LogitCollector(args.dims[0], args.mid_dim, self.beta, num_layer1=0, num_layer2=2))
                
                self.audio_encoder = AudioEncoder(args)
                self.audio_logit = nn.Sequential(self.audio_encoder, LogitCollector(args.dims[1], args.mid_dim, self.beta, num_layer1=0, num_layer2=2))
                
                self.LogitCollectors = [self.img_logit, self.audio_logit]
            elif 'food101' == args.dataset:
                self.img_backbone = ImageEncoder(args)
                self.img_logit = nn.Sequential(self.img_backbone, LogitCollector(args.dims[0], args.mid_dim, self.beta, num_layer1=1, num_layer2=2))
                
                self.txt_backbone = TextEncoder()
                self.txt_logit = nn.Sequential(self.txt_backbone, LogitCollector(args.dims[1], args.mid_dim, self.beta, num_layer1=1, num_layer2=2))
                
                self.LogitCollectors = [self.img_logit, self.txt_logit]
                self.Backbones = [self.img_backbone, self.txt_backbone]
            elif 'MVSA_Single' == args.dataset:
                self.img_backbone = ImageEncoder(args)
                self.img_logit = nn.Sequential(self.img_backbone, LogitCollector(args.dims[0], args.mid_dim, self.beta, num_layer1=2, num_layer2=4))
                
                self.txt_backbone = TextEncoder()
                self.txt_logit = nn.Sequential(self.txt_backbone, LogitCollector(args.dims[1], args.mid_dim, self.beta, num_layer1=2, num_layer2=4))
                
                self.LogitCollectors = [self.img_logit, self.txt_logit]
                self.Backbones = [self.img_backbone, self.txt_backbone]
            else:
                self.img_backbone = ImageEncoder(args)
                self.img_logit = nn.Sequential(self.img_backbone, LogitCollector(args.dims[0], args.mid_dim, self.beta, num_layer1=2, num_layer2=3))
                
                self.txt_backbone = TextEncoder()
                self.txt_logit = nn.Sequential(self.txt_backbone, LogitCollector(args.dims[1], args.mid_dim, self.beta, num_layer1=2, num_layer2=3))
                
                self.LogitCollectors = [self.img_logit, self.txt_logit]
                self.Backbones = [self.img_backbone, self.txt_backbone]
            
        else:
            self.LogitCollectors = nn.ModuleList([LogitCollector(self.dims[i], args.mid_dim, self.beta, num_layer1=2, num_layer2=args.num_layer) for i in range(self.num_views)])

        self.classifier_vmf = vMFLoss(self.mid_dim * self.num_views, self.num_classes)
    
    def forward(self, data_list, choice='Train', ret=False):
        means = {}
        kappas = {}
        weights = {}

        cb_list = []
        for view in range(self.num_views):
            means[view], kappas[view] = self.LogitCollectors[view](data_list[view])
            
            weights[view] = kappas[view]
            cb_list.append(weights[view])

        w_all = (torch.concat(cb_list, dim=-1))
        w_all = w_all / w_all.sum(-1, keepdim=True)

        logits = 0
        preds, means_w = {}, []
        weight = self.classifier_vmf.classifier.weight
        prototypes = []
        for view in range(self.num_views):
            w = w_all[:, view: view + 1]
            means_w.append(w * means[view])
            prototype = F.normalize(weight[:, view * self.mid_dim: (view + 1) * self.mid_dim], 2, dim=-1)
            preds[view] = self.classifier_vmf(means[view], kappas[view], prototype, norm=False)
            weights[view] = w
            prototypes.append(prototype)

        kappa = sum(kappas.values()) / self.num_views # sum([k.pow(2) for k in kappas.values()]).sqrt()
        feat = torch.cat(means_w, dim=-1)
        feat = feat / feat.norm(dim=-1, keepdim=True).detach()
        prototype = torch.cat(prototypes, dim=-1) / np.sqrt(self.num_views)
        logits = self.classifier_vmf(feat, kappa, prototype, norm=False)
        
        if not self.training:
            cross_entropy, means_w, self_entropy, pred_all = [], [], [], logits.softmax(-1)
            for view in range(self.num_views):
                pred = preds[view].softmax(-1)
                cross_entropy.append((pred * pred_all.clamp_min(1e-7).log()).sum(-1, keepdim=True))
                self_entropy.append((pred * pred.clamp_min(1e-7).log()).sum(-1, keepdim=True))
            cross_entropy = torch.concat(cross_entropy, dim=-1).softmax(-1)
            self_entropy = torch.concat(self_entropy, dim=-1).softmax(-1)      
            rel = (cross_entropy + w_all) / 2.
            
            for view in range(self.num_views):
                w = rel[:, view: view + 1]
                means_w.append(w * means[view])

            feat = F.normalize(torch.cat(means_w, dim=-1), 2, dim=-1)
            logits = self.classifier_vmf(feat, kappa, prototype, norm=False)
            
            kappas = (torch.concat(cb_list, dim=-1))
            if ret:
                return logits, preds, weights, kappas, self_entropy, cross_entropy, rel
        
        return logits, preds, weights
