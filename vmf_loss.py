import torch
import torch.nn.functional as F
import numpy as np
import mpmath

def estimate_kappa(model, dataloader, args, device):
    model.eval()
    all_means, classes, Ys = [torch.zeros((args.num_classes, args.mid_dim)).to(device) for _ in range(args.num_views)], torch.arange(args.num_classes).to(device).view([-1, 1]), []
    for batch in dataloader:
        if args.dataset == 'food101' or args.dataset == 'MVSA_Single':
            txt, segment, mask, img, Y, idx = batch
            X = [img.to(device), (txt.to(device), segment.to(device), mask.to(device))] #  txt, mask, segment
            Y = Y.to(device)
        elif args.dataset == 'CREMA-D' or args.dataset == 'avsbench':
            spec, image, Y, idx = batch
            X = [image.to(device), spec.to(device)]
            Y = Y.to(device)
        else:
            X, Y, idx = batch
            X = [X[v].to(device) for v in range(len(X))]
            Y = Y.to(device)
        with torch.no_grad():
            final_out, logit, tcp, means = model(X, choice='Test', ret_feat=True)
            for i in range(len(means)):
                w = (classes == Y.view([1, -1])).float()
                all_means[i] += (w.mm(means[i]))
            Ys.append(Y)
    Y = torch.concat(Ys)
    kappa = []
    for i in range(args.num_views): 
        counts = Y.unique(return_counts=True)[1].view([-1, 1]).clamp_min(1e-7)
        R = all_means[i].norm(dim=-1, keepdim=True) / counts
        kappa.append(R * (args.mid_dim - R * R) / (1 - R))
        all_means[i] /=  counts
        
    return torch.concat(kappa).reshape([-1, 1]), torch.concat(all_means, dim=1)

def log_bessel_i(v, x, terms=50):
    """
    Compute the logarithm of the modified Bessel function of the first kind, log(I_v(x)),
    using PyTorch.

    Parameters:
    - v: Order of the Bessel function (scalar or tensor).
    - x: Input tensor.
    - terms: Number of terms in the series expansion.

    Returns:
    - log(I_v(x)): Logarithm of the modified Bessel function.
    """
    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.float32)
    if not torch.is_tensor(v):
        v = torch.tensor(v, dtype=torch.float32)
    
    # Use series expansion of I_v(x) in log-space
    x = x.unsqueeze(-1)  # Add dimension for broadcasting
    k = torch.arange(terms, dtype=torch.float32, device=x.device)

    # Log of coefficients
    log_coef = (2 * k + v) * torch.log(x / 2) - (torch.lgamma(k + 1) + torch.lgamma(k + v + 1))
    
    # Compute log-sum-exp for numerical stability
    max_log_coef = torch.max(log_coef, dim=-1, keepdim=True)[0]
    sum_exp_log = torch.exp(log_coef - max_log_coef).clamp_max(1e7).sum(dim=-1)
        
    log_bessel = max_log_coef.squeeze(-1) + torch.log(sum_exp_log)
    
    
    return log_bessel

def vmf_logpartition(d, kappa):
    '''
    Evaluates the log-partition log C_d(kappa) for vMF density.
    Inspired from: https://github.com/minyoungkim21/vmf-lib
    
    Parameters
    ----------
    d: scalar (> 0)
        Dimension in which the vMF density is computed.
    kappa: torch tensor (N,)
        Concentration parameter of the vMF density.

    Returns:
    --------
    logC: torch tensor (N,) 
        Log-partition of the vMF density : log C_d(kappa)
    '''

    # Fix mpmath precision
    # mpmath.dps = 50
    
    # with torch.no_grad():
    s = 0.5 * d - 1

    # log I_s(kappa)
    logI = log_bessel_i(s, kappa)
    # From mpmath to torch 
    
    if (logI != logI).sum().item() > 0:  # there is nan
        raise ValueError('NaN is detected from the output of log-besseli()')

    logC = -0.5 * d * np.log(2 * np.pi) + s * kappa.clamp_min(1e-7).log() - logI
    return logC#, logI

class vMFLoss(torch.nn.Module):

    def __init__(self, feature_dim, num_classes, kappa=None):
        '''

        Parameters
        ----------
        feature_dim: int
            Dimension of features to project on hypersphere.
        num_classes: int
            Number of different classes to learn.
        kappas: float or list of floats 
            float: Concentration parameter of vMF loss (unique value for
            all classes).
            list of floats: Concentration parameters of vMF loss.
            * if the length of the list == num_classes:
                The i-th element of the list is the concentration 
                parameter for the i-th label/class.
            * if the length of the list < num_classes:
                Each parameter is associated to a value taken by an 
                external attribute (given in labels_set thereafter). 
                The ordering of the input list must correspond to the
                ordering of attribute values: the i-th element of the 
                list kappas is the concentration parameter for labels
                having attribute value equal to the i-th element in the
                sorted list of attributes (given in labels_set 
                thereafter). Ex: each sample has a binary (0/1) attribute
                (provided in labels_set thereafter), kappas = [10, 20]
                -> labels for which the attribute is 0 have concentration
                parameter equal to 10, labels for which the attribute is
                1 have concentration parameter equal to 20.
        labels_set: None or tensor of shape (num_classes, 2)
            Only necessary if kappas is a list with more than 
            1 element and whose length < num_classes.
            tensor (num_classes, 2):
            * column 0: unique label values 0 -> num_classes-1.
            * column 1: corresponding attribute values.
        '''
        super(vMFLoss, self).__init__()

        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.classifier = torch.nn.Linear(feature_dim, num_classes)
        torch.nn.init.orthogonal_(self.classifier.weight)
        kappa_mu = torch.ones_like(self.classifier.bias.data) * 30.
        self.register_buffer('p_kappa', kappa_mu)
  
    def forward(self, mean, kappas, prototype=None, p_kappas=None, norm=True):
        '''

        Parameters
        ----------
        x: tensor shape (N, feature_dim)
        return_loss: boolean
            Set to True to return the loss w.r.t. the labels given after.
        labels: None or tensor shape (N,)
            Only used if return_loss is True.
            Labels used for supervised learning.

        Returns
        -------
        tuple (L, pred)
        L: None if labels is None.
        0d tensor otherwise
            vMF loss corresponding to x and labels.
        pred: torch shape (N,)
            Predicted class/label for x.
        '''
        # Normalize feature vectors, shape (Nxd)
        N = mean.shape[0]
        # assert mean.shape[1] == self.feature_dim
        if norm:
            mean = F.normalize(mean, p=2, dim=1)
        # assert mean.shape == (N,self.feature_dim)

        if prototype is None:
            prototype = self.classifier.weight
            
        # Shape (NxC)
        cos = mean.mm(prototype.t())
        kappas = kappas.view((-1, 1))
        # assert cos.shape == (N, self.num_classes)

        if norm:
            # Get norm of centroids  
            mu_norm = torch.norm(prototype, p=2, dim=1, keepdim=True).t()
            # assert mu_norm.shape == (1,self.num_classes)

            # Matrix of cosines between feature i and centroid j (NxC)
            cos = cos / mu_norm # broadcasting division in PyTorch 
            # assert cos.shape == (N, self.num_classes)
        # Ensure that we have real cosines
        cos = cos.clamp(min=-1.0, max=1.0)

        # Matrix of log C_d(kappa_j) + kappa_j cos theta_ij (NxC)
        logits = vmf_logpartition(self.feature_dim, kappas) + cos * kappas
        return logits

    def get_centroids(self):
        '''
        Return
        ------
        centroids: tensor shape (Cxd)
            Tensor that lists all centroids, so that they are ordered 
            by increasing corresponding label. Ex:
            | centroid label 0 |
            | centroid label 1 |
            |        .         |
            |        .         |
            |centroid label C-1|            
        '''
        #Normalize centroids
        with torch.no_grad():
            centroids = F.normalize(self.classifier.weight, p=2, dim=1)
            assert centroids.shape == (self.num_classes, self.feature_dim)

        return centroids.detach()
