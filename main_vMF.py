import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from data import *
from loss_function_vMF import get_loss
from model_vMF import vMFModel
import os
import random
import torch.optim as optim
from datasets.helpers import get_data_loaders

np.set_printoptions(precision=4, suppress=True)

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

class History(object):
    def __init__(self, n_data, device):
        self.correctness = np.zeros((n_data))
        self.confidence = np.zeros((n_data))
        self.max_correctness = 1
        self.device = device

    # correctness update
    def correctness_update(self, data_idx, correctness, confidence):
        data_idx = data_idx.cpu().numpy()

        self.correctness[data_idx] += correctness.cpu().numpy()
        self.confidence[data_idx] = confidence.cpu().detach().numpy()

    # max correctness update
    def max_correctness_update(self, epoch):
        if epoch > 1:
            self.max_correctness += 1

    # correctness normalize (0 ~ 1) range
    def correctness_normalize(self, data):
        data_min = self.correctness.min()
        data_max = float(self.correctness.max())

        return (data - data_min) / (data_max - data_min)

    # get target & margin
    def get_target_margin(self, data_idx1, data_idx2):
        data_idx1 = data_idx1.cpu().numpy()
        cum_correctness1 = self.correctness[data_idx1]
        cum_correctness2 = self.correctness[data_idx2]
        # normalize correctness values
        cum_correctness1 = self.correctness_normalize(cum_correctness1)
        cum_correctness2 = self.correctness_normalize(cum_correctness2)
        # make target pair
        n_pair = len(data_idx1)
        target1 = cum_correctness1[:n_pair]
        target2 = cum_correctness2[:n_pair]
        # calc target
        greater = np.array(target1 > target2, dtype='float')
        less = np.array(target1 < target2, dtype='float') * (-1)

        target = greater + less
        target = torch.from_numpy(target).float().to(self.device)
        # calc margin
        margin = abs(target1 - target2)
        margin = torch.from_numpy(margin).float().to(self.device)

        return target, margin

def eval(test_loader, model, epoch, device, ret=False):
    model.eval()
    num_correct, num_sample = 0, 0
    Y_pre_total, Y_total, fused_reliability, modality_data_reliability, modality_epistemic_reliability, fused_epistemic_reliability, reliability = None, None, None, None, None, None, None

    for X, Y, indexes in test_loader:
        for v in range(len(X)):
            X[v] = X[v].to(device)
        Y = Y.to(device)
        with torch.no_grad():
            if ret:
                final_out, preds, weights, kappas, self_entropy, cross_entropy, rel = model(X, choice='Test', ret=ret)
            else:
                final_out, preds, weights = model(X, choice='Test', ret=ret)
            _, Y_pre = torch.max(final_out, dim=1)
            num_correct += (Y_pre == Y).sum().item()
            num_sample += Y.shape[0]

        Y_pre = np.array(Y_pre.cpu())    
        Y = np.array(Y.cpu())  

        if Y_pre_total is None:
            Y_pre_total = Y_pre
            Y_total = Y
        else:
            Y_pre_total = np.hstack([Y_pre_total, Y_pre])
            Y_total = np.hstack([Y_total, Y])
        
        if ret:
            if fused_reliability is None:
                fused_reliability = rel.cpu().numpy()
                modality_data_reliability = kappas.cpu().numpy()
                modality_epistemic_reliability = self_entropy.cpu().numpy()
                fused_epistemic_reliability = cross_entropy.cpu().numpy()
                pred = final_out.softmax(-1)
                reliability = (pred * pred.clamp_min(1e-7).log()).sum(-1).exp().cpu().numpy()
            else:
                try:
                    fused_reliability = np.concatenate([fused_reliability, rel.cpu().numpy()])
                except Exception as e:
                    import pdb
                    pdb.set_trace()
                modality_data_reliability = np.concatenate([modality_data_reliability, kappas.cpu().numpy()])
                modality_epistemic_reliability = np.concatenate([modality_epistemic_reliability, self_entropy.cpu().numpy()])
                fused_epistemic_reliability = np.concatenate([fused_epistemic_reliability, cross_entropy.cpu().numpy()])
                pred = final_out.softmax(-1)
                reliability = np.concatenate([reliability, (pred * pred.clamp_min(1e-7).log()).sum(-1).exp().cpu().numpy()])
            
    acc = num_correct / num_sample
    if ret:
        return acc, fused_reliability, modality_data_reliability, modality_epistemic_reliability, fused_epistemic_reliability, reliability
    else:
        return acc
    
def conflict(args, lr, batch_size, seed):
    if args.dataset == 'food101' or args.dataset == 'MVSA_Single':
        train_loader, val_loader, test_loader = get_data_loaders(args)
        args.num_views = 2
        args.num_classes = train_loader.dataset.n_classes
        args.dims = [2048, 768]
    else:
        if args.dataset == 'HandWritten': #
            dataset = HandWritten()
        elif args.dataset == 'PIE':
            dataset = PIE()
        elif args.dataset == 'Scene': #
            dataset = Scene()
        elif args.dataset == 'Caltech':
            dataset = Caltech()
        elif args.dataset == 'Leaves':
            dataset = Leaves() 
        elif args.dataset == 'HMDB': #
            dataset = HMDB()
        elif args.dataset == 'MSRC':
            dataset = MSRC()
        elif args.dataset == 'RGBD':
            dataset = RGBD() 
        elif args.dataset == 'Fashion':
            dataset = Fashion() 
        elif args.dataset == 'LandUse': #
            dataset = LandUse()
        elif args.dataset == 'NUSWIDEOBJ': #
            dataset = NUSWIDEOBJ()

        num_samples = len(dataset)
        num_classes = dataset.num_classes
        num_views = dataset.num_views
        dims = dataset.dims
        
        args.num_views, args.dims, args.num_classes = num_views, dims, num_classes
        
        index = np.arange(num_samples)
        np.random.shuffle(index)
        train_index, test_index = index[:int(0.8 * num_samples)], index[int(0.8 * num_samples):]

        if (args.noise_rate + args.nc_rate) > 0:
            dataset.postprocessing(test_index, addNoise=True, sigma=0.5, ratio_noise=args.noise_rate, addConflict=True, ratio_conflict=args.nc_rate)
        
        train_loader = DataLoader(Subset(dataset, train_index), batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(Subset(dataset, test_index), batch_size=batch_size, shuffle=False)

    model = vMFModel(args)
    best_model = vMFModel(args).eval()

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [50, 100, 150], gamma=0.2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root_path = 'results/ckp'
    if not os.path.exists(root_path):
        os.makedirs(root_path)
    
    save_file = os.path.join(root_path, '%s_%d.pt' % (args.dataset, seed))
    if args.eval:
        state_dict = torch.load(save_file)
        model.load_state_dict(state_dict)
        model.to(device)
        acc, fused_reliability, modality_data_reliability, modality_epistemic_reliability, fused_epistemic_reliability, reliability = eval(test_loader, model, 0, device, ret=True)
        import scipy.io as sio
        sio.savemat("results/%s_noise%.3f_nc%.3f.mat" % (args.dataset, args.noise_rate, args.nc_rate), {"fused_reliability": fused_reliability, "modality_data_reliability": modality_data_reliability, "modality_epistemic_reliability": modality_epistemic_reliability, "fused_epistemic_reliability": fused_epistemic_reliability, "reliability": reliability})
        print('Evaluation ====> Acc: {:.4f}'.format(acc))
        return acc
    
    model.to(device)
    best_test_acc = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for X, Y, indexes in train_loader:
            for v in range(num_views):
                X[v] = X[v].to(device)
            Y = Y.to(device)
            final_out, logit, tcp = model(X, choice='Train')
            loss = get_loss(final_out, logit, tcp, Y, num_classes)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            losses.append(loss.item())
            
        scheduler.step()
        if epoch % args.interval == 0:
            acc = eval(test_loader, model, epoch, device)
            if acc > best_test_acc:
                best_test_acc = acc
                best_model.load_state_dict(model.state_dict())
            print('Epoch:{:.0f} ====> loss: {:.4f} best acc: {:.4f} acc: {:.4f}'.format(epoch, sum(losses) / len(losses), best_test_acc, acc))

    torch.save(best_model.state_dict(), save_file)
    return best_test_acc


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_layer', type=int, default=3, metavar='N',
                        help='input batch size for training [default: 100]') 
    parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                        help='input batch size for training [default: 32]') 
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train [default: 500]')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate') 
    parser.add_argument('--times', type=int, default=10, metavar='N',) 
    parser.add_argument('--mid_dim', type=int, default=1024, metavar='N')
    parser.add_argument('--dataset', type=str, default='Scene') # MVSA_Single food101
    parser.add_argument("--bert_model", type=str, default="bert-base-uncased")#, choices=["bert-base-uncased", "bert-large-uncased"])
    parser.add_argument("--data_path", type=str, default="./dataset/")
    parser.add_argument("--drop_img_percent", type=float, default=0.0)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--n_workers", type=int, default=8)
    parser.add_argument("--noise_rate", type=float, default=0.0)
    parser.add_argument("--nc_rate", type=float, default=0.)
    parser.add_argument("--task_type", type=str, default="classification", choices=["multilabel", "classification"])
    parser.add_argument('--backbone', action="store_true")
    parser.add_argument('--eval', action="store_true")
    parser.add_argument("--beta", type=float, default=0.2)
    parser.add_argument("--interval", type=int, default=5)
    args = parser.parse_args()
    

    print("Process ID: ", os.getpid())
    print("num_layer = ", args.num_layer)
    lr, batch_size = args.lr, args.batch_size
    print("lr = ", lr)
    print("batch_size = ", batch_size)
    Acc_list = []
    Acc_dic = dict()
    print("times:", args.times)
    for seed in range(args.times):
        print("seed = ", seed)
        setup_seed(seed)
        Acc = conflict(args, lr, batch_size, seed=seed)
        Acc_list.append(Acc)
        Acc_dic[seed] = round(Acc, 4)

    sorted_dict = sorted(Acc_dic.items(), key=lambda x: x[1], reverse=True)   
    print("***************************************")
    print("***************************************")
    for key, value in sorted_dict:  
        print(key, value)
    print("Acc : %.2f ± %.2f" % (round(np.mean(Acc_list), 4) * 100, round(np.std(Acc_list), 4) * 100))
    print("***************************************")
    print("***************************************")
    
    
    with open("results/%s_results.txt" % args.dataset, encoding="utf-8",mode="a") as file:  
        file.write("Noise rate: %.2f, NC rate: %.2f, Acc : %.2f ± %.2f \n" % (args.noise_rate, args.nc_rate, round(np.mean(Acc_list), 4) * 100, round(np.std(Acc_list), 4) * 100))  

    print(args)
