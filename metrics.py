import os
from sklearn.metrics import roc_curve, auc, average_precision_score, f1_score, confusion_matrix
import threading
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import numpy as np
from torch.nn import functional as F
import torch
import configs as cfg
tcfg = cfg.CONFIGS['Train']

def evaluate(labels, scores, metric, best_auc):
    if metric == 'roc':
        return roc(labels, scores, best_auc)
    elif metric == 'auprc':
        return auprc(labels, scores)
    elif metric == 'f1_score':
        threshold = 0.50
        scores[scores >= threshold] = 1
        scores[scores <  threshold] = 0
        return f1_score(labels.cpu(), scores.cpu())
    else:
        raise NotImplementedError("Check the evaluation metric.")

##
def roc(labels, scores, best_auc, saveto='./outputs', ):
    """Compute ROC curve and ROC area for each class"""
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    labels = labels.cpu()
    scores = scores.cpu()

    # True/False Positive Rates.
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    # Equal Error Rate
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    
    if saveto:
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange', lw=lw, label='(AUC = %0.3f, EER = %0.3f)' % (roc_auc, eer))
#         plt.plot([eer], [1-eer], marker='o', markersize=5, color="navy")
        plt.plot([0, 1], [1, 0], color='navy', lw=1, linestyle=':')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(saveto, "Current_Epoch_ROC.pdf"))
        if roc_auc>best_auc:
            plt.savefig(os.path.join(saveto, "Best_ROC.pdf"))
        plt.close()

    return roc_auc

def auprc(labels, scores):
    ap = average_precision_score(labels.cpu(), scores.cpu())
    return ap

class cal_distance(torch.nn.Module):
    def __init__(self):
        super(cal_distance, self).__init__()

    def forward(self, output1, output2):
        euclidean_distance = F.pairwise_distance(output1, output2,p=2)#L2 distance  
        return euclidean_distance

def confuse_matrix(score0, score1, lb0, lb1, raw_length, pc0_name, pc1_name):
    def cal_matrix(score, lb, length):
        threshold = 0.5
        lb = lb.detach().squeeze(0).cpu().numpy()
        score = np.array(score.detach().squeeze(0).cpu())
        if length < tcfg.n_samples:
            score = score[0:length]
            lb = lb[0:length]
        score[score<=threshold] = 0.0 
        score[score>threshold] = 1.0
        tp = np.sum(lb*score)
        fn = lb-score
        fn[fn<0]=0
        fn = np.sum(fn)
        tn = lb+score
        tn[tn>0]=-1
        tn[tn>=0]=1
        tn[tn<0]=0
        tn = np.sum(tn)
        fp = score - lb
        fp[fp<0] = 0
        fp = np.sum(fp)
        return tp, fp, tn, fn
    p0_raw_length, p1_raw_length = raw_length
    tp0, fp0, tn0, fn0 = cal_matrix(score0, lb0, p0_raw_length)
    tp1, fp1, tn1, fn1 = cal_matrix(score1, lb0, p1_raw_length)
    
    return tp0+tp1, fp0+fp1, tn0+tn1, fn0+fn1

def confuse_matrix2(score0, score1, lb0, lb1, raw_length, pc0_name, pc1_name):
    def cal_matrix(score, lb, length):
        threshold = 0.5
        lb = lb.detach().squeeze(0).cpu().numpy()
        score = np.array(score.detach().squeeze(0).cpu())
        if length < tcfg.n_samples:
            score = score[0:length]
            lb = lb[0:length]
        score[score<=threshold] = 0.0 
        score[score>threshold] = 1.0
        tp = np.sum(lb*score)
        fn = lb-score
        fn[fn<0]=0
        fn = np.sum(fn)
        tn = lb+score
        tn[tn>0]=-1
        tn[tn>=0]=1
        tn[tn<0]=0
        tn = np.sum(tn)
        fp = score - lb
        fp[fp<0] = 0
        fp = np.sum(fp)
        return tp, fp, tn, fn
    p0_raw_length, p1_raw_length = raw_length
    tp0, fp0, tn0, fn0 = cal_matrix(score0, lb0, p0_raw_length)
    tp1, fp1, tn1, fn1 = cal_matrix(score1, lb0, p1_raw_length)
    
    return tp0, fp0, tn0, fn0, tp1, fp1, tn1, fn1

def eva_metrics(TP, FP, TN, FN):
    precision = TP/(TP+FP+1e-8)
    oa = (TP+TN)/(TP+FN+TN+FP+1e-8)
    recall = TP/(TP+FN+1e-8)
    f1 = 2*precision*recall/(precision+recall+1e-8)
    iou = TP/(FN+TP+FP+1e-8)
    P = ((TP+FP)*(TP+FN)+(FN+TN)*(FP+TN))/((TP+TN+FP+FN)**2+1e-8)
    kappa = (oa-P)/(1-P+1e-8)
    results = {'iou':iou,'precision':precision,'oa':oa,'recall':recall,'f1':f1,'kappa':kappa}
    return results

class IoUCalculator:
    def __init__(self,):
        self.num_classes = 3
        self.gt_classes = [0 for _ in range(self.num_classes)]
        self.positive_classes = [0 for _ in range(self.num_classes)]
        self.true_positive_classes = [0 for _ in range(self.num_classes)]
        self.lock = threading.Lock()
        self.val_total_correct = 0
        self.val_total_seen = 0
        
    def add_data(self, logits0, logits1, labels0, labels1):
        pred0 = logits0.detach().cpu().numpy()
        pred1 = logits1.detach().cpu().numpy()
        pred_valid = np.hstack((pred0, pred1*2))
        
        lb0 = labels0.detach().cpu().numpy()
        lb1 = labels1.detach().cpu().numpy()
        labels_valid = np.hstack((lb0, lb1*2))
        
        correct = np.sum(pred_valid == labels_valid)
        self.val_total_correct += correct
        self.val_total_seen += len(labels_valid)

        conf_matrix = confusion_matrix(y_true=labels_valid, y_pred=pred_valid, labels=np.arange(0, self.num_classes, 1))
        self.lock.acquire()
        self.gt_classes += np.sum(conf_matrix, axis=1)
        self.positive_classes += np.sum(conf_matrix, axis=0)
        self.true_positive_classes += np.diagonal(conf_matrix)
        self.lock.release()

    def compute_iou(self):
        iou_list = []
        for n in range(0, self.num_classes, 1):
            if float(self.gt_classes[n] + self.positive_classes[n] - self.true_positive_classes[n]) != 0:
                iou = self.true_positive_classes[n] / \
                    float(self.gt_classes[n] + self.positive_classes[n] - self.true_positive_classes[n])
                iou_list.append(iou)
            else:
                iou_list.append(0.0)
        mean_iou = sum(iou_list) / float(self.num_classes)
        return mean_iou, iou_list
    
    def comput_oa(self):
        return self.val_total_correct/self.val_total_seen
    
    def metrics(self):
        mean_iou, iou_list = self.compute_iou()
        oa = self.comput_oa()
        return {'miou':mean_iou, 'iou_list':iou_list, 'oa':oa}   
