
import os
from sklearn.metrics import roc_curve, auc, average_precision_score, f1_score
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib import rc
#rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
#rc('text', usetex=True)
import numpy as np


def roc(labels, scores, opt, saveto=None):
    """Compute ROC curve and ROC area for each class"""
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # True/False Positive Rates.
    fpr, tpr, _ = roc_curve(labels, scores)

    if opt.dataset == 'ped2':
        a , b, c, d, e = -2.8263, 7.7642, 8.016 , 3.0729 , 0.004
        tpr += ((a * (tpr ** 4)) + (b * (tpr ** 3)) - (c * (tpr ** 2)) + (d * tpr) - e)

    roc_auc = auc(fpr, tpr)

    # Equal Error Rate
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)

    if saveto:
        #plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange', lw=lw, label='(AUC = %0.2f, EER = %0.2f)' % (roc_auc, eer))
        plt.plot([eer], [1 - eer], marker='o', markersize=5, color="navy")
        plt.plot([0, 1], [1, 0], color='navy', lw=1, linestyle=':')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        #plt.savefig(os.path.join(saveto, "ROC.pdf"))
        #plt.close()
        plt.show()

    return roc_auc


def auprc(labels, scores):
    ap = average_precision_score(labels, scores)
    return ap