#1/usr/bin/env python2

import sys
import numpy as np

from scikits.learn import datasets
def precision(y_true, y_pred):
    true_pos = np.sum(y_true[y_pred == 1]==1)
    false_pos = np.sum(y_true[y_pred == 1]==0)
    return true_pos / float(true_pos + false_pos)
def recall(y_true, y_pred):
    true_pos = np.sum(y_true[y_pred == 1]==1)
    false_neg = np.sum(y_true[y_pred == 0]==1)
    return true_pos / float(true_pos + false_neg)



import label_propagation

def main(argv):
    digits = datasets.load_digits()

    # generate samples for dataset
    num_samples = len(digits.images)

    # train svm with all data
    from scikits.learn import svm
    svc = svm.SVC(probability=True)
    svc.fit(digits.data[:num_samples * 3.0/4], digits.target[:num_samples * 3.0 / 4])

    #probs_ = svc.predict_proba(digits.data)
    preds = svc.predict(digits.data[num_samples * 3.0/4:])

    print "precision/ recall for SVM with 0.75 pct data"
    prec = precision(digits.target, preds)
    rec = recall(digits.target, preds)
    #cm = confusion_matrix(digits.target, preds)
    print prec, rec

    # train svm with 50% data
    svc_50 = svm.SVC(probability=True)
    svc_50.fit(digits.data[:num_samples/2], digits.target[:num_samples/2])

    #probs_ = svc_50.predict_proba(digits.data[num_samples/2:])
    preds = svc.predict(digits.data[num_samples/2:])

    print "precision / recall for SVM with 0.50 pct data"
    prec = precision(digits.target, preds)
    rec = recall(digits.target, preds)
    #cm = confusion_matrix(digits.target, preds)
    print prec, rec

    # train label propagation with 20% data
    lp_50 = label_propagation.LabelPropagation()
    ma = []
    for t in digits.target:
        b = [0 for i in xrange(10)]
        b[t] = 1
        ma.append(b)
    dts = np.matrix(ma)

    lp_50.fit(digits.data,dts[:num_samples*0.2])

    preds = []
    for y in lp_50.Y[num_samples*0.2:]:
        preds.append(np.argmax(y))
    preds = np.array(preds)

    print "precision / recall for Label propagation with all data & 0.2 pct labels"
    prec = precision(digits.target[num_samples*0.2:], preds)
    rec = recall(digits.target[num_samples*0.2:], preds)
    print prec, rec

    ls = label_propagation.LabelSpreading()

    ls.fit(digits.data,dts[:num_samples*0.2])

    preds = []
    for y in ls.Y[num_samples*0.2:]:
        preds.append(np.argmax(y))
    preds = np.array(preds)

    print "precision / recall for Label Spreading with all data & 0.2 pct labels"
    prec = precision(digits.target[num_samples*0.2:], preds)
    rec = recall(digits.target[num_samples*0.2:], preds)
    print prec, rec



if __name__ == '__main__':
    main(sys.argv)
