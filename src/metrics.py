import sklearn.metrics as sk_metrics
from sklearn.metrics import confusion_matrix

def model_metrics(y_true, y_pred, y_scores):
        accuracy = sk_metrics.accuracy_score(y_true, y_pred)
        balanced_accuracy = sk_metrics.balanced_accuracy_score(y_true, y_pred)
        precision = sk_metrics.precision_score(y_true, y_pred, average='weighted')
        f1 = sk_metrics.f1_score(y_true, y_pred)
        auc = sk_metrics.roc_auc_score(y_true, y_scores)
        
        # calc specificity and recall with confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        specificity = (tn / (tn + fp))
        recall = (tp / (tp + fn))
        
        print("ACCURACY 💖", accuracy)
        print("BALANCED ACCURACY 👀", balanced_accuracy)
        print("SPECIFICITY (TNR) 💀", specificity)
        print("RECALL (TPR) 👁", recall)
        print("PREC 📍", precision)
        print("F1 💧", f1)
        print("AUC 💙", auc)