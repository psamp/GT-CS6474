from sklearn.model_selection import cross_validate
from metrics import model_metrics
from sklearn.metrics import recall_score
from sklearn.metrics import make_scorer

specificity = make_scorer(recall_score, pos_label=0)


def run_classifier(classifier, X_train, X_test, y_train, y_true):
    svc = classifier.fit(X_train, y_train)

    scoring = {'accuracy': 'accuracy', 'balanced_accuracy': 'balanced_accuracy', 'precision': 'precision',
               'f1': 'f1', 'roc_auc': 'roc_auc', 'recall': 'recall', 'specificity': specificity}
    scores = cross_validate(svc, X_test, y_true,
                            scoring=scoring, cv=5, return_train_score=True)

    for score in scores:
        if "test" in score:
            vals = scores[score]
            print("%0.3f %s with a standard deviation of %0.3f" %
                  (vals.mean(), score, vals.std()))

    y_scores = svc.decision_function(X_test)
    y_pred = svc.predict(X_test)

    model_metrics(y_true, y_pred, y_scores)
