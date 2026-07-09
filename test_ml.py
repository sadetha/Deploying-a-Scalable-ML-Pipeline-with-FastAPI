from pytest import approx
from ml.model import train_model, compute_model_metrics #importing for train_model and compute_model_metrics function 
from ml.data import apply_label
from sklearn.linear_model import LogisticRegression 
from sklearn.utils.validation import check_is_fitted
from sklearn.datasets import make_classification
# TODO: add necessary import
#testing source for assert: https://stackoverflow.com/questions/5142418/what-is-the-use-of-assert-in-python
# TODO: implement the first test. Change the function name and input as needed
def test_training_model():
    """
    # tests the training model to ensure the correct model is used
    """

    X_train, y_train = make_classification(random_state = 5) #generate random data, source: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html
    model_test = train_model(X_train, y_train) #training the model on random data
    assert isinstance(model_test,LogisticRegression) #asserting that the model test is truely LogisticRegression source: https://www.w3schools.com/python/ref_func_isinstance.asp and https://scikit-learn.org/stable/modules/generated/sklearn.utils.validation.check_is_fitted.html


# TODO: implement the second test. Change the function name and input as needed
def test_check_metrics():
    """
    # checking the precision, recall, and fbeta metrics
    """
    y = [1,1,0,0]
    preds = [1,0,1,0]
    #correct pos, false neg, false pos, corr neg
    #precision is true positives divided by total positives which should be 50% based on the sample data
    #recall is the ability to classiy all possitives which is 50% b/c only half of the positives were correctly classifed
    #fbeta is the harmonic mean of precsion and recall which would be 50% 
    #Source: https://medium.com/@piyushkashyap045/understanding-precision-recall-and-f1-score-metrics-ea219b908093
    precision, recall, fbeta = compute_model_metrics(y,preds) #calling check_metrics to sample data, setting equal to all three in the order of output
    assert precision == approx(0.5) #verifying that precision is equal to .5 as expected; source: https://docs.pytest.org/en/7.1.x/reference/reference.html#pytest-approx
    assert recall == approx(0.5) #verifying that recall is equal to .5 as expected
    assert fbeta == approx(0.5) #verifying that fbeta is equal to .5 as expected


# TODO: implement the third test. Change the function name and input as needed
def test_apply_label():
    """
    # verifying the apply_label functions as expected
    """
    assert apply_label([1]) == '>50K' #confirming labels function as expected
    assert apply_label([0]) == '<=50K'
    assert apply_label([2]) == None #confirm that a non-labeled value returns None