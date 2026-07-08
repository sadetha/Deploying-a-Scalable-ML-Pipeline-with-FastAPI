# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
The model takes publicly available census data and classifies it using logistic regression to classify income as greater than $50K or not. 

## Intended Use
This model is intended to be used as a method for making predictions on income classification based on data that was pulled from the 1994 census database. 

## Training Data
Census data, which was gathered from the 1994 census database is utilized to train the model. The data is ingested and split into training and testing data. OneHotEncoder converts the training data text features into to numerical data to train the model as machine learning requires data to be numerical. The features from the census data are then used to train the model utilzing logistic regression. 

## Evaluation Data
Once training is completed, the testing data is processed through the model and is used to determine precision, recall, and F1. 

## Metrics
The metrics used in the model performance includes: 
* Precision: true positives divided by total positives - 72.17%
* Recall: models ability to correctly classify all positives - 26.03%
* F1: harmonic mean of precision and recall which is used to determine the models overall ability to classify the data as it balances both aspects together - 38.26%

Source: https://medium.com/@piyushkashyap045/understanding-precision-recall-and-f1-score-metrics-ea219b908093

## Ethical Considerations
While this model has been trained on census data, there is likely bias involved in the final model outcome. This is something that should be considered as the data is used to make future decisions. Furthermore, the demographic data could inadvertently risk the model making prejudice or discriminatory predictions.

## Caveats and Recommendations
Given the bias that can occur in machine learning, it is recommended that the data is cross referenced with a source for validation. While this can be used to make income predicitons, the model should not be used to make critical decisions.