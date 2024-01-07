# Business Loan Creditworthiness Prediction using an Optimized Neural Network Model
# Abstract
This project aims to build an optimized multiple hidden layer neural network model to predict small buisness loan creditworthiness. In this project I perform heavy feature engineering and preprocessing, then using TensorFlow Keras I create an optimizing function that iteratively builds, trains, and tests different neural network models and returns the optimal model in accordance to various classification metrics. 

# Data Feature Engineering
## About the Data
The raw data used for this project is from the U.S. Small Business Administration (SBA) and was downloaded from Kaggle.com. The dataset include 27 features and 899,164 rows. The table below summarizes the features contents:

![image](https://github.com/wgemba/loan_default_prediction/assets/134420287/fe388d2e-0330-4293-9a83-6ac8b915fd54)

## Feature Engineering

The data set required significant preprocessing and feature engineering. Following several layers of preprocessing and engineering, including: transforming categorical features into numeric one, encoding large-set categorical features, removing class imbalance, removing null values, etc., the finalized dataset contains 81 features and 189,019 rows (see feature descriptions below).

![image](https://github.com/wgemba/loan_default_prediction/assets/134420287/04d8dba7-b2ce-45b9-a4fc-62e86212f367)

The full code for the feature processing can be seen in the file 'feature_engineering_and_visualization.ipynb'.

### Specific Packages Used for Preprocessing
Certain features required the use of special packages and functions to convert from categorical to numeric datatype. Due to the large set of unique values for the feature 'Zip', rather than encoding the attribute I used the package 'pgeocode'. The package 'pgeocode' is a Python library that queries geospatial data (e.g., GPS coordinates, region name, municipality name) from the GeoNames database, using a postal code as input. Using this function, I created two new attributes 'latitude' and 'longitude' inplace of the 'Zip' feature. 

The 'State' attribute and and the 'Industry' attribute (which was created based on a dictionary remapping of the NAICS attributed, using the official NAICS code dictionary) were one-hot-encoded using the OneHotEncoder function from the Scikit-Learn preprocessing package.

## Neural Network Prediction Model Optimization and Model Evaluation.

For the prediction stage (see file 'neuralnetwork_prediction.ipynb'), I employed a three hidden layer neural network model using the TensorFlow Keras library package. As there are so many hyperparameters that can influcence the model prediction results, including the number of layers and number of nodes at each layer. For this reason I made the decision to limit the model to three hidden layers with an arbitrarly selected finite set of possible nodes of 64, 128, 192, and 256. With these constrains I constructed an optimizer function that builds a neural network model for each permutation of the set of possible nodes, of which there are 4^3 = 64 possible permutations. The model contains an input layer of 81 nodes (equaling the number of features in the training dataset) and an output layer with a single node, as is the case with binary classification problems. For each hidden layer node I used the received linear unit (ReLU) activation function that returns the maximum between 0 and the input value. In other words the function returns only the positive part of the argument. For the output layer, I used the sigmoid function as the activation function. The sigmoid function (see below) returns a probability value between 0 and 1. Due to this, it is necessary to set an optimal probability threhold that can accurately separate the two classes. The standard threshold in a binary problem is 0.5, where values below 0.5 are the first class, and values above 0.5 are the opposite class. However, if there is class imbalance, as is the cases with this dataset, then the probability of one class over another can be skewed and a different threshold needs to be selected.     

Each model is evaluated based on the loss of the training sample and the cross-vlaidation sample; with training loss taking first priorty and cross-validation taking second. Based on the results, the optimal model per training loss and validation losss the one with hidden layers of (64, 64, 128) nodes, demonstraining a final training loss of 0.20 and a validation loss of 0.23. The optimizer function evaluates the models using a probability threshold of 0.5. Once I selected the optimal model above, I used the ROC Curve to identify the threshold where the geometric mean ratio between the true '0' rate and false '0' rate was greatest. Based on this measure, I determined the optimal threhold to be 0.516337. With this threshold, the results' metrics were accuracy of 0.92, precision of 0.95, test recall of 0.91 , and an f1-score of 0.93. 

To contextualize my results, I also separatley trained and tested a shallow single hidden layer model several times with different numbers of nodes, just to compare results against the three hidden layer model. Loss was consistently higher across all variations, as well as accuracy. For instance, with 100 nodes the final training loss was 0.318, the validation loss was 0.324. The resulting metrics were accuracy of 0.86, precision of 0.95, test recall of 0.91 , and an f1-score of 0.93. In a future experiment it might be interesting to try and optimize a single layer model on a finite randomized list of numbers and see if there is a model that returns better perfomance results than the three hidden layer model that I have trained.

## Considerations and Conculsions
Results in this project are heavily dependant on many different hyperparameters. The model I optimized in this project was done so using certain fixed or arbitrarly chosen hyperparameters, including number of layers, possible number of neurons, the learning rate (set to 0.001 for the experiment), as well as the number of epochs the algorithm executes. In future research it may be prudent to do further optimization of these hyperparameters, as they all may have a substantial effect on the training and performance results. However, due to the great computing and runtime requirements for such a task I did not include these tasks in this project.   
