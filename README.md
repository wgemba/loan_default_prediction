# Business Loan Credit Worthiness Prediction using an Optimized Neural Network Model

# Abstract

# Data Feature Engineering
## About the Data
The raw data used for this project is from the U.S. Small Business Administration (SBA) and was downloaded from Kaggle.com. The dataset include 27 features and 899,164 rows. The table below summarizes the features contents:

![image](https://github.com/wgemba/loan_default_prediction/assets/134420287/fe388d2e-0330-4293-9a83-6ac8b915fd54)

## Feature Engineering

The data set required significant preprocessing and feature engineering. Following several layers of preprocessing and engineering, including: transforming categorical features into numeric one, encoding large-set categorical features, removing class imbalance, removing null values, etc., the finalized dataset contains 81 features and 189,019 rows (see feature descriptions below).

![image](https://github.com/wgemba/loan_default_prediction/assets/134420287/04d8dba7-b2ce-45b9-a4fc-62e86212f367)

The full code for the feature processing can be seen in the file 'feature_engineering_and_visualization.ipynb'.

### Specific Packages Used
Certain features required the use of special packages and functions to convert from categorical to numeric datatype. Due to the large set of unique values for the feature 'Zip', rather than encoding the attribute I used the package 'pgeocode'. The package 'pgeocode' is a Python library that queries geospatial data (e.g., GPS coordinates, region name, municipality name) from the GeoNames database, using a postal code as input. Using this function, I created two new attributes 'latitude' and 'longitude' inplace of the 'Zip' feature. 

The 'State' attribute and and the 'Industry' attribute (which was created based on a dictionary remapping of the NAICS attributed, using the official NAICS code dictionary) were one-hot-encoded using the OneHotEncoder function from the Scikit-Learn preprocessing package.

## Neural Network Prediction Model Optimization and Model Evaluation.

For the prediction stage (see file 'neuralnetwork_prediction.ipynb'), I employed a three hidden layer neural network model using the TensorFlow Keras library package. As there are so many hyperparameters that can influcence the model prediction results, including the number of layers and number of nodes at each layer. For this reason I made the decision to limit the model to three hidden layers with an arbitrarly selected finite set of possible nodes of 64, 128, 192, and 256. With these constrains I constructed an optimizer function that builds a neural network model for each permutation of the set of possible nodes, of which there are 4^3 = 64 possible permutations. The model contains an input layer of 81 nodes (equaling the number of features in the training dataset) and an output layer with a single node, as is the case with binary classification problems. For each hidden layer node I used the rectived linear unit (ReLu) activation function that returns the maximum between 0 and the input value. In other words the function returns only the positive part of the argument. For the output layer, I used the sigmoid function as the activation function. The sigmoid function (see below) returns a probability value between 0 and 1. Due to this, it is necessary to set an optimal probability threhold that can accurately separate the two classes. The standard threshold in a binary problem is 0.5, where values below 0.5 are the first class, and values above 0.5 are the opposite class. However, if there is class imbalance, as is the cases with this dataset, then the probability of one class over another can be skewed and a different threshold needs to be selected.     

Each model is evaluated based on the metrics of accuracy, precision, recall, and f1-score; with accuracy taking first priority. Based on the evaluation metrics results, the optimal model per accuracy score was the one with hidden layers of (64, 64, 128) nodes, demonstration accuracy of 80.24%, precition of 85.01%, test recall of 0.824 , and an f1-score of 0.837. The optimizer function evaluates the models using a probability threshold of 0.5. Once I selected the optimal model above, I tried to see if I could improe accuracy by optimizing the the threshold. To do this, I used the ROC Curve, identifying the threshold where the geometric mean ratio between the true '0' rate and false '0' rate was greatest. Based on this measure, I determined the optimal threhold to be 0.483769, which is very close to the original 0.5. Rerunning the model evaluation using this threhold, I was able to improve accuracy slightly from 80.24% to 81.19%. Additionally, test precision improved from 85.01% to 87.92%, f1-score from 0.837 to 0.840, meanwhile recall decreased from 0.824 to 0.805.

For additional context, I separatley trained and tested a shallow single hidden layer model several times with different numbers of nodes, just to compare results against the three hidden layer model. Accuracy was consistently lower than the three hidden layer model, oscillating between 60% and 70% depending on the number of nodes in the model. In a future experiment it might be interesting to try and optimize a single layer model on a finite randomized list of numbers and see if there is a model that returns better perfomance results than the three hidden layer model that I have trained.

## Considerations and Conculsions
Results in this project are heavily dependant on many different hyperparameters. The model I optimized in this project was done so using certain fixed or arbitrarly chosen hyperparameters, including number of layers, possible number of neurons, the learning rate (set to 0.001 for the experiment), as well as the number of epochs the algorithm executes. In future research it may be prudent to do further optimization of these hyperparameters, as they all may have a substantial effect on the training and performance results. However, due to the great computing and runtime requirements for such a task I did not include these tasks in this project.   
