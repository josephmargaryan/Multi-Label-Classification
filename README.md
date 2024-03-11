# Multi-Label-Classification
Instantiating a multi-perceptron neural network which works well for a tabular dataset and using it as a base model for a voting classifier

A Multi Label Classification framework on a tabular dataset. This is a proposed framework for utilizing the power of neural networks for tabular machine-learning problems. 
The Nural network uses skip connections, as introduced in the Res-net. This helped me with the vanishing gradient problem. Since every row could correspond to multiple target classes, we treat each target class as a binary classification problem. 
I used the area under the receiving operating curve to evaluate the model.
I instantiate a Catboost, LGBM, and eXGBM. I then performed thorough hyperparameter tuning through the Optuna library. I then use these models' predictions as parameters for my neural network.
