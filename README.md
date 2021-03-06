# Targeting Agent

An MLP used to predict whether a point is on target or not. Part of the University of Hull Neural, Emergent and Agent Technologies module.

## Network Architecture:
* Number of Hidden Layers: 2 
* Activation function:
  * Hidden Layers: Sigmoid
  * Output Layer: Sigmoid
* Number of Nodes: 18 
* Learning Rate: 0.01 
* Momentum Term: 0.01 
* Optimisation Algorithm: Stochastic Gradient Descent 
* Loss Function: Binary Cross-entropy

Plot of the given data:  
![Plot of data](https://github.com/DWhettam/Targeting-Agent/blob/master/Target.png)

Over 1000 epochs, this network is classifying with ~98% accuracy

### Requirements
* [Keras](https://github.com/keras-team/keras) with [TensorFlow](https://www.tensorflow.org) backend
* [Matplotlib](https://github.com/matplotlib/matplotlib)
* [SciKit-Learn](https://github.com/scikit-learn/scikit-learn)
* [Numpy](https://github.com/numpy/numpy)
* [Pandas](https://github.com/pandas-dev/pandas)
