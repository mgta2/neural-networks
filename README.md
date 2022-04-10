# Introduction to Neural Networks Mini-Project

### Project Aims

1. Understand the fundamentals of Neural Networks.

2. Write code to implement Neural Network functionality.

3. Test the model.

## Contents

neuralnetwork.py    -    Contains the NeuralNetwork class.

theory.md    -    A brief discussion of the theory and important decisions taken.

digit_classification.ipynb    -    Testing neural networks on digit classification.

student_performance.ipynb    -    Testing neural networks on predicting student academic performance.

## Results

The project was a success. The NeuralNetwork class gave huge functionality in only 60 lines of Python code.

The final digit classification model had an accuracy of 96% on both testing and training data, which was great. The neural network did less well when predicting student academic performance; the model had an accuracy of 90% on train data and 70% on test data. I think this is for two reasons:

1 - Lack of data (only 649 total datapoints): it seems neural networks need large amounts of data to really flourish, as there are often many weights that need to be tuned.

2 - Lack of meaningful information: the dataset contains self-reported questionnaires from the students along with basic facts like how many times the student has been absent. It seems unlikely that this is enough information to fully capture what causes academic success (for instance, there is no data measuring how a student performs under pressure).

## Next Steps

The aims of this project were not to build the best model for digit classification or student performance.
This work could be taken further by carefully tuning the network's shape and learning rate parameter to try and maximise accuracy.

The sklearn package models are designed so that one can enter all the train/query data at once, not line by line as in my NeuralNetwork class here. This would be a convenient improvement to make in the future.

## Acknowledgements

- This project was inspired by Educative's brilliant "Make Your Own Neural Network in Python" course.

- The .csv formatted MNIST dataset was downloaded from https://pjreddie.com/projects/mnist-in-csv/

- The student-por.csv dataset comes from https://archive.ics.uci.edu/ml/datasets/student+performance