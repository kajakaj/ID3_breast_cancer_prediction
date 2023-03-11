# Predicting breast cancer using written from scratch ID3 algorithm
I divided the data into three sets - training, validation, and testing in the ratio of 6:2:2. I created a tree using the training set, and I tried to find the optimal depth using the validation set. Finally, I checked the quality of prediction on the testing set. The best result was approx. 76% for a depth of 2. 
 
Below is a plot showing the relationship between the depth of the tree and the accuracy of the obtained result on the validation set.  

![](https://github.com/kajakaj/ID3_breast_cancer_prediction/blob/main/resources/plot_id3.png?raw=true)
