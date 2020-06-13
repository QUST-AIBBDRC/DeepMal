##DeepMal

DeepMal:Accurate prediction of protein malonylation sites by deep neural networks

##Pipeline



###DeepMal uses the following dependencies:
* MATLAB2014a
* python 3.6 
* numpy
* scipy
* scikit-learn (Deep learning library)
* keras(Machine learning library)


###Guiding principles:

**The data contains training dataset and testing dataset.
   Training dataset includes ecoli_train,H_train and mus_train
   Testing dataset includes ecoli_test,H_test and mus_test

**Feature extraction:
   EAAC.py is the implementation of enhanced amino acid composition.
   EGAAC.py is the implementation of enhanced grouped amino acid composition.
   KNN.py is the implementation of K nearest neighbors.
   DDE.py is the implementation of dipeptide deviation from expected mean.
   BLOSUM62.py is the implementation of BLOSUM62 matrix.
   

** Classifier:
   DL.py is the implementation of DL.
   DL_1.py is the implementation of DL_1.
   DNN.py is the implementation of Deep neural network.
   GRU.py is the implementation of Recurrent neural network.
   XGBoost_classifier.py is the implementation of XGBoost.
   SVM_classifier.py is the implementation of SVM.


