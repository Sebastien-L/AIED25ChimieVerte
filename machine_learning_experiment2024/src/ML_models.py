"""
AI fairness analysis code
@author XXX
@email XXX
@date 01/02/2024
@licence BSD-3-Clause 
"""

import pandas as pd
import numpy as np

import os
import sys
import warnings
from copy import deepcopy

from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GroupShuffleSplit
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE 

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Attention, Input, LayerNormalization, Reshape, AdditiveAttention, Layer, Lambda, Flatten, Concatenate, LeakyReLU, Multiply, Softmax
from tensorflow.keras.optimizers import Adam
#from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.metrics import F1Score
from keras_tuner.tuners import SklearnTuner
from keras_tuner.oracles import BayesianOptimizationOracle
import keras_tuner
from keras_tuner import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters

tf.compat.v1.enable_eager_execution()
# adding src to the system path to find scripts in there
sys.path.insert(0, "./modules")
from threshold_optimizer import ThresholdOptimizer

DEFAULT_SEED = 42

############################################################################################################################################################
#PREPROCESSING CLASSES 
############################################################################################################################################################
class DataFrameMinMaxScaler(BaseEstimator, TransformerMixin):
	""" Custom transformer for scicit-learn, adapts MinMaxScaler() for panda dataframe"""
	def __init__(self): # no *args or ** kargs
		 self.scaler = MinMaxScaler()
		 self.n_features_in_ = 0
			
	def fit(self, X, y = None):
		return self
			
	def transform(self, X, y = None):
		try:
			modified_X = self.scaler.fit_transform(X)
			modified_X= pd.DataFrame(modified_X)
			self.n_features_in_ = len(modified_X.columns)
			return modified_X
		except:
			self.n_features_in_ = len(X)
			return X


class DeleteCorrelatedFeatures(BaseEstimator, TransformerMixin):
	""" Custom transformer for scicit-learn, meant to delete highly correlated features in a dataframe"""
	default_threshold = 0.8
	
	def __init__(self, threshold = default_threshold): # no *args or ** kargs
		self.threshold = threshold
		self.to_drop = []
		self.to_keep = []
		
	def fit(self, X, y = None):
		if isinstance(X, pd.DataFrame):
			corr_matrix = X.corr().abs()
			upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
			self.to_drop = [column for column in upper.columns if any(upper[column] >= self.threshold)]
		else:
			corr_matrix = np.absolute(np.corrcoef(X, rowvar=False))
			upper = corr_matrix*np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
			self.to_drop = [column for column in range(upper.shape[1]) if any(upper[:,column] >= self.threshold)]
		return self
	
	def transform(self, X, y = None):
		try:
			if isinstance(X, pd.DataFrame):
				X_reduced = X.drop(columns = self.to_drop)
				self.to_keep = X_reduced.columns
			else:
				X_reduced = np.delete(X, self.to_drop, axis=1)
			return X_reduced
		except:
			return X
		

class PCAFeaturesSelection(BaseEstimator, TransformerMixin):
	""" Custom transformer for scicit-learn, meant to reduce data dimentionality via PCA components"""
	default_threshold_variance = 0.95
	default_svd_solver = 'full'
	
	def __init__(self, threshold_variance = default_threshold_variance, svd_solver = default_svd_solver): # no *args or ** kargs
		self.threshold_variance = threshold_variance
		self.svd_solver = svd_solver
		self.loadings = []
		self.nb_components = 0
		
	def fit(self, X, y = None):
		self.PCA = PCA(n_components=self.threshold_variance, svd_solver=self.svd_solver)
		self.pca_components = self.PCA.fit_transform(X)
		
		self.nb_components = self.PCA.n_components_
		self.loadings = pd.DataFrame(self.PCA.components_, columns=X.columns, index=[f'PC{i+1}' for i in range(self.nb_components)])
		return self
	
	def transform(self, X, y = None):
		X_transformed = self.PCA.transform(X)
		pca_columns = [f'PC{i+1}' for i in range(self.nb_components)]
		pca_df = pd.DataFrame(X_transformed, columns=pca_columns)
		return pca_df


class DataFrameSelectKBest(BaseEstimator, TransformerMixin):
	"""Custom transformer that performs SelectKBest on a pandas DataFrame and returns a DataFrame with the selected features."""
	default_score_func = f_classif
	default_k = 10
	
	def __init__(self, score_func=default_score_func, k=default_k):
		self.score_func = score_func   # Scoring function used by SelectKBest (default: f_classif).
		self.k = k # Number of top features to select
		self.selector = None  # Fitted SelectKBest object

	def fit(self, X, y=None):
		if self.k > X.shape[1]: # if not enough features
			self.k = X.shape[1]
		self.selector = SelectKBest(score_func=self.score_func, k=self.k)
		self.selector.fit(X, y)
		return self

	def transform(self, X):
		# Get selected feature indices and names
		mask = self.selector.get_support()
		selected_features = X.columns[mask]

		return X[selected_features]
		
		
class DeleteIrrelevantFeatures(BaseEstimator, TransformerMixin):
	""" Custom transformer for scicit-learn, to delete irrelevant features (all values are the same, no variance)"""
	def __init__(self, threshold_variance = 1): # no *args or ** kargs
		 self.threshold_variance = threshold_variance
			
	def fit(self, X, y = None):
		self.unused_features = []
		for key in X.columns:
			nb_values = X[key].value_counts()
			if len(nb_values) == self.threshold_variance: # If all the values are the same
				self.unused_features.append(key)
		return self
		
	def transform(self, X, y = None):
		# print("Number of deleted features",len(self.unused_features))
		modified_X = X.drop(columns=self.unused_features)	
		return modified_X

		
		
# Custom loss combining binary crossentropy and sparse attention loss
def sparse_attention_loss(y_true, y_pred, attention_weights, sparsity_weight=1e-4):
	base_loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)
	attention_reg_loss = tf.reduce_sum(tf.square(attention_weights)) * sparsity_weight
	return base_loss + attention_reg_loss


class DotProductAttention(Layer):
	def __init__(self, **kwargs):
		super(DotProductAttention, self).__init__(**kwargs)
	
	def build(self, input_shape):
		super(DotProductAttention, self).build(input_shape)

	def call(self, inputs):
		# Directly compute attention on 2D input (batch_size, features)
		query, value = inputs
		# Compute dot-product attention scores
		attention_scores = tf.matmul(query, value, transpose_b=True)
		# Apply softmax to get attention weights
		self.attention_weights = tf.nn.softmax(attention_scores, axis=-1)
		# Compute the context (weighted sum of values)
		context = tf.matmul(self.attention_weights, value)
		return context, self.attention_weights
		
        
############################################################################################################################################################
#ML MODEL SUPER CLASS 
############################################################################################################################################################

class ML_Model:
	"""
	Genetric superclass for ML classifiers in scitcit-learn
	"""
	def __init__(self, pipeline, hyperparameter_grid, correl_threshold=0.8, grid_search=False, nb_inner_folds=5, labels = [0, 1], verbose=False, seed=DEFAULT_SEED):
		"""
		Init 3 attributes: a sklearn pipeline (self.pipeline), a sklearn GridSearchCV process (self.grid), and a grid for the hyper-parameter tuning in sklearn format (self.hyperparameter_grid)
		"""
		self.pipeline = pipeline
		self.hyperparameter_grid = hyperparameter_grid
		self.correl_threshold = correl_threshold
		self.grid_search = grid_search
		self.labels_values = labels
		self.nb_inner_folds = nb_inner_folds
		self.seed = seed
		self.verbose = verbose
		self.init_grid_search()
		
		
	def build_pipeline(self, classifier, delete_correl = True, feature_selection = None, correl_threshold = 0.8, PCA_variance = 0.95, selectK = 10):
		"""
		Build and return a sklearn pipeline to train a classifier. The default pipeline includes discarding features with no variance (DeleteIrrelevantFeatures), rescaling the features (DataFrameMinMaxScaler), and the model provided in the "classifier" parameter.
		Optional parts of the pipeline:
			- Delete highly correlated features (DeleteCorrelatedFeatures) if delete_correl is True. correl_threshold is forwarded as the argument to DeleteCorrelatedFeatures
			- PCA feature selection (PCAFeaturesSelection) if feature_selection=="PCA". PCA_variance is the variance to be explained.
			- K best feature selection based on statistical test (DataFrameSelectKBest) if feature_selection=="Kbest". selectK is the target number of feature.
		"""
		pipe = [("feature_deleter", DeleteIrrelevantFeatures()), ("data_scaler", DataFrameMinMaxScaler())]
		if delete_correl:
			pipe.append( ("features_correlated", DeleteCorrelatedFeatures(correl_threshold)) )
			
		if feature_selection == "PCA":
			pipe.append( ("features_selection", PCAFeaturesSelection(threshold_variance=PCA_variance)) )
		
		if feature_selection == "Kbest":
			pipe.append( ("features_selection", DataFrameSelectKBest(k=selectK)) )
		
		pipe.append(classifier)
		
		return Pipeline(pipe)
			
		
	def init_grid_search(self, scoring ='f1_weighted'):
		"""
		Initialize the GridSearchCV object. see GridSearchCV() documentation in sklearn for the scoring.
		"""
		stratified_inner_cross_val = StratifiedKFold(n_splits = self.nb_inner_folds, shuffle = True, random_state = self.seed)
		self.grid = GridSearchCV( 
			estimator = self.pipeline,
			param_grid = self.hyperparameter_grid,
			cv = stratified_inner_cross_val,
			verbose = 1,
			scoring ='f1_weighted',
			n_jobs=-1)
			
	def fit_predict(self, train_set_features, train_set_labels, test_set_features):
		"""
		Fit the classifier on the train set and predict on the test set. Return three values: the predictions on both the train and test set, and the probabilities outputed by the classifiers on the test set.
		"""
		with warnings.catch_warnings(record=True) as w:
			self.pipeline.fit(train_set_features, train_set_labels)
			predictions_train = self.pipeline.predict(train_set_features)
			predictions_test = self.pipeline.predict(test_set_features)
			proba_train = self.pipeline.predict_proba(train_set_features)
			proba_test = self.pipeline.predict_proba(test_set_features)
			
			if w is not None and len(w) > 0:
				print(print(w[-1].category))
				if self.verbose:
					print(w[-1].message)
					
		return predictions_train, predictions_test, proba_train, proba_test
					
	def grid_fit_predict(self, train_set_features, train_set_labels, test_set_features):
		"""
		Fit the classifier on the train set using the grid hyperparameter tuning method, and predict on the test set. Return three values: the predictions on both the train and test set, and the probabilities outputed by the classifiers on the test set.
		"""
		with warnings.catch_warnings(record=True) as w:
			self.grid.fit(train_set_features, y = train_set_labels)
			grid_predictions_train = self.grid.predict(train_set_features)
			grid_predictions_test = self.grid.predict(test_set_features)
			grid_proba_train = grid_predictions_train
			grid_proba_test = grid_predictions_test
			

			grid_proba_train = self.grid.predict_proba(train_set_features)
			grid_proba_test = self.grid.predict_proba(test_set_features)
			
			if w is not None and len(w) > 0:
				print(print(w[-1].category))
				if self.verbose:
					print(w[-1].message)

		return grid_predictions_train, grid_predictions_test, grid_proba_train, grid_proba_test
		
	def classification_threshold_optimizer(self, pred_proba_train, labels_train, pred_proba_test):
		"""
		Optimize the classification threshold using the threshold_optimizer package on the train set, and return the threshold along with the new predictions on the test set.
		"""
		# init optimization
		thresh_opt = ThresholdOptimizer(y_score = pred_proba_train, y_true = labels_train)

		# optimize for f1 score
		thresh_opt.optimize_metrics(metrics=['f1'], verbose=False)
		f1_threshold = thresh_opt.optimized_metrics['f1_score']['best_threshold']
		
		# use best accuracy threshold for test set to convert probabilities to classes
		return np.where(pred_proba_test[:,1] > f1_threshold, self.labels_values[1], self.labels_values[0])


class KerasWrapper(BaseEstimator, ClassifierMixin):
	def __init__(self, keras_model):
		self.keras_model = keras_model

	def fit(self, X, y):  
		"""Dummy fit method for compatibility; model is pre-trained."""
		return self  

	def predict(self, X):
		"""Use Keras model to predict class labels (hard voting)."""
		return np.argmax(self.keras_model.predict(X), axis=1)  

	def predict_proba(self, X):
		"""Use Keras model to predict probabilities (for soft voting)."""
		return self.keras_model.predict(X)


############################################################################################################################################################
#ML MODELS SUBCLASSES
############################################################################################################################################################

# Subclasses for specific baselines and classifiers
class StratifiedBaselineClassifier(ML_Model):
	def __init__(self, **kwargs):
		#Dummy baseline from sklearn
		dummy_random_clf_pipe = Pipeline([
			("classifier", DummyClassifier(strategy="stratified"))
		])
		super().__init__(dummy_random_clf_pipe, None, **kwargs)

class MajorityClassBaselineClassifier(ML_Model):
	def __init__(self, **kwargs):
		#Dummy baseline from sklearn
		dummy_random_clf_pipe = Pipeline([
			("classifier", DummyClassifier(strategy="most_frequent"))
		])
		super().__init__(dummy_random_clf_pipe, None, **kwargs)

class NaiveBayesClassifier(ML_Model):
	def __init__(self, correl_threshold=0.8, Kbest_features = 10, feature_selection = None, grid_search=False, **kwargs):
		# Multinomial Naive Bayes classifier
		model = ("classifier", GaussianNB())
		naive_bayes_clf_pipe = self.build_pipeline(model, feature_selection = feature_selection, correl_threshold = correl_threshold, selectK = Kbest_features)
		
		param_grid_nb = [{
				'classifier__var_smoothing': [1e-9] #default
			}]
		if grid_search:
			param_grid_nb = [{
				# Default value in sklearn: 1.0
				'classifier__var_smoothing': [1e-7, 1e-8, 1e-9, 1e-10, 1e-11] #large grid
				#'naive_bayes_clf__alpha': [10.0, 1.0, 0.1] #small grid
			}]
			
		super().__init__(naive_bayes_clf_pipe, param_grid_nb, correl_threshold, grid_search, **kwargs)

class KNeirestNeighboorsClassifier(ML_Model):
	def __init__(self, correl_threshold=0.8, Kbest_features = 10, feature_selection = None, grid_search=False, **kwargs):
		# Multinomial Naive Bayes classifier
		model = ("classifier", KNeighborsClassifier())
		knn_clf_pipe = self.build_pipeline(model, feature_selection = feature_selection, correl_threshold = correl_threshold, selectK = Kbest_features)
		
		param_grid_nb = [{
				'classifier__n_neighbors': [5] #default
			}]
		if grid_search:
			param_grid_nb = [{
				# Default value in sklearn: 1.0
				'classifier__n_neighbors': [3, 4, 5, 6, 7, 8, 9, 10, 15, 20] #large grid
			}]
			
		super().__init__(knn_clf_pipe, param_grid_nb, correl_threshold, grid_search, **kwargs)
		
class LogisticClassifier(ML_Model):
	def __init__(self, correl_threshold=0.8, Kbest_features = 10, feature_selection = None, grid_search=False, **kwargs):
		# Logistic regression
		model = ("classifier", LogisticRegression(max_iter=250, random_state=kwargs.get("seed", DEFAULT_SEED)))
		classifier_pipe = self.build_pipeline(model, feature_selection = feature_selection, correl_threshold = correl_threshold, selectK = Kbest_features)
		
		param_grid_lr = [{
			'classifier__C': [1.0], #default
			'classifier__max_iter': [250] #to converge
		}]
		if grid_search:
			param_grid_lr = [{
				# Default value in sklearn: 1.0
				'classifier__C': [100.0, 50.0, 25.0, 10.0, 5.0, 1.0, 0.5, 0.1], #large grid
				#'classifier__C': [10.0, 1.0, 0.1], #small grid
				'classifier__max_iter': [250] #to converge
			}]
		
		super().__init__(classifier_pipe, param_grid_lr, correl_threshold, grid_search, **kwargs)
		
class SVMClassifier(ML_Model):
	def __init__(self, correl_threshold=0.8, Kbest_features = 10, feature_selection = None, grid_search=False, **kwargs):	
		# SVM
		model = ("classifier", SVC(probability=True))
		svm_clf_pipe = self.build_pipeline(model, feature_selection = feature_selection, correl_threshold = correl_threshold, selectK = Kbest_features)
		
		param_grid_svm = [{
			'classifier__C': [1.0], #default
			'classifier__gamma': [1.0], #default
			'classifier__kernel': ['rbf'] #default
		}]
		
		if grid_search:
			param_grid_svm = [{
				# Default value in sklearn: 1.0
				'classifier__C': [100.0, 50.0, 25.0, 10.0, 5.0, 1.0, 0.5, 0.1], #large grid
				#'classifier__C': [10.0, 1.0, 0.1], #small grid
				# Default value in sklearn: 1.0
				'classifier__gamma': [10.0, 1.0, 0.5, 0.1, 0.01, 0.001, 0.0001], #large grid
				#'classifier__gamma': [10.0, 1.0, 0.1], #small grid
				# Default value in sklearn: rbf
				'classifier__kernel': ['rbf', 'poly', 'sigmoid'] #large+small grid
			}]

		super().__init__(svm_clf_pipe, param_grid_svm, correl_threshold, grid_search, **kwargs)



class RandomForestEnsembleClassifier(ML_Model):
	def __init__(self, correl_threshold=0.8, Kbest_features = 10, feature_selection = None, grid_search=False, **kwargs):   
		#Random Forests
		model = ("classifier", RandomForestClassifier(random_state = kwargs.get("seed", DEFAULT_SEED)))
		classifier_pipe = self.build_pipeline(model, feature_selection = feature_selection, correl_threshold = correl_threshold, selectK = Kbest_features)
		
		param_grid_rf = [{
			"classifier__n_estimators": [100], #default
			"classifier__max_depth" : [None],	#default
		}]
		if grid_search:
			param_grid_rf = [{
				# Default value in sklearn: 100
				"classifier__n_estimators": [20, 50, 100, 200, 300, 500], #large grid
				#"classifier__n_estimators": [50, 100, 200], #small grid
				# Default value in sklearn: None
				"classifier__max_depth" : [4,6,8,10,12,14,16,None],	#large grid
				#"classifier__max_depth" : [6,12,None],	#small grid
			}]
		
		super().__init__(classifier_pipe, param_grid_rf, correl_threshold, grid_search, **kwargs)


class StochasticGradientBoostingAlgorithmClassifier(ML_Model):
	def __init__(self, correl_threshold=0.8, Kbest_features = 10, feature_selection = None, grid_search=False, **kwargs):   
		#Random Forests
		model = ("classifier", GradientBoostingClassifier(n_estimators = 250, random_state = kwargs.get("seed", DEFAULT_SEED)))
		classifier_pipe = self.build_pipeline(model, feature_selection = feature_selection, correl_threshold = correl_threshold, selectK = Kbest_features)
		
		param_grid_sgb = [{
			"classifier__learning_rate": [0.1], #default
			"classifier__subsample" : [1.0],	#default
			"classifier__max_depth" : [3],	#default
			"classifier__min_samples_split" : [2],	#default
		}]
		if grid_search:
			param_grid_rf = [{
				"classifier__learning_rate": [10.0, 5.0, 1.0, 0.1, 0.01, 0.001, 0.0], #large grid
				"classifier__subsample" : [1.0, 0.9, 0.8, 0.7, 0.6, 0.5],	#large grid
				"classifier__max_depth" : [2, 3, 4, 5, 8, 10, 12, 14, 16, None],	#large grid
				"classifier__min_samples_split" : [2, 4, 6, 8, 10],	#large grid
			}]
		
		super().__init__(classifier_pipe, param_grid_sgb, correl_threshold, grid_search, **kwargs)


class MultiLayerPerceptronClassifier(ML_Model):
	def __init__(self, correl_threshold=0.8, Kbest_features = 10, feature_selection = None, grid_search=False, **kwargs): 
		#Multilayer Perceptron
		model = ("classifier", MLPClassifier(max_iter=100, random_state = kwargs.get("seed", DEFAULT_SEED)))
		classifier_pipe = self.build_pipeline(model, feature_selection = feature_selection, correl_threshold = correl_threshold, selectK = Kbest_features)
	
		param_grid_mlp = [{
			"classifier__hidden_layer_sizes": [(30,)], #default/2 small dataset
			"classifier__activation" : ['logistic'],	#logistic sigmoid function
			"classifier__solver" : ['lbfgs'],	#for small dataset
			"classifier__alpha" : [0.0001],	#default
			"classifier__learning_rate" : ['constant']	#default
		}]
		if grid_search:
			param_grid_mlp = [{
				"classifier__hidden_layer_sizes": [(10,), (20,), (30,), (50,), (30,20,), (50, 30,), (20, 10,)], #large grid
				"classifier__activation" : ['logistic', 'relu'],	#large grid
				"classifier__solver" : ['lbfgs', 'adam'],	#large grid
				"classifier__alpha" : [0.5, 0.1, 0.05, 0.01, 0.001, 0.0001, 0.0],	#large grid
				"classifier__learning_rate" : ['constant'],	#large grid
				"classifier__max_iter" : [30, 50, 100, 250, 500],	#large grid
			}]
		
		super().__init__(classifier_pipe, param_grid_mlp, correl_threshold, grid_search, **kwargs)
		
		
class SoftVotingClassifier(ML_Model, BaseEstimator, ClassifierMixin):
	def __init__(self, correl_threshold=0.8, Kbest_features = 10, feature_selection = None, grid_search=False, **kwargs):   
		#VotingClassifier
		self.feature_selection = feature_selection
		self.correl_threshold = correl_threshold
		self.Kbest_features = Kbest_features
		self.grid_search = grid_search
		self.estimators = []
		self.final_estimator=final_estimator=self	
		model = ("classifier", self)
		self.pipeline = self.build_pipeline(model, feature_selection = self.feature_selection, correl_threshold = self.correl_threshold, selectK = self.Kbest_features)
		super().__init__(self.pipeline, [{}], correl_threshold, [{}], **kwargs) 

	def fit(self, X, y):
		self.pipeline[:-1].fit(X, y)
		return self
		
	def predict(self, X):
		X_test_preprocessed = self.pipeline[:-1].transform(X)
		self.input_dim = X_test_preprocessed.shape[1]
			
		probas = []
		for name, model in self.estimators:
			if "FF" in name:
				model_preds = model.predict(X_test_preprocessed)
				probas.append(np.column_stack((1 - model_preds, model_preds)))
			else:
				probas.append(model.predict_proba(X_test_preprocessed))
		#print(probas)
		avg_proba = np.average(np.array(probas), axis=0)
		return np.argmax(avg_proba, axis=1)

	def predict_proba(self, X):
		"""Returns averaged probabilities for soft voting."""
		X_test_preprocessed = self.pipeline[:-1].transform(X)
		probas = []
		for name, model in self.estimators:
			if "FF" in name:
				model_preds = model.predict(X_test_preprocessed)
				probas.append(np.column_stack((1 - model_preds, model_preds)))
			else:
				probas.append(model.predict_proba(X_test_preprocessed))
		return np.average(probas, axis=0)
		
	def fit_predict(self, train_set_features, train_set_labels, test_set_features):
		"""
		Fit the classifier on the train set and predict on the test set. Return three values: the predictions on both the train and test set, and the probabilities outputed by the classifiers on the test set.
		"""
		with warnings.catch_warnings(record=True) as w:
			self.fit(train_set_features, train_set_labels)
			predictions_train = self.predict(train_set_features)
			predictions_test = self.predict(test_set_features)
			proba_train = self.predict_proba(train_set_features)
			proba_test = self.predict_proba(test_set_features)
			
			if w is not None and len(w) > 0:
				print(print(w[-1].category))
				if self.verbose:
					print(w[-1].message)
					
		return predictions_train, predictions_test, proba_train, proba_test


class CustomStackingClassifier(ML_Model, BaseEstimator, ClassifierMixin):
	def __init__(self, final_estimator, correl_threshold=0.8, Kbest_features = 10, feature_selection = None, grid_search=False, **kwargs):   
		#CustomStackingClassifier
		self.feature_selection = feature_selection
		self.correl_threshold = correl_threshold
		self.Kbest_features = Kbest_features
		self.grid_search = grid_search
		self.estimators = []	
		self.final_estimator=final_estimator
		model = ("classifier", self.final_estimator)
		self.pipeline = self.build_pipeline(model, feature_selection = self.feature_selection, correl_threshold = self.correl_threshold, selectK = self.Kbest_features)
		super().__init__(self.pipeline, [{}], correl_threshold, [{}], **kwargs) 

	def fit(self, X, y):
		self.pipeline[:-1].fit(X, y)
		X_train_preprocessed = self.pipeline[:-1].transform(X)
			
		base_probas = []
		for name, model in self.estimators:
			if "FF" in name:
				model_preds = model.predict(X_train_preprocessed)
				base_probas.append(np.column_stack((1 - model_preds, model_preds)))
			else:
				base_probas.append(model.predict_proba(X_train_preprocessed))
		base_probas = np.column_stack(base_probas)
		self.final_estimator.fit(base_probas, y)
		return self
		
	def predict(self, X):
		X_test_preprocessed = self.pipeline[:-1].transform(X)
		self.input_dim = X_test_preprocessed.shape[1]
		
		base_probas = []
		for name, model in self.estimators:
			if "FF" in name:
				model_preds = model.predict(X_test_preprocessed)
				base_probas.append(np.column_stack((1 - model_preds, model_preds)))
			else:
				base_probas.append(model.predict_proba(X_test_preprocessed))
			
		base_probas = np.column_stack(base_probas)
		return self.final_estimator.predict(base_probas)

	def predict_proba(self, X):
		"""Returns averaged probabilities for soft voting."""
		X_test_preprocessed = self.pipeline[:-1].transform(X)
		self.input_dim = X_test_preprocessed.shape[1]
		
		base_probas = []
		for name, model in self.estimators:
			if "FF" in name:
				model_preds = model.predict(X_test_preprocessed)
				base_probas.append(np.column_stack((1 - model_preds, model_preds)))
			else:
				base_probas.append(model.predict_proba(X_test_preprocessed))
			
		base_probas = np.column_stack(base_probas)
		return self.final_estimator.predict_proba(base_probas)
		
	def fit_predict(self, train_set_features, train_set_labels, test_set_features):
		"""
		Fit the classifier on the train set and predict on the test set. Return three values: the predictions on both the train and test set, and the probabilities outputed by the classifiers on the test set.
		"""
		with warnings.catch_warnings(record=True) as w:
			self.fit(train_set_features, train_set_labels)
			predictions_train = self.predict(train_set_features)
			predictions_test = self.predict(test_set_features)
			proba_train = self.predict_proba(train_set_features)
			proba_test = self.predict_proba(test_set_features)
			
			if w is not None and len(w) > 0:
				print(print(w[-1].category))
				if self.verbose:
					print(w[-1].message)
					
		return predictions_train, predictions_test, proba_train, proba_test
	   

class FeedForwardClassifier(ML_Model):
	def __init__(self, nb_hidden_layers = 1, correl_threshold=0.8, Kbest_features = 10, feature_selection = None, grid_search=False, **kwargs): 
		#Feed-forward neural network
		param_grid_ffnn = [{
			'classifier__learning_rate': [0.001],
			'classifier__dropout_rate': [0.2],
			'classifier__hidden_units': [16],
			'classifier__batch_size': [32],
			'classifier__epochs': [25]
		}]
		
		if grid_search:
			param_grid_ffnn = [{
				'classifier__learning_rate': [0.001, 0.01],
				'classifier__dropout_rate': [0.0, 0.25, 0.5],
				'classifier__hidden_units': [16, 32, 64, 128],
				'classifier__batch_size': [16, 32],
				'classifier__epochs': [25, 50]
			}]
		
		self.nb_hidden_layers = nb_hidden_layers
		self.Kbest_features = Kbest_features
		self.feature_selection = feature_selection
		self.input_dim = 0
		
		super().__init__(None, param_grid_ffnn, correl_threshold, grid_search, **kwargs)
		
		
	def init_grid_search(self, scoring ='f1_weighted'):
		"""
		Initialize the GridSearchCV object. see GridSearchCV() documentation in sklearn for the scoring.
		"""
		stratified_inner_cross_val = StratifiedKFold(n_splits = self.nb_inner_folds, shuffle = True, random_state = self.seed)
		"""self.grid = SklearnTuner(
			oracle=BayesianOptimizationOracle(objective=keras_tuner.Objective('val_f1_score', 'max'), max_trials=20, seed=self.seed),
			hypermodel= self.create_model,
			cv=stratified_inner_cross_val)"""
			
		self.grid = RandomSearch(self.create_model, #lambda hp: create_model_out(hp, input_dim=self.input_dim),#
				objective='val_f1_score',
				max_trials=20,
				executions_per_trial=3,
				directory='my_dir',
				project_name='trainFFNN'
				)
		
	def create_model(self, hp=None):
		#grid
		if self.grid_search and hp:
			hidden_units = hp.Choice('hidden_units', [16, 32, 64, 128])
			dropout_rate = hp.Float('dropout_rate', 0.0, 0.5, step=0.25)
			learning_rate = hp.Choice('learning_rate', [1e-2, 1e-3])
		else:
			hidden_units = 16
			dropout_rate = 0.2
			learning_rate = 0.001
		
		# Define input
		inputs = Input(shape=(self.input_dim,))
	
		x = inputs
		for _ in range(self.nb_hidden_layers):
			x = Dense(hidden_units)(x)
			x = LeakyReLU()(x)
			x = Dropout(dropout_rate)(x)

		# Output layer
		outputs = Dense(1, activation='sigmoid', name='output')(x)
		
		# Define and compile the model with custom loss function
		def custom_loss(y_true, y_pred):
			return sparse_attention_loss(y_true, y_pred[0], y_pred[1])

		# Define and compile the model
		model = Model(inputs=inputs, outputs=outputs)
		model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=[F1Score(average='weighted')])
		return model
	
		
	def build_pipeline(self, X_train):
		verb = 0 if not self.verbose else 1
		model_keras = self.create_model()
		model = ("classifier", model_keras)
		self.pipeline = super().build_pipeline(model, feature_selection = self.feature_selection, correl_threshold = self.correl_threshold, selectK = self.Kbest_features)
		self.init_grid_search()
		
		
	def grid_fit_predict(self, train_set_features, train_set_labels, test_set_features):
		"""
		#Override super method to account for attention weights.
		#Fit the classifier on the train set using the grid hyperparameter tuning method, and predict on the test set. Return three values: the predictions on both the train and test set, and the probabilities outputed by the classifiers on the test set.
		"""
		
		def custom_fit(hp, model, X, y):
			batch_size = hp.Int('batch_size', 16, 32, step=16)
			epochs = hp.Int('epochs', 50, 100, step=50)
			return model.fit(X, y, batch_size=batch_size,epochs=epochs,verbose=0)
			
		with warnings.catch_warnings(record=True) as w:
			self.pipeline[:-1].fit(train_set_features, train_set_labels)
			X_train_preprocessed = self.pipeline[:-1].transform(train_set_features)
			X_test_preprocessed = self.pipeline[:-1].transform(test_set_features)
			self.input_dim = X_train_preprocessed.shape[1]
			self.init_grid_search()
			
			if self.grid_search:
				X_train, X_val, y_train, y_val = train_test_split(X_train_preprocessed, train_set_labels, test_size=0.2, random_state=self.seed)
				#self.grid.search(X_train_preprocessed, train_set_labels, custom_fit=custom_fit)
				#self.grid.search(X_train_preprocessed, train_set_labels, batch_size=hp.Choice('batch_size', values=[16, 32]), epochs=self.hyperparameter_grid[0]["classifier__epochs"][0])
				self.grid.search(X_train, y_train, validation_data=(X_val, y_val), batch_size=self.hyperparameter_grid[0]["classifier__batch_size"][0], epochs=self.hyperparameter_grid[0]["classifier__epochs"][0])
				self.best_params_ = self.grid.get_best_hyperparameters(1)[0]
				self.best_model = self.grid.hypermodel.build(self.best_params_)
				#print(vars(self.best_params_))
				self.best_model.fit(X_train_preprocessed, train_set_labels, epochs=self.hyperparameter_grid[0]["classifier__epochs"][0], batch_size=self.hyperparameter_grid[0]["classifier__batch_size"][0], verbose=0)  # Retrain on the entire dataset
			else:
				self.best_model = self.create_model()
				self.best_model.fit(X_train_preprocessed, train_set_labels, batch_size=self.hyperparameter_grid[0]["classifier__batch_size"][0], epochs=self.hyperparameter_grid[0]["classifier__epochs"][0])
				self.best_params_ = self.hyperparameter_grid[0]
		
			grid_predictions_train = self.best_model.predict(X_train_preprocessed)
			grid_predictions_test = self.best_model.predict(X_test_preprocessed)
			grid_proba_test = grid_predictions_test
			grid_proba_train = grid_predictions_train
			
			grid_predictions_train = (grid_predictions_train > 0.5).astype(int)
			grid_predictions_test = (grid_predictions_test > 0.5).astype(int)
			
			grid_predictions_train = np.array(grid_predictions_train).flatten() #flatten for compatibility with sklearn
			grid_predictions_test = np.array(grid_predictions_test).flatten() #flatten for compatibility with sklearn

			if w is not None and len(w) > 0:
				print(print(w[-1].category))
				if self.verbose:
					print(w[-1].message)

		return grid_predictions_train, grid_predictions_test, grid_proba_train, grid_proba_test

	
		
class FeedForwardSelfAttentionClassifier(ML_Model):
	def __init__(self, nb_hidden_layers = 1, correl_threshold=0.8, Kbest_features = 10, feature_selection = None, grid_search=False, **kwargs): 
		#Feed-forward neural network
		param_grid_ffnn = [{
			'classifier__learning_rate': [0.001],
			'classifier__dropout_rate': [0.2],
			'classifier__hidden_units': [16],
			'classifier__batch_size': [16],
			'classifier__epochs': [50]
		}]
		
		if grid_search:
			param_grid_ffnn = [{
				'classifier__learning_rate': [0.001, 0.01],
				'classifier__dropout_rate': [0.0, 0.25, 0.5],
				'classifier__hidden_units': [16, 32, 64, 128],
				'classifier__batch_size': [16, 32],
				'classifier__epochs': [50, 100]
			}]
		
		self.nb_hidden_layers = nb_hidden_layers
		self.Kbest_features = Kbest_features
		self.feature_selection = feature_selection
		self.input_dim = 0
		
		super().__init__(None, param_grid_ffnn, correl_threshold, grid_search, **kwargs)
		
		
	def init_grid_search(self, scoring ='f1_weighted'):
		"""
		Initialize the GridSearchCV object. see GridSearchCV() documentation in sklearn for the scoring.
		"""
		stratified_inner_cross_val = StratifiedKFold(n_splits = self.nb_inner_folds, shuffle = True, random_state = self.seed)
		"""self.grid = SklearnTuner(
			oracle=BayesianOptimizationOracle(objective=keras_tuner.Objective('val_f1_score', 'max'), max_trials=20, seed=self.seed),
			hypermodel= self.create_model,
			cv=stratified_inner_cross_val)"""

			
		self.grid = RandomSearch(self.create_model, #lambda hp: create_model_out(hp, input_dim=self.input_dim),#
				objective='f1_score',
				max_trials=3,
				executions_per_trial=3,
				directory='my_dir',
				project_name='hello_world'
				)
		
	def create_model(self, hp=None):
		#grid
		if self.grid_search:
			hidden_units = hp.Int('hidden_units', 16, 128, step=16)
			dropout_rate = hp.Float('dropout_rate', 0.0, 0.5, step=0.25)
			learning_rate = hp.Choice('learning_rate', [1e-2, 1e-3])
		else:
			hidden_units = 16
			dropout_rate = 0.2
			learning_rate = 0.001
		
		# Define input
		inputs = Input(shape=(self.input_dim,))
		
		# Attention layer
		inputs_reshaped = inputs #Reshape((self.input_dim, 1))(inputs)  # Reshape for attention
		#attention_layer = AdditiveAttention()
		attention_layer = DotProductAttention(name="dot_attention")
		attention_output, attention_weights = attention_layer([inputs_reshaped, inputs_reshaped])  # Get both output and weights
		attention_flattened = Flatten()(attention_output)
		
		 # Combine attention with raw inputs
		combined = Concatenate()([inputs, attention_flattened])  # Combine raw and weighted features
	
		# Dense layers
		x = combined
		for _ in range(self.nb_hidden_layers):
			x = Dense(hidden_units)(x)
			x = LeakyReLU()(x)
			x = Dropout(dropout_rate)(x)

		# Output layer
		outputs = Dense(1, activation='sigmoid', name='output')(x)
		
		# Define and compile the model with custom loss function
		def custom_loss(y_true, y_pred):
			output = y_pred[0]  # model output
			attention_weights = y_pred[1]  # attention weights
			return sparse_attention_loss(y_true, output, attention_weights)
			
		# Define and compile the model
		#model = Model(inputs=inputs, outputs=[outputs, attention_weights])
		#model.compile(optimizer=Adam(learning_rate=learning_rate), loss={'output': 'binary_crossentropy', 'dot_attention': None}, metrics=[F1Score(average='weighted')])
		model = Model(inputs=inputs, outputs=outputs)
		model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=[F1Score(average='weighted')])
		return model
	
		
	def build_pipeline(self, X_train):
		self.input_dim = X_train.shape[1]
		verb = 0 if not self.verbose else 1
		model_keras = None # build later with keras
		model = ("classifier", model_keras)
		self.pipeline = super().build_pipeline(model, feature_selection = self.feature_selection, correl_threshold = self.correl_threshold, selectK = self.Kbest_features)
		self.init_grid_search()
		
		
	def grid_fit_predict(self, train_set_features, train_set_labels, test_set_features):
		"""
		#Override super method to account for attention weights.
		#Fit the classifier on the train set using the grid hyperparameter tuning method, and predict on the test set. Return three values: the predictions on both the train and test set, and the probabilities outputed by the classifiers on the test set.
		"""
		
		def custom_fit(hp, model, X, y):
			batch_size = hp.Int('batch_size', 16, 32, step=16)
			epochs = hp.Int('epochs', 50, 100, step=50)
			return model.fit(X, y, batch_size=batch_size,epochs=epochs,verbose=0)
			
		with warnings.catch_warnings(record=True) as w:
			self.pipeline[:-1].fit(train_set_features, train_set_labels)
			X_train_preprocessed = self.pipeline[:-1].transform(train_set_features)
			X_test_preprocessed = self.pipeline[:-1].transform(test_set_features)
			self.input_dim = X_train_preprocessed.shape[1]
			self.init_grid_search()
			
			if self.grid_search:
				self.grid.search(X_train_preprocessed, train_set_labels, custom_fit=custom_fit)
				self.best_params_ = self.grid.get_best_hyperparameters(1)[0]
				self.best_model = self.grid.hypermodel.build(self.best_params_)
				print(vars(self.best_params_))
				self.best_model.fit(X_train_preprocessed, train_set_labels, epochs=self.hyperparameter_grid[0]["classifier__epochs"][0], batch_size=self.hyperparameter_grid[0]["classifier__batch_size"][0], verbose=0)  # Retrain on the entire dataset
			else:
				self.best_model = self.create_model()
				self.best_model.fit(X_train_preprocessed, train_set_labels, batch_size=self.hyperparameter_grid[0]["classifier__batch_size"][0], epochs=self.hyperparameter_grid[0]["classifier__epochs"][0])
				self.best_params_ = self.hyperparameter_grid[0]
			
			grid_predictions_train = self.best_model.predict(X_train_preprocessed)
			grid_predictions_test = self.best_model.predict(X_test_preprocessed)
			grid_proba_test = grid_predictions_test
			grid_proba_train = grid_predictions_train
			
			grid_predictions_train = (grid_predictions_train > 0.5).astype(int)
			grid_predictions_test = (grid_predictions_test > 0.5).astype(int)
			
			grid_predictions_train = np.array(grid_predictions_train).flatten() #flatten for compatibility with sklearn
			grid_predictions_test = np.array(grid_predictions_test).flatten() #flatten for compatibility with sklearn

			if w is not None and len(w) > 0:
				print(print(w[-1].category))
				if self.verbose:
					print(w[-1].message)

		return grid_predictions_train, grid_predictions_test, grid_proba_train, grid_proba_test
	
		



class FeedForwardFusionClassifier(ML_Model):
	
	def __init__(self, nb_hidden_layers = 1, correl_threshold=0.8, Kbest_features = 10, feature_selection = None, grid_search=False, **kwargs): 
		#Feed-forward neural network
		param_grid_ffnn = [{
			'classifier__learning_rate': [0.001],
			'classifier__dropout_rate': [0.2],
			'classifier__hidden_units': [16],
			'classifier__batch_size': [16],
			'classifier__epochs': [25]
		}]
		
		if grid_search:
			param_grid_ffnn = [{
				'classifier__learning_rate': [0.001, 0.01],
				'classifier__dropout_rate': [0.0, 0.25, 0.5],
				'classifier__hidden_units': [16, 32, 64, 128],
				'classifier__batch_size': [16, 32],
				'classifier__epochs': [25, 50]
			}]
		
		self.nb_hidden_layers = nb_hidden_layers
		self.Kbest_features = Kbest_features
		self.feature_selection = feature_selection
		self.input_dim = 0
		
		super().__init__(None, param_grid_ffnn, correl_threshold, grid_search, **kwargs)
		
		
	def modality_column_selection(self, mod):
		ofga_patterns = "gaze_angle_"
		ofp_patterns = "pose_"
		ofau_patterns = " AU"
		of_patterns = r"(gaze_angle_|pose_| AU)"
		
		mixed = r"(screen_prop|down_prop|up_prop)"
		
		fix_pattens = r"fixation(?!.*_)"
		sac_patterns = r"(path|eyemovement)(?!.*_)"
		pup_patterns = r"pupil(?!.*_)"
		hd_patterns = r"distance(?!.*_)"
		et_noaoi = r"(fixation|path|eyemovement|pupil|distance)(?!.*_)"
		et_aoionly = r"_(?!.*(gaze_angle_|pose_| AU|screen_prop|down_prop|up_prop))"
		et_all = r"(?!.*(gaze_angle_|pose_| AU|screen_prop|down_prop|up_prop))"
		
		if mod == "mixed": return mixed
		if mod == "OF_gaze":	return ofga_patterns
		if mod == "OF_pose":	return ofp_patterns
		if mod == "OF_AU":	return ofau_patterns
		if mod == "OF":	return of_patterns
		if mod == "Mixed":	return mixed
		if mod == "ET_fix":	return fix_pattens
		if mod == "ET_sac":	return sac_patterns
		if mod == "ET_pup":	return pup_patterns
		if mod == "ET_hd":	return hd_patterns
		if mod == "ET_noaoi":	return et_noaoi
		if mod == "ET_oiaonly":	return et_aoionly
		if mod == "ET":	return et_all
		
		
		
		
	def init_grid_search(self, scoring ='f1_weighted'):
		"""
		Initialize the GridSearchCV object. see GridSearchCV() documentation in sklearn for the scoring.
		"""
		stratified_inner_cross_val = StratifiedKFold(n_splits = self.nb_inner_folds, shuffle = True, random_state = self.seed)
		"""self.grid = SklearnTuner(
			oracle=BayesianOptimizationOracle(objective=keras_tuner.Objective('val_f1_score', 'max'), max_trials=20, seed=self.seed),
			hypermodel= self.create_model,
			cv=stratified_inner_cross_val)"""
			
		self.grid = RandomSearch(self.create_model, #lambda hp: create_model_out(hp, input_dim=self.input_dim),#
				objective='val_f1_score',
				max_trials=20,
				executions_per_trial=3,
				directory='my_dir',
				project_name='trainFFNN'
				)
				
				
	def attention_block_single(self, inputs):
		# Concatenate the inputs along the last axis
		concatenated = Concatenate(axis=-1)(inputs)

		# Attention weights computation
		attention_weights = Dense(len(inputs), activation='tanh')(concatenated)
		attention_weights = Softmax()(attention_weights)

		# Apply the attention weights to each input
		weighted_inputs = [
			Multiply()([attention_weights[:, i:i+1], inp])
			for i, inp in enumerate(inputs)
		]
		# Combine weighted inputs
		merged_output = Concatenate(axis=-1)(weighted_inputs)
		return merged_output
		
	def attention_block_multi(self, inputs):
		weighted_inputs = []
		for inp in inputs:
			# Compute attention scores for each feature
			attention_scores = Dense(inp.shape[-1], activation='tanh')(inp)
			attention_weights = Softmax()(attention_scores)
			
			# Apply attention weights to the input
			weighted_input = Multiply()([attention_weights, inp])
			weighted_inputs.append(weighted_input)
		
		# Combine weighted inputs
		merged_output = Concatenate(axis=-1)(weighted_inputs)
		return merged_output
		
		
	def create_model(self, hp=None):
		#grid
		if self.grid_search:
			hidden_units1 = hp.Int('hidden_units', 8, 32, step=8)
			hidden_units2 = hp.Int('hidden_units', 8, 32, step=8)
			hidden_units3 = 1
			hidden_units4 = hp.Int('hidden_units', 8, 32, step=8)
			dropout_rate = hp.Float('dropout_rate', 0.0, 0.5, step=0.25)
			learning_rate = hp.Choice('learning_rate', [1e-2, 1e-3])
		else:
			hidden_units1 = 16
			hidden_units2 = 16
			hidden_units3 = 1
			hidden_units4 = 128
			dropout_rate = 0.1
			learning_rate = 0.001
		
		# Define three input layers
		inputET = Input(shape=(self.input_dim_ET,), name="InputET")
		inputOF = Input(shape=(self.input_dim_OF,), name="InputOF")
		inputM = Input(shape=(self.input_dim_Mixed,), name="InputMixed")
	
		# Process each input independently
		processedET = Dense(hidden_units1)(inputET)
		processedET = LeakyReLU()(processedET)
		processedET = Dropout(dropout_rate)(processedET)

		processedOF = Dense(hidden_units2)(inputOF)
		processedOF = LeakyReLU()(processedOF)
		processedOF = Dropout(dropout_rate)(processedOF)
		
		processedM = Dense(hidden_units3)(inputM)
		processedM = LeakyReLU()(processedM)


		# Add hidden layers with dropout
		x = Concatenate()([processedET, processedOF, processedM])
		
		hidden_units_temp = hidden_units4
		for _ in range(self.nb_hidden_layers):
			x = Dense(hidden_units_temp, activation='relu')(x)
			x = LeakyReLU()(x)
			x = Dropout(dropout_rate)(x)
			hidden_units_temp = int(hidden_units_temp / 2)

		# Output layer
		output = Dense(1, activation='sigmoid', name='output')(x)

		# Create model
		model = Model(inputs=[inputET, inputOF, inputM], outputs=output)
		model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=[F1Score(average='weighted')])
		return model
	
		
	def build_pipeline(self, X_train):
		verb = 0 if not self.verbose else 1
		model_keras = None # build later with keras
		model = ("classifier", model_keras)
		self.pipeline = super().build_pipeline(model, feature_selection = self.feature_selection, correl_threshold = self.correl_threshold, selectK = self.Kbest_features)
		self.init_grid_search()

	
	def grid_fit_predict(self, train_set_features, train_set_labels, test_set_features):
		"""
		#Override super method to account for attention weights.
		#Fit the classifier on the train set using the grid hyperparameter tuning method, and predict on the test set. Return three values: the predictions on both the train and test set, and the probabilities outputed by the classifiers on the test set.
		"""
		
		def custom_fit(hp, model, X, y):
			batch_size = hp.Int('batch_size', 16, 32, step=16)
			epochs = hp.Int('epochs', 50, 100, step=50)
			return model.fit(X, y, batch_size=batch_size,epochs=epochs,verbose=0)
			
		with warnings.catch_warnings(record=True) as w:
			train_set_ET_features = train_set_features.loc[:, train_set_features.columns.str.contains(self.modality_column_selection("ET"))]
			train_set_OF_features = train_set_features.loc[:, train_set_features.columns.str.contains(self.modality_column_selection("OF"))]
			train_set_Mixed_features = train_set_features.loc[:, train_set_features.columns.str.contains(self.modality_column_selection("Mixed"))]
			test_set_ET_features = test_set_features.loc[:, test_set_features.columns.str.contains(self.modality_column_selection("ET"))]
			test_set_OF_features = test_set_features.loc[:, test_set_features.columns.str.contains(self.modality_column_selection("OF"))]
			test_set_Mixed_features = test_set_features.loc[:, test_set_features.columns.str.contains(self.modality_column_selection("Mixed"))]
			
			self.pipeline[:-1].fit(train_set_ET_features, train_set_labels)
			X_train_ET_preprocessed = self.pipeline[:-1].transform(train_set_ET_features)
			X_test_ET_preprocessed = self.pipeline[:-1].transform(test_set_ET_features)
			self.input_dim_ET = X_train_ET_preprocessed.shape[1]
			
			self.pipeline[:-1].fit(train_set_OF_features, train_set_labels)
			X_train_OF_preprocessed = self.pipeline[:-1].transform(train_set_OF_features)
			X_test_OF_preprocessed = self.pipeline[:-1].transform(test_set_OF_features)
			self.input_dim_OF = X_train_OF_preprocessed.shape[1]
			
			self.pipeline[1].fit(train_set_Mixed_features, train_set_labels)
			X_train_Mixed_preprocessed = self.pipeline[1].transform(train_set_Mixed_features)
			X_test_Mixed_preprocessed = self.pipeline[1].transform(test_set_Mixed_features)
			self.input_dim_Mixed = X_train_Mixed_preprocessed.shape[1]
			
			self.input_dim = self.input_dim_ET + self.input_dim_OF + self.input_dim_Mixed
						
			fit_train_sets = [X_train_ET_preprocessed, X_train_OF_preprocessed, X_train_Mixed_preprocessed]
			fit_test_sets = [X_test_ET_preprocessed, X_test_OF_preprocessed, X_test_Mixed_preprocessed]
			
			if self.grid_search:
				X_train, X_val, y_train, y_val = train_test_split(X_train_preprocessed, train_set_labels, test_size=0.2, random_state=self.seed)
				self.grid.search(fit_train_sets, train_set_labels, validation_data=(X_val, y_val), batch_size=self.hyperparameter_grid[0]["classifier__batch_size"][0], epochs=self.hyperparameter_grid[0]["classifier__epochs"][0])
				self.best_params_ = self.grid.get_best_hyperparameters(1)[0]
				self.best_model = self.grid.hypermodel.build(self.best_params_)
				self.best_model.fit(fit_train_sets, train_set_labels, epochs=self.hyperparameter_grid[0]["classifier__epochs"][0], batch_size=self.hyperparameter_grid[0]["classifier__batch_size"][0], verbose=0)  # Retrain on the entire dataset
			else:
				self.best_model = self.create_model()
				self.best_model.fit(fit_train_sets, train_set_labels, batch_size=self.hyperparameter_grid[0]["classifier__batch_size"][0], epochs=self.hyperparameter_grid[0]["classifier__epochs"][0])
				self.best_params_ = self.hyperparameter_grid[0]
			
			grid_predictions_train = self.best_model.predict(fit_train_sets)
			grid_predictions_test = self.best_model.predict(fit_test_sets)
			grid_proba_test = grid_predictions_test
			grid_proba_train = grid_predictions_train
			
			grid_predictions_train = (grid_predictions_train > 0.5).astype(int)
			grid_predictions_test = (grid_predictions_test > 0.5).astype(int)
			
			grid_predictions_train = np.array(grid_predictions_train).flatten() #flatten for compatibility with sklearn
			grid_predictions_test = np.array(grid_predictions_test).flatten() #flatten for compatibility with sklearn

			if w is not None and len(w) > 0:
				print(print(w[-1].category))
				if self.verbose:
					print(w[-1].message)

		return grid_predictions_train, grid_predictions_test, grid_proba_train, grid_proba_test


class FeedForwardFusionETDetClassifier(FeedForwardFusionClassifier):
	def create_model(self, hp=None):
		#grid
		if self.grid_search:
			hidden_units1 = hp.Int('hidden_units', 8, 32, step=8)
			hidden_units2 = hp.Int('hidden_units', 8, 32, step=8)
			hidden_units3 = 1
			hidden_units4 = hp.Int('hidden_units', 8, 32, step=8)
			dropout_rate = hp.Float('dropout_rate', 0.0, 0.5, step=0.25)
			learning_rate = hp.Choice('learning_rate', [1e-2, 1e-3])
		else:
			hidden_units1 = 16
			hidden_units2 = 16
			hidden_units3 = 1
			hidden_units4 = 128
			dropout_rate = 0.1
			learning_rate = 0.001
		
		# Define three input layers
		inputETfix = Input(shape=(self.input_dim_ETf,), name="InputETf")
		inputETsac = Input(shape=(self.input_dim_ETs,), name="InputETs")
		inputETpup = Input(shape=(self.input_dim_ETp,), name="InputETp")
		inputEThd = Input(shape=(self.input_dim_EThd,), name="InputEThd")
		inputETaoi = Input(shape=(self.input_dim_ETaoi,), name="InputETaoi")
	
		# Process each input independently
		processedETf = Dense(hidden_units1)(inputETfix)
		processedETf = LeakyReLU()(processedETf)
		processedETf = Dropout(dropout_rate)(processedETf)
		
		processedETs = Dense(hidden_units1)(inputETsac)
		processedETs = LeakyReLU()(processedETs)
		processedETs = Dropout(dropout_rate)(processedETs)
		
		processedETp = Dense(hidden_units1)(inputETpup)
		processedETp = LeakyReLU()(processedETp)
		processedETp = Dropout(dropout_rate)(processedETp)
		
		processedEThd = Dense(hidden_units1)(inputEThd)
		processedEThd = LeakyReLU()(processedEThd)
		processedEThd = Dropout(dropout_rate)(processedEThd)
		
		processedETaoi = Dense(hidden_units1)(inputETaoi)
		processedETaoi = LeakyReLU()(processedETaoi)
		processedETaoi = Dropout(dropout_rate)(processedETaoi)

		# Add hidden layers with dropout
		x = Concatenate()([processedETf, processedETs, processedETp, processedEThd, processedETaoi])
		hidden_units_temp = hidden_units4
		for _ in range(self.nb_hidden_layers):
			x = Dense(hidden_units_temp, activation='relu')(x)
			x = LeakyReLU()(x)
			x = Dropout(dropout_rate)(x)
			hidden_units_temp = int(hidden_units_temp / 2)

		# Output layer
		output = Dense(1, activation='sigmoid', name='output')(x)

		# Create model
		model = Model(inputs=[inputETfix, inputETsac, inputETpup, inputEThd, inputETaoi], outputs=output)
		model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=[F1Score(average='weighted')])
		return model
	
	def grid_fit_predict(self, train_set_features, train_set_labels, test_set_features):
		"""
		#Override super method to account for attention weights.
		#Fit the classifier on the train set using the grid hyperparameter tuning method, and predict on the test set. Return three values: the predictions on both the train and test set, and the probabilities outputed by the classifiers on the test set.
		"""
		
		def custom_fit(hp, model, X, y):
			batch_size = hp.Int('batch_size', 16, 32, step=16)
			epochs = hp.Int('epochs', 50, 100, step=50)
			return model.fit(X, y, batch_size=batch_size,epochs=epochs,verbose=0)
			
		with warnings.catch_warnings(record=True) as w:
			train_set_ETf_features = train_set_features.loc[:, train_set_features.columns.str.contains(self.modality_column_selection("ET_fix"))]
			train_set_ETs_features = train_set_features.loc[:, train_set_features.columns.str.contains(self.modality_column_selection("ET_sac"))]
			train_set_ETp_features = train_set_features.loc[:, train_set_features.columns.str.contains(self.modality_column_selection("ET_pup"))]
			train_set_EThd_features = train_set_features.loc[:, train_set_features.columns.str.contains(self.modality_column_selection("ET_hd"))]
			train_set_ETaoi_features = train_set_features.loc[:, train_set_features.columns.str.contains(self.modality_column_selection("ET_oiaonly"))]
			
			test_set_ETf_features = test_set_features.loc[:, test_set_features.columns.str.contains(self.modality_column_selection("ET_fix"))]
			test_set_ETs_features = test_set_features.loc[:, test_set_features.columns.str.contains(self.modality_column_selection("ET_sac"))]
			test_set_ETp_features = test_set_features.loc[:, test_set_features.columns.str.contains(self.modality_column_selection("ET_pup"))]
			test_set_EThd_features = test_set_features.loc[:, test_set_features.columns.str.contains(self.modality_column_selection("ET_hd"))]
			test_set_ETaoi_features = test_set_features.loc[:, test_set_features.columns.str.contains(self.modality_column_selection("ET_oiaonly"))]
			
			
			self.pipeline[:-1].fit(train_set_ETf_features, train_set_labels)
			X_train_ETf_preprocessed = self.pipeline[:-1].transform(train_set_ETf_features)
			X_test_ETf_preprocessed = self.pipeline[:-1].transform(test_set_ETf_features)
			self.input_dim_ETf = X_train_ETf_preprocessed.shape[1]
			
			self.pipeline[:-1].fit(train_set_ETs_features, train_set_labels)
			X_train_ETs_preprocessed = self.pipeline[:-1].transform(train_set_ETs_features)
			X_test_ETs_preprocessed = self.pipeline[:-1].transform(test_set_ETs_features)
			self.input_dim_ETs = X_train_ETs_preprocessed.shape[1]
			
			self.pipeline[:-1].fit(train_set_ETp_features, train_set_labels)
			X_train_ETp_preprocessed = self.pipeline[:-1].transform(train_set_ETp_features)
			X_test_ETp_preprocessed = self.pipeline[:-1].transform(test_set_ETp_features)
			self.input_dim_ETp = X_train_ETp_preprocessed.shape[1]
			
			self.pipeline[:-1].fit(train_set_EThd_features, train_set_labels)
			X_train_EThd_preprocessed = self.pipeline[:-1].transform(train_set_EThd_features)
			X_test_EThd_preprocessed = self.pipeline[:-1].transform(test_set_EThd_features)
			self.input_dim_EThd = X_train_EThd_preprocessed.shape[1]
			
			self.pipeline[:-1].fit(train_set_ETaoi_features, train_set_labels)
			X_train_ETaoi_preprocessed = self.pipeline[:-1].transform(train_set_ETaoi_features)
			X_test_ETaoi_preprocessed = self.pipeline[:-1].transform(test_set_ETaoi_features)
			self.input_dim_ETaoi = X_train_ETaoi_preprocessed.shape[1]
			
			self.input_dim = self.input_dim_ETf + self.input_dim_ETs + self.input_dim_ETp + self.input_dim_EThd + self.input_dim_ETaoi
						
			fit_train_sets = [X_train_ETf_preprocessed, X_train_ETs_preprocessed, X_train_ETp_preprocessed, X_train_EThd_preprocessed, X_train_ETaoi_preprocessed]
			fit_test_sets = [X_test_ETf_preprocessed, X_test_ETs_preprocessed, X_test_ETp_preprocessed, X_test_EThd_preprocessed, X_test_ETaoi_preprocessed]
			
			if self.grid_search:
				X_train, X_val, y_train, y_val = train_test_split(X_train_preprocessed, train_set_labels, test_size=0.2, random_state=self.seed)
				self.grid.search(fit_train_sets, train_set_labels, validation_data=(X_val, y_val), batch_size=self.hyperparameter_grid[0]["classifier__batch_size"][0], epochs=self.hyperparameter_grid[0]["classifier__epochs"][0])
				self.best_params_ = self.grid.get_best_hyperparameters(1)[0]
				self.best_model = self.grid.hypermodel.build(self.best_params_)
				self.best_model.fit(fit_train_sets, train_set_labels, epochs=self.hyperparameter_grid[0]["classifier__epochs"][0], batch_size=self.hyperparameter_grid[0]["classifier__batch_size"][0], verbose=0)  # Retrain on the entire dataset
			else:
				self.best_model = self.create_model()
				self.best_model.fit(fit_train_sets, train_set_labels, batch_size=self.hyperparameter_grid[0]["classifier__batch_size"][0], epochs=self.hyperparameter_grid[0]["classifier__epochs"][0])
				self.best_params_ = self.hyperparameter_grid[0]
			
			grid_predictions_train = self.best_model.predict(fit_train_sets)
			grid_predictions_test = self.best_model.predict(fit_test_sets)
			grid_proba_test = grid_predictions_test
			grid_proba_train = grid_predictions_train
			
			grid_predictions_train = (grid_predictions_train > 0.5).astype(int)
			grid_predictions_test = (grid_predictions_test > 0.5).astype(int)
			
			grid_predictions_train = np.array(grid_predictions_train).flatten() #flatten for compatibility with sklearn
			grid_predictions_test = np.array(grid_predictions_test).flatten() #flatten for compatibility with sklearn

			if w is not None and len(w) > 0:
				print(print(w[-1].category))
				if self.verbose:
					print(w[-1].message)

		return grid_predictions_train, grid_predictions_test, grid_proba_train, grid_proba_test
		
		
class FeedForwardFusionAttClassifier(FeedForwardFusionClassifier):
	def create_model(self, hp=None):
		#grid
		if self.grid_search:
			hidden_units1 = hp.Int('hidden_units', 8, 32, step=8)
			hidden_units2 = hp.Int('hidden_units', 8, 32, step=8)
			hidden_units3 = 1
			hidden_units4 = hp.Int('hidden_units', 8, 32, step=8)
			dropout_rate = hp.Float('dropout_rate', 0.0, 0.5, step=0.25)
			learning_rate = hp.Choice('learning_rate', [1e-2, 1e-3])
		else:
			hidden_units1 = 16
			hidden_units2 = 16
			hidden_units3 = 1
			hidden_units4 = 128
			dropout_rate = 0.1
			learning_rate = 0.001
		
		# Define three input layers
		inputET = Input(shape=(self.input_dim_ET,), name="InputET")
		inputOF = Input(shape=(self.input_dim_OF,), name="InputOF")
		inputM = Input(shape=(self.input_dim_Mixed,), name="InputMixed")
	
		# Process each input independently
		processedET = Dense(hidden_units1)(inputET)
		processedET = LeakyReLU()(processedET)
		processedET = Dropout(dropout_rate)(processedET)

		processedOF = Dense(hidden_units2)(inputOF)
		processedOF = LeakyReLU()(processedOF)
		processedOF = Dropout(dropout_rate)(processedOF)
		
		processedM = Dense(hidden_units3)(inputM)
		processedM = LeakyReLU()(processedM)

		# Apply attention mechanism
		attention_output = self.attention_block_single([processedET, processedOF, processedM])
		attention_output = Dropout(dropout_rate)(attention_output)

		# Add hidden layers with dropout
		x = attention_output
		hidden_units_temp = hidden_units4
		for _ in range(self.nb_hidden_layers):
			x = Dense(hidden_units_temp, activation='relu')(x)
			x = LeakyReLU()(x)
			x = Dropout(dropout_rate)(x)
			hidden_units_temp = int(hidden_units_temp / 2)

		# Output layer
		output = Dense(1, activation='sigmoid', name='output')(x)

		# Create model
		model = Model(inputs=[inputET, inputOF, inputM], outputs=output)
		model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=[F1Score(average='weighted')])
		return model
		
		
class FeedForwardFusionETDetAttClassifier(FeedForwardFusionETDetClassifier):
	def create_model(self, hp=None):
		#grid
		if self.grid_search:
			hidden_units1 = hp.Int('hidden_units', 8, 32, step=8)
			hidden_units2 = hp.Int('hidden_units', 8, 32, step=8)
			hidden_units3 = 1
			hidden_units4 = hp.Int('hidden_units', 8, 32, step=8)
			dropout_rate = hp.Float('dropout_rate', 0.0, 0.5, step=0.25)
			learning_rate = hp.Choice('learning_rate', [1e-2, 1e-3])
		else:
			hidden_units1 = 16
			hidden_units2 = 16
			hidden_units3 = 1
			hidden_units4 = 128
			dropout_rate = 0.1
			learning_rate = 0.001
		
		# Define three input layers
		inputETfix = Input(shape=(self.input_dim_ETf,), name="InputETf")
		inputETsac = Input(shape=(self.input_dim_ETs,), name="InputETs")
		inputETpup = Input(shape=(self.input_dim_ETp,), name="InputETp")
		inputEThd = Input(shape=(self.input_dim_EThd,), name="InputEThd")
		inputETaoi = Input(shape=(self.input_dim_ETaoi,), name="InputETaoi")
	
		# Process each input independently
		processedETf = Dense(hidden_units1)(inputETfix)
		processedETf = LeakyReLU()(processedETf)
		processedETf = Dropout(dropout_rate)(processedETf)
		
		processedETs = Dense(hidden_units1)(inputETsac)
		processedETs = LeakyReLU()(processedETs)
		processedETs = Dropout(dropout_rate)(processedETs)
		
		processedETp = Dense(hidden_units1)(inputETpup)
		processedETp = LeakyReLU()(processedETp)
		processedETp = Dropout(dropout_rate)(processedETp)
		
		processedEThd = Dense(hidden_units1)(inputEThd)
		processedEThd = LeakyReLU()(processedEThd)
		processedEThd = Dropout(dropout_rate)(processedEThd)
		
		processedETaoi = Dense(hidden_units1)(inputETaoi)
		processedETaoi = LeakyReLU()(processedETaoi)
		processedETaoi = Dropout(dropout_rate)(processedETaoi)

		# Apply attention mechanism
		attention_output = self.attention_block_single([processedETf, processedETs, processedETp, processedEThd, processedETaoi])
		attention_output = Dropout(dropout_rate)(attention_output)

		# Add hidden layers with dropout
		x = attention_output
		hidden_units_temp = hidden_units4
		for _ in range(self.nb_hidden_layers):
			x = Dense(hidden_units_temp, activation='relu')(x)
			x = LeakyReLU()(x)
			x = Dropout(dropout_rate)(x)
			hidden_units_temp = int(hidden_units_temp / 2)

		# Output layer
		output = Dense(1, activation='sigmoid', name='output')(x)

		# Create model
		model = Model(inputs=[inputETfix, inputETsac, inputETpup, inputEThd, inputETaoi], outputs=output)
		model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=[F1Score(average='weighted')])
		return model
