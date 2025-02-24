"""
AI fairness analysis code
@author XXX
@email XXX
@date 01/02/2024
@licence BSD-3-Clause 
"""

import pandas as pd
import numpy as np
import pickle

import os
import sys
import glob
import warnings
import re
from collections import OrderedDict

from sklearn.base import BaseEstimator, TransformerMixin 
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GroupShuffleSplit
# from modules.ordinal import OrdinalClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier


from sklearn.dummy import DummyClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.feature_selection import VarianceThreshold
from imblearn.over_sampling import SMOTE 
from copy import deepcopy

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.metrics import fbeta_score

from tensorflow.keras.models import clone_model

from sklearn.exceptions import UndefinedMetricWarning

# adding src to the system path to find scripts in there
sys.path.insert(0, "./src")
sys.path.insert(0, "./modules")

# local imports
from ML_models import *

from threshold_optimizer import ThresholdOptimizer
import maddlib
import fair_performances
import compute_abroca as abroca

############################################################################################################################################################
#CONSTANT
############################################################################################################################################################
RAND_STATE_SEED = 2204 #for reproducibility 2204
EXPORT_SUFFIX = "1" #suffix to add to all exported file (for version control)
NBFOLDS = 8 # number of cross validation folds
NBFOLDS_GRID = 5 # number of inner folds for the grid search hyperparameter tuning
GRIDSEARCH = False #True do to grid-search hyperparameter tuning
CORREL_THRESHOLD = 0.8 # thresholds for discarding correlated features (1 means no feature removal) 
FEATSELECT = "PCA" # "Kbest" for PCA feature selection, "Kbest" for statistical-based feature selection, None otherwise
KBEST = 30 # Top K feature for Kbest feature selection
SMOTE_MINORITY = True # True/False: to turn on/off Smote over-sampling of the minority classes
VERBOSE = False # True/False

LABEL_KEY = "Answer" # the key of the labels in the feature dataframe (successfully passed the course)
LABEL_VALUES = [0, 1] #the values of the label (attention/memory: 0 for no, 1 for yes)
STUDENT_ID_KEY = "Part_id" # the grouping key used to do cross-validation over users (effectively using the StratifiedGroupKFold function of Sklearn)

DIR_FEATURES = "features"
SENSITIVE_ATTRIBUTES = ["Gender", "Tiredness", "PriorKnowledge"]
TARGETS = ["QCM", "SelfReports"]
WINDOWS = [10, 20, 30, "Content"]
#AOIS = ["useraoi_importance", "useraoi_types", "useraoi_type_importance", "useraoi_detailed", "staticAOI4x4", "staticAOI3x3", "staticAOI2x1"]
AOIS = ["useraoi_importance", "useraoi_types", "useraoi_type_importance", "staticAOI2x1"]
MODALITIES = ["ET_All", "ET_AOIonly", "ET_noAOI", "ET_fixations", "ET_saccades","ET_pupils","ET_headdistance"]
FUSION_STYLE = ["pre", "in", "post"] #["pre", "in", "post"]

DIR_OUTPUT = "output"
DIR_PICKLE = "pickle"

###################################################################################################################################
#FUNCTIONS
###################################################################################################################################


def list_feature_sets(modalities, targets, windows, aois, prefix = "MLfeatures"):
	"""
	Create and return a list with the name of the files for all feature sets, which are a combination of their modality, window, AOI and target.
	"""
	feature_sets_list = []
	for m in modalities:
		for t in targets:
			for window in windows:
				if "multimodal_All" in m or "AOIonly" in m or "ET_All" in m:
					for aoi in aois:
						feature_sets_list.append( ("_".join([prefix, m, t, str(window), aoi]))+".csv")
				else:
					feature_sets_list.append( ("_".join([prefix, m, t, str(window), "noAOI"]))+".csv")
	return feature_sets_list
	

def list_fusion_feature_sets(modalities, targets, windows, aois, prefix = "MLfeatures"):
	"""
	Create and return a list with the name of the files for all feature sets, which are a combination of their modality, window, AOI and target.
	"""
	feature_sets_list = []
	m = "multimodal_All"
	for t in targets:
		for window in windows:
			for aoi in aois:
				feature_sets_list.append( ("_".join([prefix, m, t, str(window), aoi]))+".csv")
	return feature_sets_list
	
def clean_up_features(df):
	"""
	Cleanup features based on feature inspection: remove unused events, duplicates events, and combine similar, less used events
	"""
	dfout = df.drop(list(df.filter(['Sc_id'])), axis=1)
	dfout = dfout.fillna(0)
	return dfout

def merge_features_label(df_features, df_labels):
	"""
	Merge the features dataframe with the label to predict, by joining on the STUDENT_ID_KEY
	"""
	df_labels2 = df_labels[[STUDENT_ID_KEY, LABEL_KEY]]
	return pd.merge(df_features, df_labels2, on = STUDENT_ID_KEY, how ='inner')


def merge_features_demo(df_features, df_demo):
	"""
	Merge the features dataframe with the demographics' one to predict, by joining on the STUDENT_ID_KEY
	"""
	return pd.merge(df_features, df_demo, on = STUDENT_ID_KEY, how ='inner')


def create_k_folds(df):
	"""
	Create the train and test sets for k-folds cross-validation, and return a dict with the weeks as the key and the index of the cv folds as the value.
	Uses the StratifiedGroupKFold method of sklearn to create the folds using student_id as the group, and returns the index of the sample within each fold.
	"""
	cv_split_indices = []

	#for week, week_features in dfweeks:
	cv = StratifiedGroupKFold(n_splits = NBFOLDS, shuffle=True, random_state = RAND_STATE_SEED)
	
	for train_set_index, test_set_index in cv.split(df, y = df[LABEL_KEY], groups=df[STUDENT_ID_KEY]):
		cv_split_indices.append( (train_set_index, test_set_index) )
	return cv_split_indices


def feature_label_split(dataset):
	""" Split the features and label for the data. Also remove irrelevant columns."""
	dataset_features = dataset.drop(columns=[LABEL_KEY,STUDENT_ID_KEY])
	dataset_labels = dataset[LABEL_KEY].copy()
	return (dataset_features, dataset_labels)  


def th_scoring(th, y, prob):
	pred = (prob > th).astype(int)
	return 0 if not pred.any() else -fbeta_score(y, pred, beta=0.1) 


def calculate_roc_auc(y_true, y_pred, average='weighted'):
    if len(set(y_true)) == 1:  # Only one class present in y_true
        return ""
    return roc_auc_score(y_true, y_pred, average=average)
    
    
def init_classifiers():
	#Create the final list of all classifiers
	kwargs = {"nb_inner_folds": NBFOLDS_GRID, "labels": LABEL_VALUES, "verbose": VERBOSE, "seed": RAND_STATE_SEED}
	classifiers = {
		"Base1": StratifiedBaselineClassifier(**kwargs),
		"Base2": MajorityClassBaselineClassifier(**kwargs),
		"NB": NaiveBayesClassifier(CORREL_THRESHOLD, KBEST, FEATSELECT, GRIDSEARCH, **kwargs),
		#"KNN": KNeirestNeighboorsClassifier(CORREL_THRESHOLD, KBEST, FEATSELECT, GRIDSEARCH, **kwargs),
		"LR": LogisticClassifier(CORREL_THRESHOLD, KBEST, FEATSELECT, GRIDSEARCH, **kwargs),
		"SVM": SVMClassifier(CORREL_THRESHOLD, KBEST, FEATSELECT, GRIDSEARCH, **kwargs),
		#"SGB": StochasticGradientBoostingAlgorithmClassifier(CORREL_THRESHOLD, KBEST, FEATSELECT, GRIDSEARCH, **kwargs),
		"RF": RandomForestEnsembleClassifier(CORREL_THRESHOLD, KBEST, FEATSELECT, GRIDSEARCH, **kwargs),
		#"MLP": MultiLayerPerceptronClassifier(CORREL_THRESHOLD, KBEST, FEATSELECT, GRIDSEARCH, **kwargs),
		"FFNN1": FeedForwardClassifier(1, CORREL_THRESHOLD, KBEST, FEATSELECT, GRIDSEARCH, **kwargs),
		"FFNN2": FeedForwardClassifier(2, CORREL_THRESHOLD, KBEST, FEATSELECT, GRIDSEARCH, **kwargs),
		"FFNNAtt1": FeedForwardSelfAttentionClassifier(1, CORREL_THRESHOLD, KBEST, FEATSELECT, GRIDSEARCH, **kwargs),
		"FFNNAtt2": FeedForwardSelfAttentionClassifier(2, CORREL_THRESHOLD, KBEST, FEATSELECT, GRIDSEARCH, **kwargs)
	}
	return classifiers

def init_fusion_models():
	#Create the final list of all models for modality fusion
	kwargs = {"nb_inner_folds": NBFOLDS_GRID, "labels": LABEL_VALUES, "verbose": VERBOSE, "seed": RAND_STATE_SEED}
	models = {
		"Base1": StratifiedBaselineClassifier(**kwargs),
		"Base2": MajorityClassBaselineClassifier(**kwargs),
		"FFDETC1": FeedForwardFusionETDetClassifier(1, CORREL_THRESHOLD, KBEST, FEATSELECT, GRIDSEARCH, **kwargs),
		"FFDETC2": FeedForwardFusionETDetClassifier(2, CORREL_THRESHOLD, KBEST, FEATSELECT, GRIDSEARCH, **kwargs),
		"FFDETCA1": FeedForwardFusionETDetAttClassifier(1, CORREL_THRESHOLD, KBEST, FEATSELECT, GRIDSEARCH, **kwargs),
		"FFDETCA2": FeedForwardFusionETDetAttClassifier(2, CORREL_THRESHOLD, KBEST, FEATSELECT, GRIDSEARCH, **kwargs)
	}
	return models

def init_ensemble_models():
	#Create the final list of all ensemble models
	kwargs = {"nb_inner_folds": NBFOLDS_GRID, "labels": LABEL_VALUES, "verbose": VERBOSE, "seed": RAND_STATE_SEED}
	models = {
		"Base1": StratifiedBaselineClassifier(**kwargs),
		"Base2": MajorityClassBaselineClassifier(**kwargs),
		"ENSV": SoftVotingClassifier(CORREL_THRESHOLD, KBEST, FEATSELECT, GRIDSEARCH, **kwargs),
		"ENSDT": CustomStackingClassifier(DecisionTreeClassifier(max_depth=4), CORREL_THRESHOLD, KBEST, FEATSELECT, GRIDSEARCH, **kwargs),
	}
	return models
	
def results_classifiers(fset, k, clf, hyperparam, train_set_labels, test_set_labels, predictions_train, predictions_test, predictions2_test, nb_features_in, res_madd, res_abroca, res_groupf1, prefix=""):
	regout = re.match(r"MLfeatures_([^_]+)_([^_]+)_([^_]+)_([^_]+)_(.+)", fset)
	modality = ""
	featuretypes = ""
	target = ""
	window = ""
	aoi = ""
	if regout:
		# Extracted components
		modality = regout.group(1)
		featuretypes = regout.group(2)
		target = regout.group(3)
		window = regout.group(4)
		aoi = regout.group(5)
    
	d = OrderedDict( {
		"feat_set": prefix+fset,
		"target": target,
		"modality": modality,
		"features_type": featuretypes,
		"window": window,
		"aoi": aoi,
		"K": k,
		"classifier": clf,
		"grid_params": hyperparam,
		"accuracy_train": accuracy_score(train_set_labels, predictions_train),
		"f1_train": f1_score(train_set_labels, predictions_train, average='weighted'),
		"auc_train": calculate_roc_auc(train_set_labels, predictions_train, average='weighted'),
		"accuracy": accuracy_score(test_set_labels, predictions_test),
		"precision": precision_score(test_set_labels, predictions_test, average='weighted'),
		"recall": recall_score(test_set_labels, predictions_test, average='weighted'),
		"f1_score": f1_score(test_set_labels, predictions_test, average='weighted'),
		"f1_score_th": f1_score(test_set_labels, predictions2_test, average='weighted'),
		"AUC_score": calculate_roc_auc(test_set_labels, predictions_test, average='weighted'),
		"AUC_score_th": calculate_roc_auc(test_set_labels, predictions2_test, average='weighted'),
		"features_in": nb_features_in
	})
	#add madd values
	for i in range(len(SENSITIVE_ATTRIBUTES)):
		if i < len(res_madd):
			d["MADD_"+SENSITIVE_ATTRIBUTES[i]] = res_madd[i]
	#add abroca values
	for i in range(len(SENSITIVE_ATTRIBUTES)):
		if i < len(res_abroca):
			d["ABROCA_"+SENSITIVE_ATTRIBUTES[i]] = res_abroca[i]
	#add groupf1 values
	for i in range(len(SENSITIVE_ATTRIBUTES)):
		if i < len(res_groupf1):
			d["GROUPF1_"+SENSITIVE_ATTRIBUTES[i]+"_G0"] = res_groupf1[i][0]
			d["GROUPF1_"+SENSITIVE_ATTRIBUTES[i]+"_G1"] = res_groupf1[i][1]
	return d


def train_models_single_dataset(dfInput, classifiers, sensitive_features = [], use_sensitive_features = False):
	"""
	Train a set of classifiers on the provided dfInput data.
	dfInput: a dictionary of pandas dataframes where the keys are the name of the feature sets and the value the corresponding dataframe.
	classifiers: a list of classifiers initialized with the init_classifiers() function
	returns: a dataframe with the prediction performance of the trained classifiers, whose rows are formatted with the results_classifiers() function
	"""
	levels_results_grid = {}
			
	fold_results = []
	pred_results = {}
	pred_proba = {}
	pred_labels = {}
	grid_classifier = {}

	# Iterate over the feature sets and fold to train the models
	for fset, df_features in dfInput.items():
		print(f"------------------ Feature set {fset} ------------------")
		
		pred_results[fset] = {}
		pred_proba[fset] = {}
		pred_labels[fset] = {}
		grid_classifier[fset] = {}
		cv_split_indices = create_k_folds(df_features) #create the k folds (outer loop)

		for k in range(NBFOLDS):
			print(f"------------------ K {k} ------------------")
			
			(train_set_index, test_set_index) = cv_split_indices[k]
			train_set = df_features.iloc[train_set_index]
			test_set = df_features.iloc[test_set_index]
			
			if VERBOSE:
				print ("Label distribution:")
				print("-Train", train_set[LABEL_KEY].value_counts())
				print("-Test", test_set[LABEL_KEY].value_counts())
				
				
			train_set_features, train_set_labels  = feature_label_split(train_set)
			test_set_features, test_set_labels  = feature_label_split(test_set)

			if SMOTE_MINORITY: # Oversample minority classes, in the train sets ONLY
				smoter = SMOTE(random_state = RAND_STATE_SEED)
				(train_set_features, train_set_labels) = smoter.fit_resample(train_set_features, train_set_labels)
				
			for classifier_name, classifier in classifiers.items():
				if classifier_name not in pred_results[fset]: pred_results[fset][classifier_name] = [] 
				if classifier_name not in pred_proba[fset]: pred_proba[fset][classifier_name] = [] 
				if classifier_name not in pred_labels[fset]: pred_labels[fset][classifier_name] = []
				if classifier_name not in grid_classifier[fset]: grid_classifier[fset][classifier_name] = []
				
				if VERBOSE:
					print(train_set_features.info())
					print(test_set_features.info())
					print(train_set_labels.shape)
					print(test_set_labels.shape)
				

				if "Base" in classifier_name: # No grid search for the baselines
					grid_predictions_train, grid_predictions, _, grid_proba_test = classifier.fit_predict(train_set_features, train_set_labels, test_set_features)
					res = results_classifiers(fset, k, classifier_name, "", train_set_labels, test_set_labels, grid_predictions_train, grid_predictions, grid_predictions, 0, [], [], [])
					
					if VERBOSE:
						print("Baseline res", res)
					
					fold_results.append(res)
					pred_results[fset][classifier_name] += list(grid_predictions)
					pred_proba[fset][classifier_name] += list(grid_proba_test)
					pred_labels[fset][classifier_name] += list(test_set_labels)  
				else:
					# Build pipeline for Keras model:
					if "FFNN" in classifier_name:
						classifier.build_pipeline(train_set_features)
					
					if use_sensitive_features:
						grid_predictions_train, grid_predictions, grid_proba_train, grid_proba_test = classifier.grid_fit_predict(train_set_features, train_set_labels, test_set_features)
					else:
						grid_predictions_train, grid_predictions, grid_proba_train, grid_proba_test = classifier.grid_fit_predict(train_set_features.loc[:, ~train_set_features.columns.isin(sensitive_features)], train_set_labels, test_set_features.loc[:, ~test_set_features.columns.isin(sensitive_features)])
					nb_used_features = classifier.grid.best_estimator_['classifier'].n_features_in_ if "FFNN" not in classifier_name else classifier.input_dim
					best_params = classifier.grid.best_params_ if "FFNN" not in classifier_name else classifier.best_params_
				 
					if "FFNN" not in classifier_name: #threshold optimization makes sense only for sklearn models
						grid_predictions2 = classifier.classification_threshold_optimizer(grid_proba_train, train_set_labels, grid_proba_test)
					else:
						grid_predictions2 = grid_predictions
						
					#fairness eval
					res_madd = []
					res_abroca = []
					res_groupf1 = []
					df_madd = test_set_features.copy(deep=True).reset_index() #need to reset index for the MADD to work
					df_madd["pred_proba"] = grid_proba_test[:, 1]  if "FFNN" not in classifier_name else grid_proba_test[:, 0] 	#for abroca
					df_madd["final_result"] = test_set_labels.to_numpy() #for abroca

					g1 = 1
					g0 = 0
					for sattr in sensitive_features:
						nbg1 = df_madd[sattr].value_counts().get(g1, 0)
						nbg0 = df_madd[sattr].value_counts().get(g0, 0)
						
						if nbg1 == 0 or nbg0 == 0: # if one gorup is empty, skip as fairness metric would not make sense
							res_madd.append("")
							res_abroca.append("")
							res_groupf1.append(["", ""])
							continue
							
						#MADD
						res_madd.append(maddlib.MADD(h='auto', X_test=df_madd, pred_proba=df_madd["pred_proba"], sf=sattr, model=fset+"_"+classifier_name))

						#ABROCA
						if nbg1 > nbg0:  # select by index value (class 0 or class 1)
							majority_group = 1
						else:
							majority_group = 0
						res_abroca.append(abroca.compute_abroca(
													   df=df_madd,
													   pred_col='pred_proba',
													   label_col='final_result',
													   protected_attr_col=sattr,
													   majority_protected_attr_val=majority_group,
													   n_grid=10000,
													   plot_slices=False
													   ))
						
						#group F1
						res_groupf1.append(fair_performances.group_performances(X_test=df_madd, Y_test=df_madd["final_result"], pred=grid_predictions, sf=sattr))
					del df_madd

					res = results_classifiers(fset, k, classifier_name, str(best_params), train_set_labels, test_set_labels, grid_predictions_train, grid_predictions, grid_predictions2, nb_used_features, res_madd, res_abroca, res_groupf1)
					
					fold_results.append(res)
					
					#save models as we go
					if "FFNN" in classifier_name: #Keras with attention
						savepath = os.path.join(DIR_PICKLE, "models", fset+"_"+classifier_name+"_"+str(np.random.random())+".h5")
						classifier.best_model.save(savepath)
						grid_classifier[fset][classifier_name].append(savepath)
					#elif "FFNN" in classifier_name: #clone Keras model
					#	savepath = os.path.join(DIR_PICKLE, "models", fset+"_"+classifier_name+"_"+str(np.random.random())+".h5")
					#	classifier.grid.best_estimator_['classifier'].model.save(savepath)
					#	grid_classifier[fset][classifier_name].append(savepath)
					else:
						savepath = os.path.join(DIR_PICKLE, "models", fset+"_"+classifier_name+"_"+str(np.random.random())+".skpkl")
						with open(savepath, "wb") as file:
							pickle.dump(classifier.grid.best_estimator_['classifier'], file)
						grid_classifier[fset][classifier_name].append(savepath)
					

					#pred_results[fset][classifier_name] += list(grid_predictions)
					#pred_proba[fset][classifier_name] += list(grid_proba_test)
					#pred_labels[fset][classifier_name] += list(test_set_labels)
					
					if VERBOSE: #prints for debug:
						print(classifier_name, "res", res)
						#print(classifier.grid.best_estimator_)
						print(str(best_params))
						
			#output results and pickles
			df_results = pd.DataFrame.from_records(fold_results)
			output_name = "dfres_" + str(NBFOLDS)+"folds_" + ("grid_" if GRIDSEARCH else "nogrid_") + ("smote_" if SMOTE_MINORITY else "nosmote_") + (str(FEATSELECT)+"_" if FEATSELECT else "nofeatselect_")+ str(CORREL_THRESHOLD)+"correl_" + str(RAND_STATE_SEED)+"seed"+EXPORT_SUFFIX
			df_results.to_csv(f"./{DIR_OUTPUT}/{output_name}.csv", index=False)
			df_results.to_pickle(f"./{DIR_PICKLE}/{output_name}.pkl")

			with open(f"./{DIR_PICKLE}/{output_name}_models.pkl", "wb") as file:
				pickle.dump(grid_classifier, file)

	print("Single model training done.")
	return pd.DataFrame.from_records(fold_results), grid_classifier


def train_models_fusion(dfInput, models, sensitive_features = [], use_sensitive_features = False):
	"""
	Train a set of classifiers on the provided dfInput data.
	dfInput: a dictionary of pandas dataframes where the keys are the name of the feature sets and the value the corresponding dataframe.
	models: a list of models initialized with the init_fusion_models() function
	returns: a dataframe with the prediction performance of the trained classifiers, whose rows are formatted with the results_classifiers() function
	"""
	levels_results_grid = {}
			
	fold_results = []
	pred_results = {}
	pred_proba = {}
	pred_labels = {}
	grid_classifier = {}

	# Iterate over the feature sets and fold to train the models
	for fset, df_features in dfInput.items():
		print(f"------------------ Feature set {fset} ------------------")
		
		pred_results[fset] = {}
		pred_proba[fset] = {}
		pred_labels[fset] = {}
		grid_classifier[fset] = {}
		cv_split_indices = create_k_folds(df_features) #create the k folds (outer loop)

		for k in range(NBFOLDS):
			print(f"------------------ K {k} ------------------")
			
			(train_set_index, test_set_index) = cv_split_indices[k]
			train_set = df_features.iloc[train_set_index]
			test_set = df_features.iloc[test_set_index]
			
			if VERBOSE:
				print ("Label distribution:")
				print("-Train", train_set[LABEL_KEY].value_counts())
				print("-Test", test_set[LABEL_KEY].value_counts())
				
				
			train_set_features, train_set_labels  = feature_label_split(train_set)
			test_set_features, test_set_labels  = feature_label_split(test_set)

			if SMOTE_MINORITY: # Oversample minority classes, in the train sets ONLY
				smoter = SMOTE(random_state = RAND_STATE_SEED)
				(train_set_features, train_set_labels) = smoter.fit_resample(train_set_features, train_set_labels)
				
			for classifier_name, classifier in models.items():
				if classifier_name not in pred_results[fset]: pred_results[fset][classifier_name] = [] 
				if classifier_name not in pred_proba[fset]: pred_proba[fset][classifier_name] = [] 
				if classifier_name not in pred_labels[fset]: pred_labels[fset][classifier_name] = []
				if classifier_name not in grid_classifier[fset]: grid_classifier[fset][classifier_name] = []
				
				if VERBOSE:
					print(train_set_features.info())
					print(test_set_features.info())
					print(train_set_labels.shape)
					print(test_set_labels.shape)
				

				if "Base" in classifier_name: # No grid search for the baselines
					grid_predictions_train, grid_predictions, _, grid_proba_test = classifier.fit_predict(train_set_features, train_set_labels, test_set_features)
					res = results_classifiers(fset, k, classifier_name, "", train_set_labels, test_set_labels, grid_predictions_train, grid_predictions, grid_predictions, 0, [], [], [])
					
					if VERBOSE:
						print("Baseline res", res)
					
					fold_results.append(res)
					pred_results[fset][classifier_name] += list(grid_predictions)
					pred_proba[fset][classifier_name] += list(grid_proba_test)
					pred_labels[fset][classifier_name] += list(test_set_labels)  
				else:
					# Build pipeline for Keras model:
					classifier.build_pipeline(train_set_features)
					
					if use_sensitive_features:
						grid_predictions_train, grid_predictions, grid_proba_train, grid_proba_test = classifier.grid_fit_predict(train_set_features, train_set_labels, test_set_features)
					else:
						grid_predictions_train, grid_predictions, grid_proba_train, grid_proba_test = classifier.grid_fit_predict(train_set_features.loc[:, ~train_set_features.columns.isin(sensitive_features)], train_set_labels, test_set_features.loc[:, ~test_set_features.columns.isin(sensitive_features)])
					nb_used_features = classifier.input_dim
					best_params = classifier.best_params_
				 
					grid_predictions2 = grid_predictions
						
					#fairness eval
					res_madd = []
					res_abroca = []
					res_groupf1 = []
					df_madd = test_set_features.copy(deep=True).reset_index() #need to reset index for the MADD to work
					df_madd["pred_proba"] = grid_proba_test[:, 0] 	#for abroca
					df_madd["final_result"] = test_set_labels.to_numpy() #for abroca

					g1 = 1
					g0 = 0
					for sattr in sensitive_features:
						nbg1 = df_madd[sattr].value_counts().get(g1, 0)
						nbg0 = df_madd[sattr].value_counts().get(g0, 0)
						
						if nbg1 == 0 or nbg0 == 0: # if one gorup is empty, skip as fairness metric would not make sense
							res_madd.append("")
							res_abroca.append("")
							res_groupf1.append(["", ""])
							continue
							
						#MADD
						res_madd.append(maddlib.MADD(h='auto', X_test=df_madd, pred_proba=df_madd["pred_proba"], sf=sattr, model=fset+"_"+classifier_name))

						#ABROCA
						if nbg1 > nbg0:  # select by index value (class 0 or class 1)
							majority_group = 1
						else:
							majority_group = 0
						res_abroca.append(abroca.compute_abroca(
													   df=df_madd,
													   pred_col='pred_proba',
													   label_col='final_result',
													   protected_attr_col=sattr,
													   majority_protected_attr_val=majority_group,
													   n_grid=10000,
													   plot_slices=False
													   ))
						
						#group F1
						res_groupf1.append(fair_performances.group_performances(X_test=df_madd, Y_test=df_madd["final_result"], pred=grid_predictions, sf=sattr))
					del df_madd

					res = results_classifiers(str(fset), k, classifier_name, str(best_params), train_set_labels, test_set_labels, grid_predictions_train, grid_predictions, grid_predictions2, nb_used_features, res_madd, res_abroca, res_groupf1, "fusion")
					
					fold_results.append(res)
					
					#save models as we go
					savepath = os.path.join(DIR_PICKLE, "models", fset+"_"+classifier_name+"_"+str(np.random.random())+".h5")
					classifier.best_model.save(savepath)
					grid_classifier[fset][classifier_name].append(savepath)
					
					if VERBOSE: #prints for debug:
						print(classifier_name, "res", res)
						#print(classifier.grid.best_estimator_)
						print(str(best_params))
						
			#output results and pickles
			df_results = pd.DataFrame.from_records(fold_results)
			output_name = "dfres_fusion" + str(NBFOLDS)+"folds_" + ("grid_" if GRIDSEARCH else "nogrid_") + ("smote_" if SMOTE_MINORITY else "nosmote_") + (str(FEATSELECT)+"_" if FEATSELECT else "nofeatselect_")+ str(CORREL_THRESHOLD)+"correl_" + str(RAND_STATE_SEED)+"seed"+EXPORT_SUFFIX
			df_results.to_csv(f"./{DIR_OUTPUT}/{output_name}.csv", index=False)
			df_results.to_pickle(f"./{DIR_PICKLE}/{output_name}.pkl")

			with open(f"./{DIR_PICKLE}/{output_name}_models.pkl", "wb") as file:
				pickle.dump(grid_classifier, file)

	print("Fusion model training done.")
	return pd.DataFrame.from_records(fold_results), grid_classifier


def train_models_ensemble(dfInput, ens_classifiers, saved_models, sensitive_features = [], use_sensitive_features = False):
	"""
	Train a set of classifiers on the provided dfInput data.
	dfInput: a dictionary of pandas dataframes where the keys are the name of the feature sets and the value the corresponding dataframe.
	models: a list of models initialized with the init_fusion_models() function
	returns: a dataframe with the prediction performance of the trained classifiers, whose rows are formatted with the results_classifiers() function
	"""
	levels_results_grid = {}
			
	fold_results = []
	pred_results = {}
	pred_proba = {}
	pred_labels = {}
	grid_classifier = {}

	# Iterate over the feature sets and fold to train the models
	for fset, df_features in dfInput.items():
		print(f"------------------ Feature set {fset} ------------------")
		
		pred_results[fset] = {}
		pred_proba[fset] = {}
		pred_labels[fset] = {}
		grid_classifier[fset] = {}
		cv_split_indices = create_k_folds(df_features) #create the k folds (outer loop)

		for k in range(NBFOLDS):
			print(f"------------------ K {k} ------------------")
			
			(train_set_index, test_set_index) = cv_split_indices[k]
			train_set = df_features.iloc[train_set_index]
			test_set = df_features.iloc[test_set_index]
			
			if VERBOSE:
				print ("Label distribution:")
				print("-Train", train_set[LABEL_KEY].value_counts())
				print("-Test", test_set[LABEL_KEY].value_counts())
				
				
			train_set_features, train_set_labels  = feature_label_split(train_set)
			test_set_features, test_set_labels  = feature_label_split(test_set)

			if SMOTE_MINORITY: # Oversample minority classes, in the train sets ONLY
				smoter = SMOTE(random_state = RAND_STATE_SEED)
				(train_set_features, train_set_labels) = smoter.fit_resample(train_set_features, train_set_labels)
				
			for classifier_name, classifier in ens_classifiers.items():
				if classifier_name not in pred_results[fset]: pred_results[fset][classifier_name] = [] 
				if classifier_name not in pred_proba[fset]: pred_proba[fset][classifier_name] = [] 
				if classifier_name not in pred_labels[fset]: pred_labels[fset][classifier_name] = []
				if classifier_name not in grid_classifier[fset]: grid_classifier[fset][classifier_name] = []
				
				if VERBOSE:
					print(train_set_features.info())
					print(test_set_features.info())
					print(train_set_labels.shape)
					print(test_set_labels.shape)
				

				if "Base" in classifier_name: # No grid search for the baselines
					grid_predictions_train, grid_predictions, _, grid_proba_test = classifier.fit_predict(train_set_features, train_set_labels, test_set_features)
					res = results_classifiers(fset, k, classifier_name, "", train_set_labels, test_set_labels, grid_predictions_train, grid_predictions, grid_predictions, 0, [], [], [])
					
					if VERBOSE:
						print("Baseline res", res)
					
					fold_results.append(res)
					pred_results[fset][classifier_name] += list(grid_predictions)
					pred_proba[fset][classifier_name] += list(grid_proba_test)
					pred_labels[fset][classifier_name] += list(test_set_labels)  
				else:
					# Build pipeline for Keras model:
					classifier.estimators = []
					for model_name, trained_model_path in saved_models[fset].items():
						if "Base" in model_name: continue #ignore baselines
						elif "FF" in model_name:
							load_model = KerasWrapper(tf.keras.models.load_model(trained_model_path[k]))
						else:
							with open(trained_model_path[k], "rb") as f:
								load_model = pickle.load(f)
						classifier.estimators.append( (model_name, load_model) )
					
					if use_sensitive_features:
						grid_predictions_train, grid_predictions, grid_proba_train, grid_proba_test = classifier.fit_predict(train_set_features, train_set_labels, test_set_features)
					else:
						grid_predictions_train, grid_predictions, grid_proba_train, grid_proba_test = classifier.fit_predict(train_set_features.loc[:, ~train_set_features.columns.isin(sensitive_features)], train_set_labels, test_set_features.loc[:, ~test_set_features.columns.isin(sensitive_features)])
					
					nb_used_features = len(classifier.estimators)
					best_params = ""
					grid_predictions2 = classifier.classification_threshold_optimizer(grid_proba_train, train_set_labels, grid_proba_test)
						
					#fairness eval
					res_madd = []
					res_abroca = []
					res_groupf1 = []
					df_madd = test_set_features.copy(deep=True).reset_index() #need to reset index for the MADD to work
					df_madd["pred_proba"] = grid_proba_test[:, 0] 	#for abroca
					df_madd["final_result"] = test_set_labels.to_numpy() #for abroca

					g1 = 1
					g0 = 0
					for sattr in sensitive_features:
						nbg1 = df_madd[sattr].value_counts().get(g1, 0)
						nbg0 = df_madd[sattr].value_counts().get(g0, 0)
						
						if nbg1 == 0 or nbg0 == 0: # if one gorup is empty, skip as fairness metric would not make sense
							res_madd.append("")
							res_abroca.append("")
							res_groupf1.append(["", ""])
							continue
							
						#MADD
						res_madd.append(maddlib.MADD(h='auto', X_test=df_madd, pred_proba=df_madd["pred_proba"], sf=sattr, model=fset+"_"+classifier_name))

						#ABROCA
						if nbg1 > nbg0:  # select by index value (class 0 or class 1)
							majority_group = 1
						else:
							majority_group = 0
						res_abroca.append(abroca.compute_abroca(
													   df=df_madd,
													   pred_col='pred_proba',
													   label_col='final_result',
													   protected_attr_col=sattr,
													   majority_protected_attr_val=majority_group,
													   n_grid=10000,
													   plot_slices=False
													   ))
						
						#group F1
						res_groupf1.append(fair_performances.group_performances(X_test=df_madd, Y_test=df_madd["final_result"], pred=grid_predictions, sf=sattr))
					del df_madd

					res = results_classifiers(str(fset), k, classifier_name, str(best_params), train_set_labels, test_set_labels, grid_predictions_train, grid_predictions, grid_predictions2, nb_used_features, res_madd, res_abroca, res_groupf1, "ensemble")
					
					fold_results.append(res)
					
					#save models as we go
					savepath = os.path.join(DIR_PICKLE, "models", fset+"_"+classifier_name+"_"+str(np.random.random())+".skpkl")
					with open(savepath, "wb") as file:
						pickle.dump(classifier.final_estimator, file)
					grid_classifier[fset][classifier_name].append(savepath)
					
					if VERBOSE: #prints for debug:
						print(classifier_name, "res", res)
						#print(classifier.grid.best_estimator_)
						print(str(best_params))
						
			#output results and pickles
			df_results = pd.DataFrame.from_records(fold_results)
			output_name = "dfres_ens" + str(NBFOLDS)+"folds_" + ("grid_" if GRIDSEARCH else "nogrid_") + ("smote_" if SMOTE_MINORITY else "nosmote_") + (str(FEATSELECT)+"_" if FEATSELECT else "nofeatselect_")+ str(CORREL_THRESHOLD)+"correl_" + str(RAND_STATE_SEED)+"seed"+EXPORT_SUFFIX
			df_results.to_csv(f"./{DIR_OUTPUT}/{output_name}.csv", index=False)
			df_results.to_pickle(f"./{DIR_PICKLE}/{output_name}.pkl")

			with open(f"./{DIR_PICKLE}/{output_name}_models.pkl", "wb") as file:
				pickle.dump(grid_classifier, file)

	print("Ensemble model training done.")
	return pd.DataFrame.from_records(fold_results), grid_classifier	
	
	
#############################################################################################################################
#Main program
#############################################################################################################################

#open the data
features_sets = list_feature_sets(MODALITIES, TARGETS, WINDOWS, AOIS)
fusion_features_sets = list_fusion_feature_sets(MODALITIES, TARGETS, WINDOWS, AOIS)
df_SA = pd.read_csv("./df_student_info.csv")

# open all features sets as pandas' dataframes
grouped_features = {}
for fset in features_sets:
	tmp_set = pd.read_csv(os.path.join(DIR_FEATURES, fset))
	tmp_set = clean_up_features(tmp_set)
	tmp_set = merge_features_demo(tmp_set, df_SA)
	grouped_features[fset[:-4]] = deepcopy(tmp_set)

fusion_grouped_features = {}
for fset in fusion_features_sets:
	tmp_set = pd.read_csv(os.path.join(DIR_FEATURES, fset))
	tmp_set = clean_up_features(tmp_set)
	tmp_set = merge_features_demo(tmp_set, df_SA)
	fusion_grouped_features[fset[:-4]] = deepcopy(tmp_set)

if VERBOSE:
	for fset, df_features in grouped_features.items():
		print(f"------------------ Feature set {fset} ------------------")
		print(df_features[LABEL_KEY].value_counts())
		print(df_features.info())
		print(df_features.describe())
		print(df_features.head())
		print(df_features[LABEL_KEY].value_counts())

#train classifiers
classifiers = init_classifiers()
models = init_fusion_models()
ensembles = init_ensemble_models()
if "pre" in FUSION_STYLE:
	df_results, trained_classifiers_pre = train_models_single_dataset(grouped_features, classifiers, SENSITIVE_ATTRIBUTES)
if "in" in FUSION_STYLE:
	df_results, trained_classifiers_in = train_models_fusion(fusion_grouped_features, models, SENSITIVE_ATTRIBUTES)
if "post" in FUSION_STYLE:
	if VERBOSE:
		print(trained_classifiers_pre)
	df_results, trained_classifiers = train_models_ensemble(fusion_grouped_features, ensembles, trained_classifiers_pre, SENSITIVE_ATTRIBUTES)

if VERBOSE:
	print(f"------------------ Results ------------------")
	print(df_results.info())
	print(df_results.describe())
	print(df_results.head())

#output results and pickles
"""
output_name = "dfres_" + str(NBFOLDS)+"folds_" + ("grid_" if GRIDSEARCH else "nogrid_") + ("smote_" if SMOTE_MINORITY else "nosmote_") + (str(FEATSELECT)+"_" if FEATSELECT else "nofeatselect_")+ str(CORREL_THRESHOLD)+"correl_" + str(RAND_STATE_SEED)+"seed"+EXPORT_SUFFIX
df_results.to_csv(f"./{DIR_OUTPUT}/{output_name}.csv", index=False)
df_results.to_pickle(f"./{DIR_PICKLE}/{output_name}.pkl")

with open(f"./{DIR_PICKLE}/{output_name}_models.pkl", "wb") as file:
    pickle.dump(trained_classifiers, file)
"""
print("Done. "+str(EXPORT_SUFFIX))