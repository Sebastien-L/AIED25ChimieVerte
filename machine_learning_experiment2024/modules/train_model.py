"""
AI fairness analysis code
@author Sebastien Lalle
@email sebastien.lalle at lip6.fr
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
RAND_STATE_SEED = 2204 #for reproducibility
EXPORT_SUFFIX = "_runtest1" #suffix to add to all exported file (for version control)
NBFOLDS = 5 # number of cross validation folds
NBFOLDS_GRID = 5 # number of inner folds for the grid search hyperparameter tuning
GRIDSEARCH = True #True do to grid-search hyperparameter tuning
CORREL_THRESHOLD = 0.8 # thresholds for discarding correlated features (1 means no feature removal) 
FEATSELECT = "PCA" # "PCA" for PCA feature selection, "Kbest" for statistical-based feature selection, None otherwise
KBEST = 70 # Top K feature for Kbest feature selection
SMOTE_MINORITY = False # True/False: to turn on/off Smote over-sampling of the minority classes
VERBOSE = True # True/False

LABEL_KEY = "Answer" # the key of the labels in the feature dataframe (successfully passed the course)
LABEL_VALUES = [0, 1] #the values of the label (attention/memory: 0 for no, 1 for yes)
STUDENT_ID_KEY = "Part_id" # the grouping key used to do cross-validation over users (effectively using the StratifiedGroupKFold function of Sklearn)

DIR_FEATURES = "features"
SENSITIVE_ATTRIBUTES = ["Gender", "Tiredness", "PriorKnowledge"]
TARGETS = ["QCM", "SelfReports"]
WINDOWS = [10, 20, 30, "Content"]
AOIS = ["useraoi_importance", "useraoi_types", "useraoi_type_importance", "useraoi_detailed", "staticAOI4x4", "staticAOI3x3", "staticAOI2x1"]
MODALITIES = ["ET_All", "ET_AOIonly", "ET_noAOI", "ET_fixations", "ET_saccades","ET_pupils","ET_headdistance","OF_All", "OF_AU","OF_gaze","OF_pose","multimodal_All", "multimodal_AOIonly", "multimodal_noAOI"]

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


def init_classifiers():
	#Create the final list of all classifiers
	kwargs = {"nb_inner_folds": NBFOLDS_GRID, "labels": LABEL_VALUES, "verbose": VERBOSE, "seed": RAND_STATE_SEED}
	classifiers = {
		"Base1": StratifiedBaselineClassifier(**kwargs),
		"Base2": MajorityClassBaselineClassifier(**kwargs),"""
		"NB": NaiveBayesClassifier(CORREL_THRESHOLD, KBEST, FEATSELECT, GRIDSEARCH, **kwargs),
		"KNN": KNeirestNeighboorsClassifier(CORREL_THRESHOLD, KBEST, FEATSELECT, GRIDSEARCH, **kwargs),
		"LR": LogisticClassifier(CORREL_THRESHOLD, KBEST, FEATSELECT, GRIDSEARCH, **kwargs),
		"SVM": SVMClassifier(CORREL_THRESHOLD, KBEST, FEATSELECT, GRIDSEARCH, **kwargs),
		"SGB": StochasticGradientBoostingAlgorithmClassifier(CORREL_THRESHOLD, KBEST, FEATSELECT, GRIDSEARCH, **kwargs),"""
		"RF": RandomForestEnsembleClassifier(CORREL_THRESHOLD, KBEST, FEATSELECT, GRIDSEARCH, **kwargs)
		#"MLP": MultiLayerPerceptronClassifier(CORREL_THRESHOLD, KBEST, FEATSELECT, GRIDSEARCH, **kwargs),
		#"FFNN1": FeedForwardClassifier(1, CORREL_THRESHOLD, KBEST, FEATSELECT, GRIDSEARCH, **kwargs),
		#"FFNN2": FeedForwardClassifier(2, CORREL_THRESHOLD, KBEST, FEATSELECT, GRIDSEARCH, **kwargs),
		#"FFNNAtt1": FeedForwardSelfAttentionClassifier(1, CORREL_THRESHOLD, KBEST, FEATSELECT, GRIDSEARCH, **kwargs),
		#"FFNNAtt2": FeedForwardSelfAttentionClassifier(2, CORREL_THRESHOLD, KBEST, FEATSELECT, GRIDSEARCH, **kwargs)
	}
	return classifiers


def results_classifiers(fset, k, clf, hyperparam, train_set_labels, test_set_labels, predictions_train, predictions_test, predictions2_test, nb_features_in, res_madd, res_abroca, res_groupf1):
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
		"feat_set": fset,
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
		"auc_train": roc_auc_score(train_set_labels, predictions_train, average='weighted'),
		"accuracy": accuracy_score(test_set_labels, predictions_test),
		"precision": precision_score(test_set_labels, predictions_test, average='weighted'),
		"recall": recall_score(test_set_labels, predictions_test, average='weighted'),
		"f1_score": f1_score(test_set_labels, predictions_test, average='weighted'),
		"f1_score_th": f1_score(test_set_labels, predictions2_test, average='weighted'),
		"AUC_score": roc_auc_score(test_set_labels, predictions_test, average='weighted'),
		"AUC_score_th": roc_auc_score(test_set_labels, predictions2_test, average='weighted'),
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
					nb_used_features = classifier.grid.best_estimator_['classifier'].n_features_in_
				 
					if "FFNN" not in classifier_name: #threshold optimization makes sense only for sklearn models
						#opti threshold
						"""if use_sensitive_features:
							train_proba = classifier.grid.best_estimator_['classifier'].predict_proba(train_set_features)
						else:
							train_proba = classifier.grid.best_estimator_['classifier'].predict_proba(train_set_features.loc[:, ~train_set_features.columns.isin(sensitive_features)])"""
						grid_predictions2 = classifier.classification_threshold_optimizer(grid_proba_train, train_set_labels, grid_proba_test)
					else:
						grid_predictions2 = grid_predictions
						
					#fairness eval
					res_madd = []
					res_abroca = []
					res_groupf1 = []
					df_madd = test_set_features.copy(deep=True).reset_index() #need to reset index for the MADD to work
					df_madd["pred_proba"] = grid_proba_test[:, 1] #for abroca
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
						res_madd.append(maddlib.MADD(h='auto', X_test=df_madd, pred_proba=grid_proba_test[:, 1], sf=sattr, model=fset+"_"+classifier_name))

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

					"""thresholds = np.linspace(0,1, 100)[1:-1]
					scores = [th_scoring(th, train_set_labels, classifier.grid.predict_proba(train_set_features)[:,1]) for th in thresholds]
					newth = thresholds[np.argmin( np.min(scores) )]
					grid_predictions2 = [1 if x >= newth else 0 for x in grid_proba_test[:, 1]]
					"""
					res = results_classifiers(fset, k, classifier_name, str(classifier.grid.best_params_), train_set_labels, test_set_labels, grid_predictions_train, grid_predictions, grid_predictions2, nb_used_features, res_madd, res_abroca, res_groupf1)
					
					fold_results.append(res)
					
					if "FFNN" in classifier_name: #clone Keras model
						#print(dir(classifier.grid.best_estimator_['classifier'].model))
						classifier.grid.best_estimator_['classifier'].model.save(os.path.join(DIR_PICKLE, "models", "test.h5"))
						grid_classifier[fset][classifier_name].append(os.path.join(DIR_PICKLE, "models", "test.h5"))
					else:
						grid_classifier[fset][classifier_name].append(deepcopy(classifier.grid.best_estimator_['classifier']))
					

					#pred_results[fset][classifier_name] += list(grid_predictions)
					#pred_proba[fset][classifier_name] += list(grid_proba_test)
					#pred_labels[fset][classifier_name] += list(test_set_labels)
					
					if VERBOSE: #prints for debug:
						print(classifier_name, "res", res)
						print(classifier.grid.best_estimator_)
						print(classifier.grid.best_params_)
						print(classifier.grid.best_estimator_["features_correlated"].to_keep)
						print(classifier.grid.best_estimator_["features_correlated"].to_drop)		   

	print("Single model training done.")
	return pd.DataFrame.from_records(fold_results), grid_classifier



#############################################################################################################################
#Main program
#############################################################################################################################

#open the data
features_sets = list_feature_sets(MODALITIES, TARGETS, WINDOWS, AOIS)
features_sets = ["MLfeatures_multimodal_All_QCM_30_useraoi_type_importance.csv"] #TEST
df_SA = pd.read_csv("./df_student_info.csv")

# open all features sets as pandas' dataframes
grouped_features = {}
for fset in features_sets:
	tmp_set = pd.read_csv(os.path.join(DIR_FEATURES, fset))
	tmp_set = clean_up_features(tmp_set)
	tmp_set = merge_features_demo(tmp_set, df_SA)
	grouped_features[fset[:-4]] = deepcopy(tmp_set)
	

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
df_results, trained_classifiers = train_models_single_dataset(grouped_features, classifiers, SENSITIVE_ATTRIBUTES)


if VERBOSE:
	print(f"------------------ Results ------------------")
	print(df_results.info())
	print(df_results.describe())
	print(df_results.head())

#output results and pickles
output_name = "dfres_" + str(NBFOLDS)+"folds_" + ("grid_" if GRIDSEARCH else "nogrid_") + ("smote_" if SMOTE_MINORITY else "nosmote_") + (str(FEATSELECT)+"_" if FEATSELECT else "nofeatselect_")+ str(CORREL_THRESHOLD)+"correl_" + str(RAND_STATE_SEED)+"seed"+EXPORT_SUFFIX
df_results.to_csv(f"./{DIR_OUTPUT}/{output_name}.csv", index=False)
df_results.to_pickle(f"./{DIR_PICKLE}/{output_name}.pkl")

with open(f"./{DIR_PICKLE}/{output_name}_models.pkl", "wb") as file:
    pickle.dump(trained_classifiers, file)

print("Done.")