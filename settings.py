processors = 4
feature_selection = None #classifier_name  #None

classifiers = { 'SupportVectorMachine': 'SVM',
				'RandomForests': 'RF',
				#'XGBoost': 'XGB',
				'LinearSVM': 'LSVM',
				'ExtraTreeForests': 'XTREE'}

#classifier_name = classifiers['SupportVectorMachine'] # Classifier
classifier_name = classifiers['RandomForests'] # Feature Ranking

estimators = { 'SupportVectorRegression': 'SVR',
			   'LinearSupportVectorRegression': 'LSVR',
			   'LogisticRegression': 'LR',
			   #'IsotonicRegression': 'IR'
			   }

estimator_name = estimators['LogisticRegression']
