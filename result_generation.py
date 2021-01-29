import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict

# fuction to check the performance of the regression model using kfold cross validation on explained variance
# also checking the score with the training and test dataset
def predictions(classification_model, X_train, y_train, X_test, y_test):
    classification_model.fit(X_train, y_train)
    # here we are taking the k fold parameter as 10. It will divide the whole dataset into 10 equal parts and check performance taking each part one time as test data and other parts as training data
    y_pred = cross_val_predict(estimator=classification_model, X = X_train, y = y_train, cv = 10)
    y_pred2 = classification_model.predict(X_test)
    report_lr = precision_recall_fscore_support(y_train, y_pred, average='binary')
    report_lr2 = precision_recall_fscore_support(y_test, y_pred2, average='binary')
    print("training metrics")
    print ("\nprecision = %0.4f, recall = %0.4f, F1 = %0.4f, accuracy = %0.4f\n" % \
           (report_lr[0], report_lr[1], report_lr[2], accuracy_score(y_train, y_pred)))
    print("testing metrics")
    print ("\nprecision = %0.4f, recall = %0.4f, F1 = %0.4f, accuracy = %0.4f\n" % \
           (report_lr2[0], report_lr2[1], report_lr2[2], accuracy_score(y_test, y_pred2)))

if __name__ == "__main__":
	df=pd.read_csv("training_set.csv")
	df_majority = df[df.Y==0]
	df_minority = df[df.Y==1]
	 
	# Upsample minority class
	df_minority_upsampled = resample(df_minority, 
	                                 replace=True,     # sample with replacement
	                                 n_samples=2376,    # to match majority class
	                                 random_state=123) # reproducible results
	 
	# Combine majority class with upsampled minority class
	df_upsampled = pd.concat([df_majority, df_minority_upsampled])

	Y=df_upsampled["Y"]
	X=df_upsampled[df_upsampled.columns[1:58]]

	X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

	#Scalling of data
	sc = StandardScaler()
	X_train = sc.fit_transform(X_train)
	X_test = sc.transform (X_test)

	estimator=RandomForestClassifier(max_depth=None,max_features='log2',min_samples_leaf=1,min_samples_split=2,n_estimators=700,random_state=1)

	predictions(estimator,X_train,y_train,X_test,y_test)
