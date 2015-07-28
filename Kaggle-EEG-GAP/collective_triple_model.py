# contains important preprocessing tools to read in our data
from tools import *
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.lda import LDA
from sklearn.qda import QDA


# will hold all ids and all predictions
all_ids = []
all_predictions_lda = []
all_predictions_qda = []
all_predictions_lr = []
all_predictions_avg = []
subsample = 100
num_subjects = 13
num_series = 9
human_labels = ['HandStart', 'FirstDigitTouch', 'BothStartLoadPhase', 'LiftOff', 'Replace', 'BothReleased']


lr = LogisticRegression()
qda = QDA()
lda = LDA()

for subject_id in range(1,num_subjects):
  y_raw = []
  raw = []

  # read in training data
  for series_id in range(1,num_series):
    data,labels = prepare_data_train(subject_id, series_id)
    raw.append(data)
    y_raw.append(labels)

  # concatanenate the data sets into one dataframe
  X = pd.concat(raw)
  y = pd.concat(y_raw)

  X_train = np.asarray(X.astype(float))
  y = np.asarray(y.astype(float))

  X_train = data_preprocess_train(X_train)
  # train the classifier for each label
  for i in range(6):
    print 'Training subject_id = ',subject_id, ' label: ',human_labels[i]
    y_train = y[:,i]
    lr.fit(X_train[::subsample,:], y_train[::subsample])
    qda.fit(X_train[::subsample,:], y_train[::subsample])
    lda.fit(X_train[::subsample,:], y_train[::subsample])

print 'Training Complete'





print 'Testing'

for subject_id in range(1,num_subjects):

  test = [] # testing data to be stored here
  idx = [] # test data ids
  
  for series_id in range(9,11):
  	data = prepare_data_test(subject_id, series_id)
  	test.append(data)
  	print 'size of current dataset: subject=',subject_id,', series=',series_id,' is ',data.shape
  	idx.append(np.array(data['id']))

  X_test = pd.concat(test)
  all_ids.append(np.concatenate(idx))
  X_test = X_test.drop(['id'], axis=1)
  X_test = np.asarray(X_test.astype(float))


  current_prediction_lda = np.empty((X_test.shape[0], 6)) # number of test samples X number of labels
  current_prediction_lr = np.empty((X_test.shape[0], 6)) # number of test samples X number of labels
  current_prediction_qda = np.empty((X_test.shape[0], 6)) # number of test samples X number of labels
  X_test = data_preprocess_test(X_test)

  for i in range(6):
    print 'testing subject_id=',subject_id
    current_prediction_lr[:,i] = lr.predict_proba(X_test)[:,1]
    current_prediction_qda[:,i] = qda.predict_proba(X_test)[:,1]
    current_prediction_lda[:,i] = lda.predict_proba(X_test)[:,1]

  	# print 'predicted:',current_prediction[:,i]

  all_predictions_lda.append(current_prediction_lda)
  all_predictions_qda.append(current_prediction_qda)
  all_predictions_lr.append(current_prediction_lr)

  all_predictions_avg.append( (current_prediction_lda+current_prediction_qda+current_prediction_lr)/3 )

print 'testing complete'


print 'ids ',np.concatenate(all_ids).shape
print 'predictions ',np.concatenate(all_predictions_avg).shape


submission_file = 'qda_1.csv'

submission = pd.DataFrame(index=np.concatenate(all_ids), \
	columns = human_labels, \
	data = np.concatenate(all_predictions_qda))

submission.to_csv(submission_file, index_label='id', float_format='%.3f')


submission_file = 'lda_1.csv'

submission = pd.DataFrame(index=np.concatenate(all_ids), \
  columns = human_labels, \
  data = np.concatenate(all_predictions_lda))

submission.to_csv(submission_file, index_label='id', float_format='%.3f')



submission_file = 'avg_1.csv'

submission = pd.DataFrame(index=np.concatenate(all_ids), \
  columns = human_labels, \
  data = np.concatenate(all_predictions_avg))

submission.to_csv(submission_file, index_label='id', float_format='%.3f')


print 'SAVED COMPLETE'


