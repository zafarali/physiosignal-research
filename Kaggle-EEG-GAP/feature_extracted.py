# contains important preprocessing tools to read in our data
from tools import *
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression



# will hold all ids and all predictions
all_ids = []
all_predictions = []
subsample = 100
num_subjects = 13
num_series = 9
human_labels = ['HandStart', 'FirstDigitTouch', 'BothStartLoadPhase', 'LiftOff', 'Replace', 'BothReleased']


lr = LogisticRegression()


def nextpow2(i):
    """ 
    Find the next power of 2 for number i
    
    """
    n = 1
    while n < i: 
        n *= 2
    return n


def feature_extract(data):
  """
    Returns data at lower time resolution but with certain features extracted
  """
  n_samples, n_channels = data.shape
  rescaled_t = range(200, n_samples, 200) #do these calculations ever 200 data points

  results = np.zeros( ( len(rescaled_t)-1, n_channels*4 ) )
  
  # print results.shape

  f_sampling = 500

  for idx, t in enumerate(rescaled_t):
    data_t = data[:t, :]

    data_t = data_t - np.mean(data_t, axis=0)

    # print 'data so far is:', data_t.shape
    sample_length = data_t.shape[0]

    window = np.hamming( sample_length )

    # obtain the windowed data
    data_t = (data_t.T*window).T

    NFFT = nextpow2(sample_length)

    Y = np.fft.fft(data_t, n=NFFT, axis=0) / sample_length
    PSD = 2*np.abs( Y[0:NFFT/2,:] )
    f = f_sampling / 2*np.linspace(0,1, NFFT/2)

    # print PSD.shape

    # delta bands
    idx_delta = np.where(f<4)
    mean_delta = np.mean(np.mean(PSD[idx_delta, :], axis=0), axis=0)
    # print mean_delta
    # print mean_delta.shape
    # theta bands
    idx_theta = np.where( (f>=4) & (f<=8) )
    mean_theta = np.mean(np.mean(PSD[idx_theta, :], axis=0), axis=0)
    # print mean_theta.shape

    # alpha bands
    idx_alpha = np.where( (f>=8) & (f<=12) )
    mean_alpha = np.mean(np.mean(PSD[idx_alpha, :], axis=0), axis=0)
    # print mean_alpha.shape

    # beta bands
    idx_beta = np.where( (f>=12) & (f<30) )
    mean_beta = np.mean(np.mean(PSD[idx_beta, :], axis=0), axis=0)
    # print mean_beta.shape

    feature_vector = np.concatenate( ( mean_beta, mean_alpha, mean_theta, mean_delta ) , axis=0 )

    feature_vector = np.log10(feature_vector)
    # print feature_vector.shape
    # print feature_vector
    # print feature_vector
    results[idx] = feature_vector


  return results


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

  X_train = feature_extract(X_train)

  # train the classifier for each label
  for i in range(6):
    print 'Training subject_id = ',subject_id, ' label: ',human_labels[i]
    y_train = y[:,i]
    lr.fit(X_train[::subsample,:], y_train[::subsample])

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


  current_prediction = np.empty((X_test.shape[0], 6)) # number of test samples X number of labels
  X_test = data_preprocess_test(X_test)

  X_test = feature_extract(X_test)

  for i in range(6):
  	print 'testing subject_id=',subject_id
  	current_prediction[:,i] = lr.predict_proba(X_test)[:,1]

  	# print 'predicted:',current_prediction[:,i]

  all_predictions.append(current_prediction)

print 'testing complete'


print 'ids ',np.concatenate(all_ids).shape
print 'predictions ',np.concatenate(all_predictions).shape


submission_file = 'submission4.csv'

submission = pd.DataFrame(index=np.concatenate(all_ids), \
	columns = human_labels, \
	data = np.concatenate(all_predictions))

submission.to_csv(submission_file, index_label='id', float_format='%.3f')

print 'SAVED COMPLETE'