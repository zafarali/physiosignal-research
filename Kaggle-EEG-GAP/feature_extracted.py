# contains important preprocessing tools to read in our data
from tools import *
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression



# will hold all ids and all predictions
all_ids = []
all_predictions = []
subsample = 150
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


def rescale_y(y, idicies):

  results = np.zeros( ( len(rescaled_t) ) )

  for idx, t in enumerate(rescaled_t):
    y_t = y[t-1000:t]
    y_t = np.max(y_t)
    results[idx] = y_t

  return results

def feature_extract2(data):
  """
    Returns data at lower time resolution but with certain features extracted
  """
  n_samples, n_channels = data.shape
  rescaled_t = range(1000, n_samples, 1) #do these calculations everY 1 data points

  results = np.zeros( ( n_samples, n_channels*4 ) )
  
  # print results.shape

  f_sampling = 500
  
  print 'total idx:',len(rescaled_t)

  for t in rescaled_t:
    # print 'extracting FFT information for idx:',idx,' time_since_zero=',t
    data_t = data[t-1000:t, :]

    data_t = data_t - np.mean(data_t, axis=0)

    # print 'data so far is:', data_t.shape
    sample_length = data_t.shape[0]

    window = np.hamming( sample_length )

    # obtain the windowed data
    data_t = (data_t.T*window).T

    NFFT = nextpow2(sample_length)
    # print 'calculating FFT'
    Y = np.fft.fft(data_t, n=NFFT, axis=0) / sample_length
    PSD = 2*np.abs( Y[0:NFFT/2,:] )
    f = f_sampling / 2*np.linspace(0,1, NFFT/2)
    # print 'FFT calculated'
    # print PSD.shape

    # delta bands
    idx_delta = np.where(f<4)
    mean_delta = np.mean(np.mean(PSD[idx_delta, :], axis=0), axis=0)
    # print 'caclulated delta'
    # print mean_delta
    # print mean_delta.shape
    # theta bands
    idx_theta = np.where( (f>=4) & (f<=8) )
    mean_theta = np.mean(np.mean(PSD[idx_theta, :], axis=0), axis=0)

    # print 'caclulated theta'
    # print mean_theta.shape

    # alpha bands
    idx_alpha = np.where( (f>=8) & (f<=12) )
    mean_alpha = np.mean(np.mean(PSD[idx_alpha, :], axis=0), axis=0)

    # print 'caclulated alpha'
    # print mean_alpha.shape

    # beta bands
    idx_beta = np.where( (f>=12) & (f<30) )
    mean_beta = np.mean(np.mean(PSD[idx_beta, :], axis=0), axis=0)
    # print 'beta'
    # print mean_beta.shape

    feature_vector = np.concatenate( ( mean_beta, mean_alpha, mean_theta, mean_delta ) , axis=0 )

    feature_vector = np.log10(feature_vector)
    # print feature_vector.shape
    # print feature_vector
    # print feature_vector
    if t == 1000:
      results[0:t] = feature_vector

    results[t] = feature_vector


  return results, rescaled_t



def feature_extract(data, start, jump):
  """
    Returns data at lower time resolution but with certain features extracted
  """
  n_samples, n_channels = data.shape
  rescaled_t = range(start, n_samples, jump) #do these calculations everY 1 data points

  results = np.zeros( ( len(rescaled_t), n_channels*4 ) )
  
  # print results.shape

  f_sampling = 500
  
  print 'total idx:',len(rescaled_t)

  for idx, t in enumerate(rescaled_t):
    # print 'extracting FFT information for idx:',idx,' time_since_zero=',t
    data_t = data[t-jump:t, :]

    data_t = data_t - np.mean(data_t, axis=0)

    # print 'data so far is:', data_t.shape
    sample_length = data_t.shape[0]

    window = np.hamming( sample_length )

    # obtain the windowed data
    data_t = (data_t.T*window).T

    NFFT = nextpow2(sample_length)
    # print 'calculating FFT'
    Y = np.fft.fft(data_t, n=NFFT, axis=0) / sample_length
    PSD = 2*np.abs( Y[0:NFFT/2,:] )
    f = f_sampling / 2*np.linspace(0,1, NFFT/2)
    # print 'FFT calculated'
    # print PSD.shape

    # delta bands
    idx_delta = np.where(f<4)
    mean_delta = np.mean(np.mean(PSD[idx_delta, :], axis=0), axis=0)
    # print 'caclulated delta'
    # print mean_delta
    # print mean_delta.shape
    # theta bands
    idx_theta = np.where( (f>=4) & (f<=8) )
    mean_theta = np.mean(np.mean(PSD[idx_theta, :], axis=0), axis=0)

    # print 'caclulated theta'
    # print mean_theta.shape

    # alpha bands
    idx_alpha = np.where( (f>=8) & (f<=12) )
    mean_alpha = np.mean(np.mean(PSD[idx_alpha, :], axis=0), axis=0)

    # print 'caclulated alpha'
    # print mean_alpha.shape

    # beta bands
    idx_beta = np.where( (f>=12) & (f<30) )
    mean_beta = np.mean(np.mean(PSD[idx_beta, :], axis=0), axis=0)
    # print 'beta'
    # print mean_beta.shape

    feature_vector = np.concatenate( ( mean_beta, mean_alpha, mean_theta, mean_delta ) , axis=0 )

    feature_vector = np.log10(feature_vector)
    # print feature_vector.shape
    # print feature_vector
    # print feature_vector
    results[idx] = feature_vector


  return results, rescaled_t


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
  print 'extracting features'
  X_train, rescaled_t = feature_extract(X_train, 2000, 1000)
  print 'features extracted'
  # train the classifier for each label
  for i in range(6):
    print 'Training subject_id = ',subject_id, ' label: ',human_labels[i]
    y_train = rescale_y( y[:,i], rescaled_t )
    try:
      lr.fit(X_train[::subsample,:], y_train[::subsample])
    except Exception as e:
      print 'Skipped training due to '+str(e)

print 'Training Complete'


import pickle
with open('feature_extracted_trained.pickle', 'w') as f:
  pickle.dump( lr , f )
  print 'Saved model'


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

  X_test = data_preprocess_test(X_test)

  X_test, _ = feature_extract2(X_test)
  current_prediction = np.empty((X_test.shape[0], 6)) # number of test samples X number of labels

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