import numpy as np
from sklearn.svm import SVC
import time

from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV


def printerr(K,X): print( np.linalg.norm( K-X@X.T ) / np.linalg.norm( K ) ) # error between a kernel and feature representation

def randker(n): K = np.random.randn(n,n); return K@K.T # generate a random kernel

def nystroem(KS,r):
    """
    nystroem: compute a Nystrom approximate factorization of a partial kernel.

    :param KS: A partial kernel of size (m,n), where n is the number of points and m<=n the number of landmarks.
        If K is the kernel, it should be KS=K[:m,:]
    :param r: rank of the factorization, should satisfy m<=n
    :return: X a matrix of size (n,r), so that K ~ X@X.T
    :return: Lambda a vector of size r containing the singular values of the truncated kernel.
    :return: LambdaC a vector of size |S|-r containing the removed singular values
    """
    m = KS.shape[0]
    # small sub-kernel
    KSS = KS[:,:m]
    # compute SVD of the small kernel
    U, Lambda, VT = np.linalg.svd(KSS, full_matrices=True)
    # print( np.linalg.norm(KSS - U@np.diag(Lambda)@VT)/ np.linalg.norm(KSS)) # shoud be zero
    epsi = 1e-8  # be careful when we divide by Lambda near 0
    return KS.T @ U[:,:r] @ np.diag(1./np.sqrt(epsi + Lambda[:r])), Lambda[:r], Lambda[r:]

# compute the "real" kronecker kernel
def KronKernel(KP,KM,IP,IM):
  n = IP.shape[0]
  Kkron = np.zeros((n,n))
  for i in range(n):
    for j in range(n):
      Kkron[i,j] = KP[IP[i],IP[j]] * KM[IM[i],IM[j]]
  return Kkron

def KronFeat(X,Y,LambdaX,LambdaY,r):
  """
  KronFeat - compute approximate feature of a Kronecker kernel.

  :param X: The first set of features, of size (n,rX), approximating a kernel KX ~ X@X.T
  :param Y: The second set of features, of size (n,rY), approximating a kernel KY ~ Y@Y.T
  :LambdaX: standard deviation ("importance") of the feature of X, vector of length rX
  :LambdaY: standard deviation ("importance") of the feature of Y, vector of length rY
  :r: size of the computed feature, should satisfy r<=rX*rY
  :return: Xkron of size (n,r), so that kron(KX,KY) ~ Xkron @ Xkron.T
  """
  LambdaXY = LambdaX[:,None] * LambdaY[None,:]
  IndFeat = np.argsort(-LambdaXY.flatten())[:r]
  n = X.shape[0]
  Xkron = np.zeros((n,r))
  for i in range(n):
    x = X[i,:]
    y = Y[i,:]
    xy = x[:,None] * y[None,:]
    Xkron[i,:] = xy.flatten()[IndFeat]
  return Xkron








def pred_SVM_Kernel(KP,KM,all_train_interactions_arr, all_test_interactions_arr,C=10):
  """
  pred_SVM_Kernel - predict using a SVM with a Kronecker kernel.

  :param KP: The first kernel, of size (nP,nP)
  :param KM: The second kernel, of size (nM,nM)
  :param all_train_interactions_arr: list of tuples (IP_train, IM_train, y_train) where IP_train, IM_train are the indices of the interactions in the training set, and y_train is the label of the interaction
  :param all_test_interactions_arr: list of tuples (IP_test, IM_test, y_test) where IP_test, IM_test are the indices of the interactions in the test set, and y_test is the label of the interaction
  :param C: SVM regularization parameter
  :return: all_p_pred_arr: list of arrays of size (n_test,2) containing the predicted probability of each interaction to be positive
  :return: t: time taken to compute the prediction
  """


  t1 = time.time()

  n_fold = len(all_train_interactions_arr)

  all_p_pred_arr = []

  for i in range(n_fold):

    IP_train, IM_train, y_train = all_train_interactions_arr[i][:,0],all_train_interactions_arr[i][:,1],all_train_interactions_arr[i][:,2]
    IP_test, IM_test = all_test_interactions_arr[i][:,0],all_test_interactions_arr[i][:,1]

    K_train = KronKernel(KP,KM,IP_train,IM_train)

    clf = SVC(kernel='precomputed', C = C, class_weight='balanced',probability=True)
    clf.fit(K_train, y_train)

    n = IP_test.shape[0]
    m = IP_train.shape[0]
    K_test_train = np.zeros((n,m))
    for i in range(n):
      for j in range(m):
        K_test_train[i,j] = KP[IP_test[i],IP_train[j]] * KM[IM_test[i],IM_train[j]]
    p_pred = clf.predict_proba(K_test_train)
    all_p_pred_arr.append(p_pred[:,1])

  t2 = time.time()
  return all_p_pred_arr,(t2-t1)/n_fold



def pred_SVM_features(XP,XM,LambdaP,LambdaM,r,all_train_interactions_arr, all_test_interactions_arr,C=10):
  """
  pred_SVM_features - predict using a linear SVM on some features of Kronecker kernel.

  :param XP : The first set of features, of size (nP,rX), approximating a kernel KX ~
  :param XM : The second set of features, of size (nM,rY), approximating a kernel KY ~
  :param r: size of the computed feature, should satisfy r<=rX*rY

  :param all_train_interactions_arr: list of tuples (IP_train, IM_train, y_train) where IP_train, IM_train are the indices of the interactions in the training set, and y_train is the label of the interaction
  :param all_test_interactions_arr: list of tuples (IP_test, IM_test, y_test) where IP_test, IM_test are the indices of the interactions in the test set, and y_test is the label of the interaction
  :param C: SVM regularization parameter
  :return: all_p_pred_arr: list of arrays of size (n_test,2) containing the predicted probability of each interaction to be positive
  :return: t: time taken to compute the prediction
  """

  t1 = time.time()

  n_fold = len(all_train_interactions_arr)

  all_p_pred_arr = []

  for i in range(n_fold):

    IP_train, IM_train, y_train = all_train_interactions_arr[i][:,0],all_train_interactions_arr[i][:,1],all_train_interactions_arr[i][:,2]
    IP_test, IM_test = all_test_interactions_arr[i][:,0],all_test_interactions_arr[i][:,1]

    # compute the features of the Kronecker kernel
    Xkron_train = KronFeat(XP[IP_train,:],XM[IM_train,:],LambdaP,LambdaM,r)

    clf = SVC(kernel='linear', C = C, class_weight='balanced',probability=True,random_state=0)
    clf.fit(Xkron_train, y_train)

    # compute the features of the test set
    Xkron_test = KronFeat(XP[IP_test,:],XM[IM_test,:],LambdaP,LambdaM,r)

    p_pred = clf.predict_proba(Xkron_test)
    all_p_pred_arr.append(p_pred[:,1])

  t2 = time.time()
  return all_p_pred_arr,(t2-t1)/n_fold

def pred_SVM_features_LinearSVC(XP,XM,LambdaP,LambdaM,r,all_train_interactions_arr, all_test_interactions_arr,C=10):
  """
   pred_SVM_features - predict using a LinearSVM on some features of Kronecker kernel.

  :param XP : The first set of features, of size (nP,rX), approximating a kernel KX ~
  :param XM : The second set of features, of size (nM,rY), approximating a kernel KY ~
  :param r: size of the computed feature, should satisfy r<=rX*rY

  :param all_train_interactions_arr: list of tuples (IP_train, IM_train, y_train) where IP_train, IM_train are the indices of the interactions in the training set, and y_train is the label of the interaction
  :param all_test_interactions_arr: list of tuples (IP_test, IM_test, y_test) where IP_test, IM_test are the indices of the interactions in the test set, and y_test is the label of the interaction
  :param C: SVM regularization parameter
  :return: all_p_pred_arr: list of arrays of size (n_test,2) containing the predicted probability of each interaction to be positive
  :return: t: time taken to compute the prediction
  """

  t1 = time.time()

  n_fold = len(all_train_interactions_arr)

  all_p_pred_arr = []

  for i in range(n_fold):

    IP_train, IM_train, y_train = all_train_interactions_arr[i][:,0],all_train_interactions_arr[i][:,1],all_train_interactions_arr[i][:,2]
    IP_test, IM_test = all_test_interactions_arr[i][:,0],all_test_interactions_arr[i][:,1]

    # compute the features of the Kronecker kernel
    Xkron_train = KronFeat(XP[IP_train,:],XM[IM_train,:],LambdaP,LambdaM,r)

    svm = LinearSVC(loss = "hinge", C = C, class_weight='balanced',max_iter = 2000)
    calibrated_clf = CalibratedClassifierCV(svm)
    calibrated_clf.fit(Xkron_train, y_train)

    # compute the features of the test set
    Xkron_test = KronFeat(XP[IP_test,:],XM[IM_test,:],LambdaP,LambdaM,r)

    p_pred = calibrated_clf.predict_proba(Xkron_test)
    all_p_pred_arr.append(p_pred[:,1])

  t2 = time.time()
  return all_p_pred_arr,(t2-t1)/n_fold


#svm = LinearSVC(loss = "hinge", C = C, class_weight='balanced')
#clf = CalibratedClassifierCV(svm)
#clf.fit(Xkron_train, y_train)
#y_proba = clf.predict_proba(X_test)

# Nystrom method on the protein kernel

def features_norm(KP,KM):
  # features for protein kernel
  nP = KP.shape[0] # full kernel (non centered and normalized)
  XP,LambdaP,LambdaDP = nystroem(KP,nP)

  # features for molecule kernel
  nM = KM.shape[0] # full kernel (non centered and normalized)
  # partially observed kernel
  XM,LambdaM,LambdaDM = nystroem(KM,nM)

  # Scale the features for better performance
  XP_c = XP - XP.mean(axis = 0)
  XM_c = XM - XM.mean(axis = 0)

  XP_cn = XP_c / np.linalg.norm(XP_c,axis = 1)[:,None]
  XM_cn = XM_c / np.linalg.norm(XM_c,axis = 1)[:,None]

  return XP_cn,XM_cn,LambdaP,LambdaM
