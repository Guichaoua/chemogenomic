import numpy as np
import pickle
import pandas as pd

from sklearn.preprocessing import KernelCenterer
import sklearn.model_selection as model_selection
from sklearn import metrics

from sklearn.model_selection import GroupKFold

from rdkit import Chem, DataStructs, RDConfig
from rdkit.Chem import AllChem

def load_data():
    dir = "/Users/gguichaoua/Dropbox/gwenn/these/Cluster/data/drugbank_v5.1.5/Sh/Sh_base/preprocessed/"
    # make a dataframe of the interactions
    Sh_base_list_interactions = pickle.load(open(dir+'Sh_base_list_interactions.data', 'rb'))
    df = pd.DataFrame(Sh_base_list_interactions, columns=['uniprot', 'DBid'])
    # add to df the smiles and th indice of the ligand
    Sh_base_dict_ind2mol = pickle.load(open(dir+'Sh_base_dict_ind2mol.data', 'rb'))
    Sh_base_dict_mol2ind = pickle.load(open(dir+'Sh_base_dict_mol2ind.data', 'rb'))
    Sh_base_dict_DBid2smiles = pickle.load(open(dir+'Sh_base_dict_DBid2smiles.data', 'rb'))
    df['smiles'] = df['DBid'].map(Sh_base_dict_DBid2smiles)
    df['ind2mol'] = df['DBid'].map(Sh_base_dict_mol2ind)
    # add to df the fasta and indice of the protein
    Sh_base_dict_ind2prot = pickle.load(open(dir+'Sh_base_dict_ind2prot.data', 'rb'))
    Sh_base_dict_prot2ind = pickle.load(open(dir+'Sh_base_dict_prot2ind.data', 'rb'))
    Sh_base_dict_uniprot2fasta = pickle.load(open(dir+'Sh_base_dict_uniprot2fasta.data', 'rb'))
    # add to dbid the smiles of the ligand
    df['fasta'] = df['uniprot'].map(Sh_base_dict_uniprot2fasta)
    df['ind2prot'] = df['uniprot'].map(Sh_base_dict_prot2ind)

    df['inter'] = 1
    # Return data
    return df

def load_small_data_set(n,choice):
    # load data
    df = pd.read_csv('data/Consensus_CompoundBioactivity_Dataset_v1.1_Sh_all.csv',low_memory=False)
    if choice == "random_ligns":
        # n number of ligns
        df_small = df[df.indsmiles < 15000]
        df_small = df_small.sample(n,replace=False)
        print("ok")
    elif choice == "random_mol":
        # number of the first n molecules
        df_small = df[df.indsmiles < n]

    df_small_p = df_small[df_small['interaction+'] == True]

    # give indice to all smiles we keep and all fasta we keep (ie with interactions +)

    # make dict smiles2ind and dict ind2smiles
    df_sm = df_small_p[["smiles"]].drop_duplicates().reset_index()
    #df_sm = df_small_p[["standardized smiles"]].drop_duplicates().reset_index()
    df_sm.drop(columns=["index"],inplace=True)
    dict_ind2smiles = df_sm.to_dict()["smiles"]
    #dict_ind2smiles = df_sm.to_dict()["standardized smiles"]
    print("nombre de smiles: ",len(dict_ind2smiles))
    dict_smiles2ind = {v: k for k, v in dict_ind2smiles.items()}

    df_prot = df_small_p[["fasta"]].drop_duplicates().reset_index()
    df_prot.drop(columns=["index"],inplace=True)
    dict_ind2fasta = df_prot.to_dict()["fasta"]
    print("nombre de fasta: ",len(dict_ind2fasta))
    dict_fasta2ind = {v: k for k, v in dict_ind2fasta.items()}

    # add this number to df_small
    df_small["old_indsmiles"] = df_small["indsmiles"].astype(int)
    df_small["indsmiles"] = df_small["smiles"].map(dict_smiles2ind)

    df_small["old_indfasta"] = df_small["indfasta"].astype(int)
    df_small["indfasta"] = df_small["fasta"].map(dict_fasta2ind)

    # we drop when indsmiles is Nan
    indsmiles_index_with_nan = df_small.index[df_small.loc[:,"indsmiles"].isnull()]
    df_small = df_small.drop(indsmiles_index_with_nan,0)
    df_small["indsmiles"] = df_small["indsmiles"].astype(int)
    # we drop when indfasta is Nan
    indfasta_index_with_nan = df_small.index[df_small.loc[:,"indfasta"].isnull()]
    df_small = df_small.drop(indfasta_index_with_nan,0)
    df_small["indfasta"] = df_small["indfasta"].astype(int)


    # make matrix of interactions with the score
    #intMat = df_small.pivot(index='indfasta', columns="indsmiles", values='score').to_numpy(dtype=np.float16)

    return df_small, dict_ind2smiles, dict_ind2fasta





def center_and_normalise_kernel(K_temp):

    #center phi
    K_temp = KernelCenterer().fit_transform(K_temp)
    #n = K_temp.shape[0]
    #J = np.eye(n) - np.ones(n)/n
    #K_temp = J@K@J

    # normalise phi
    D = np.diag(K_temp)
    K_norm = K_temp*1/np.sqrt(D[:,None]*D[None,:])

    return K_norm


def K_mol_norm_calcul(dict_ind2smiles):
    nb_mol = len(dict_ind2smiles)
    # make a list of smiles in order of indice
    ms = [Chem.MolFromSmiles(dict_ind2smiles[i]) for i in dict_ind2smiles]
    # make a list of fingerprints
    list_fingerprint = [AllChem.GetMorganFingerprint(m, 2) for m in ms]

    KM = np.zeros((nb_mol, nb_mol),dtype = np.float64)

    for i in range(nb_mol):
        KM[i,:] = DataStructs.BulkTanimotoSimilarity(list_fingerprint[i], list_fingerprint[:])

    print(KM.shape)

    KM_norm = center_and_normalise_kernel(KM)
    return KM,KM_norm

def K_mol_norm(df_small):
    # noyau sur les mol
    with open('data/all_base_K_mol_MorganFP_part_0.data','rb') as f:
        K_mol = pickle.load(f)

    # reduire K_mol avec  indices des ligands de df_small
    IM = df_small.old_indsmiles.unique()
    K_mol = K_mol[IM,:]
    KM = K_mol[:,IM]

    print(KM.shape)

    KM_norm = center_and_normalise_kernel(KM)
    return KM,KM_norm


def K_prot_norm(df_small):
    # noyau sur les protéines
    with open('data/all_base_K_prot.data', 'rb') as f:
        K_prot = pickle.load(f)

    # reduire K_prot avec  indices des ligands de df_small
    IP = df_small.old_indfasta.unique()
    K_prot = K_prot[IP,:]
    KP = K_prot[:,IP]

    print(KP.shape)

    KP_norm = center_and_normalise_kernel(KP)
    return KP,KP_norm


def make_train_test(df,nb_folds,p):
  """
    make train and test sets
    p is the proportion of the total positive interactions
  """

  # algo Matthieu corrected
  intMat = df.pivot(index='indfasta', columns="indsmiles", values='score').to_numpy(dtype=np.float16)

  # Set the different folds
  skf_positive = model_selection.KFold(shuffle=True, n_splits=nb_folds)

  all_train_interactions_arr = []
  all_test_interactions_arr = []

  n_p,n_m = intMat.shape
  Ip, Jm = np.where(intMat==1)
  nb_positive_inter = int(len(Ip))
  Inp, Jnm = np.where(intMat==0)
  Inkp, Jnkm = np.where(np.isnan(intMat))

  for train_index, test_index in skf_positive.split(range(nb_positive_inter)):
      # 9' pour train
      train_index = np.random.choice(train_index, int(p*len(train_index)), replace=False)

      Mm, bin_edges = np.histogram(Ip[train_index], bins = range(n_p+1)) # np.array with  #interactions for each protein of the train at the beginning

      Mp, bin_edges = np.histogram(Jm[train_index], bins = range(n_m+1)) # np.array with  #interactions for each drugs at the beginning (how manu time it can be chosen)

      train = np.zeros([1,3], dtype=int)

      nb_prot = len(list(set(Ip[train_index]))) # number of different prot in train
      for i in range(nb_prot):

          j = np.argmax(Mm) # choose protein with the maximum of interactions in the train

          indice_P = Jm[train_index][np.where(Ip[train_index]==j)[0]]  #np.array with index of interactions + in train
          indice_N = [k for k in Jm[train_index] if intMat[j][k]==0]
          indice_NK = [k for k in Jm[train_index] if np.isnan(intMat[j][k])] #np.array  with index of interactions not known

          indice_freq_mol = np.where(Mp>1)[0]  #drug's index with more than 2 interactions +
          indice_poss_mol = np.where(Mp == 1)[0]  #drug's index with 1 interaction +

          indice_freq_one_prot = np.intersect1d(indice_N, indice_freq_mol)
          indice_poss_one_prot = np.intersect1d(indice_N, indice_poss_mol)

          nb_positive_interactions = len(indice_P)
          nb_frequent_hitters_negative_interactions = len(indice_freq_one_prot)

          indice_freq_one_prot = np.intersect1d(indice_N, indice_freq_mol)
          indice_poss_one_prot = np.intersect1d(indice_N, indice_poss_mol)
          indice_freq_one_prot_NK = np.intersect1d(indice_NK, indice_freq_mol)
          indice_poss_one_prot_NK = np.intersect1d(indice_NK, indice_poss_mol)

          if len(indice_P) <= len(indice_freq_one_prot):
              # we shoot at random nb_positive_interactions in drugs with a lot of interactions
              indice_N_one_prot = np.random.choice(indice_freq_one_prot,
                                                  len(indice_P), replace = False)
          elif len(indice_P) <= len(indice_freq_one_prot) + len(indice_poss_one_prot):
              # we shoot at random nb_positive_interactions in drugs with a lot of interactions
              nb_negative_interactions_remaining = len(indice_P) - len(indice_freq_one_prot)
              indice_N_one_prot_poss = np.random.choice(indice_poss_one_prot,
                                                      nb_negative_interactions_remaining, replace = False )
              indice_N_one_prot = np.concatenate((indice_freq_one_prot,
                                              indice_N_one_prot_poss))
          elif len(indice_P) <= len(indice_freq_one_prot) + len(indice_poss_one_prot) + len(indice_freq_one_prot_NK):
              # we shoot at random nb_positive_interactions in drugs with a lot of interactions
              nb_negative_interactions_remaining = len(indice_P) - len(indice_freq_one_prot) - len(indice_poss_one_prot)
              indice_N_one_prot_poss = np.random.choice(indice_freq_one_prot_NK,
                                                      nb_negative_interactions_remaining, replace = False )
              indice_N_one_prot = np.concatenate((indice_freq_one_prot,
                                              indice_poss_one_prot, indice_N_one_prot_poss))
          else:
              # we shoot at random nb_positive_interactions in drugs with a lot of interactions
              nb_negative_interactions_remaining = len(indice_P) - len(indice_freq_one_prot) - len(indice_poss_one_prot) - len(indice_freq_one_prot_NK)
              #print("nb_negative_interactions_remaining", nb_negative_interactions_remaining) # pas de solution...
              #print(indice_poss_one_prot_NK.shape)
              indice_N_one_prot_poss = np.random.choice(indice_poss_one_prot_NK,
                                                      nb_negative_interactions_remaining, replace = False )
              indice_N_one_prot = np.concatenate((indice_freq_one_prot,
                                              indice_poss_one_prot, indice_freq_one_prot_NK, indice_N_one_prot_poss))

          Mp[indice_N_one_prot.astype(int)]-=1

          # this protein has been processed
          Mm[j] = 0

          indice = np.r_[indice_P,indice_N_one_prot].astype(int)
          etiquette = [x if not np.isnan(x) else 0 for x in intMat[j][indice]]
          A = np.stack((indice, etiquette), axis=-1)
          B = np.c_[np.zeros(A.shape[0])+j,A].astype(int)
          train = np.concatenate((train,B))

      train = train[1:]
      all_train_interactions_arr.append(train)
      print("train", train.shape)

      # test
      test_index =  np.random.choice(test_index, int(p*len(test_index)), replace=False)
      # interactions + in test
      indice_P_t = np.c_[Ip[test_index],Jm[test_index], np.ones(len(test_index))].astype(int)

      # interactions - in test
      a = np.r_[np.c_[Inp,Jnm]] # all the zeros in the matrix (and NK ?)
      a1 = set(map(tuple, a))
      b = train[:,:2]   # all the interactions in the train
      b1 = set(map(tuple, b))
      indice_N_t = np.array(list(a1 - b1))#[:indice_P_t.shape[0],:] # we keep the same number of interactions - than interactions + in test, choosing the 0 in the matrix
      #print(len(indice_N_t))

      # add interactions np.nan in test

      if len(indice_N_t) == 0:
          # initialization
          indice_N_t = np.array([-1, -1]).reshape(1,2)

      c = np.r_[np.c_[Inkp,Jnkm]] # all the np.nan in the matrix

      if len(indice_N_t) < indice_P_t.shape[0]:
          # we add some interactions - in test to have the same number of interactions + and - in test choose in the np.nan in the matrix
          k = 0
          while len(indice_N_t) < indice_P_t.shape[0]+1:
              i = np.random.randint(0, len(c))
              if tuple(c[i]) not in b1:
                  indice_N_t = np.concatenate((indice_N_t, c[i].reshape(1,2)))
                  k += 1

      # we drop the first row of indice_N_t if is [-1, -1]
      if indice_N_t[0,0] == -1:
          indice_N_t = indice_N_t[1:,:]

      indice_N_t = indice_N_t[:len(indice_P_t),:]

      # we add the column of 0 for the etiquette
      indice_N_t = np.c_[indice_N_t, np.zeros(len(indice_N_t))].astype(int)
      test = np.r_[indice_P_t,indice_N_t]

      all_test_interactions_arr.append(test)
      print("test", test.shape)


  print("Train/test datasets prepared.")
  with open('data/train_arr.data', 'wb') as f:
    pickle.dump(all_train_interactions_arr, f)
  with open('data/test_arr.data', 'wb') as f:
    pickle.dump(all_test_interactions_arr, f)
  return all_train_interactions_arr, all_test_interactions_arr

def make_train_test_mol_orphan(df,nb_folds):
  """
    make train and test sets
    the molecules in the test set are not in the train set
  """

  # algo Matthieu corrected
  intMat = df.pivot(index='indfasta', columns="indsmiles", values='score').to_numpy(dtype=np.float16)

  # Set the different folds

  all_train_interactions_arr = []
  all_test_interactions_arr = []

  n_p,n_m = intMat.shape
  Ip, Jm = np.where(intMat==1)

  groups = np.array(Jm)
  group_kfold = GroupKFold(n_splits=5)

  nb_positive_inter = int(len(Ip))
  Inp, Jnm = np.where(intMat==0)
  Inkp, Jnkm = np.where(np.isnan(intMat))

  for train_index, test_index in group_kfold.split(range(nb_positive_inter), groups=groups):
      # 9' pour train

      Mm, bin_edges = np.histogram(Ip[train_index], bins = range(n_p+1)) # np.array with  #interactions for each protein of the train at the beginning

      Mp, bin_edges = np.histogram(Jm[train_index], bins = range(n_m+1)) # np.array with  #interactions for each drugs at the beginning (how manu time it can be chosen)

      train = np.zeros([1,3], dtype=int)

      nb_prot = len(list(set(Ip[train_index]))) # number of different prot in train
      for i in range(nb_prot):

          j = np.argmax(Mm) # choose protein with the maximum of interactions in the train

          indice_P = Jm[train_index][np.where(Ip[train_index]==j)[0]]  #np.array with index of interactions + in train
          indice_N = [k for k in Jm[train_index] if intMat[j][k]==0]
          indice_NK = [k for k in Jm[train_index] if np.isnan(intMat[j][k])] #np.array  with index of interactions not known

          indice_freq_mol = np.where(Mp>1)[0]  #drug's index with more than 2 interactions +
          indice_poss_mol = np.where(Mp == 1)[0]  #drug's index with 1 interaction +

          indice_freq_one_prot = np.intersect1d(indice_N, indice_freq_mol)
          indice_poss_one_prot = np.intersect1d(indice_N, indice_poss_mol)

          nb_positive_interactions = len(indice_P)
          nb_frequent_hitters_negative_interactions = len(indice_freq_one_prot)

          indice_freq_one_prot = np.intersect1d(indice_N, indice_freq_mol)
          indice_poss_one_prot = np.intersect1d(indice_N, indice_poss_mol)
          indice_freq_one_prot_NK = np.intersect1d(indice_NK, indice_freq_mol)
          indice_poss_one_prot_NK = np.intersect1d(indice_NK, indice_poss_mol)

          if len(indice_P) <= len(indice_freq_one_prot):
              # we shoot at random nb_positive_interactions in drugs with a lot of interactions
              indice_N_one_prot = np.random.choice(indice_freq_one_prot,
                                                  len(indice_P), replace = False)
          elif len(indice_P) <= len(indice_freq_one_prot) + len(indice_poss_one_prot):
              # we shoot at random nb_positive_interactions in drugs with a lot of interactions
              nb_negative_interactions_remaining = len(indice_P) - len(indice_freq_one_prot)
              indice_N_one_prot_poss = np.random.choice(indice_poss_one_prot,
                                                      nb_negative_interactions_remaining, replace = False )
              indice_N_one_prot = np.concatenate((indice_freq_one_prot,
                                              indice_N_one_prot_poss))
          elif len(indice_P) <= len(indice_freq_one_prot) + len(indice_poss_one_prot) + len(indice_freq_one_prot_NK):
              # we shoot at random nb_positive_interactions in drugs with a lot of interactions
              nb_negative_interactions_remaining = len(indice_P) - len(indice_freq_one_prot) - len(indice_poss_one_prot)
              indice_N_one_prot_poss = np.random.choice(indice_freq_one_prot_NK,
                                                      nb_negative_interactions_remaining, replace = False )
              indice_N_one_prot = np.concatenate((indice_freq_one_prot,
                                              indice_poss_one_prot, indice_N_one_prot_poss))
          else:
              # we shoot at random nb_positive_interactions in drugs with a lot of interactions
              nb_negative_interactions_remaining = len(indice_P) - len(indice_freq_one_prot) - len(indice_poss_one_prot) - len(indice_freq_one_prot_NK)
              #print("nb_negative_interactions_remaining", nb_negative_interactions_remaining) # pas de solution...
              #print(indice_poss_one_prot_NK.shape)
              indice_N_one_prot_poss = np.random.choice(indice_poss_one_prot_NK,
                                                      nb_negative_interactions_remaining, replace = False )
              indice_N_one_prot = np.concatenate((indice_freq_one_prot,
                                              indice_poss_one_prot, indice_freq_one_prot_NK, indice_N_one_prot_poss))

          Mp[indice_N_one_prot.astype(int)]-=1

          # this protein has been processed
          Mm[j] = 0

          indice = np.r_[indice_P,indice_N_one_prot].astype(int)
          etiquette = [x if not np.isnan(x) else 0 for x in intMat[j][indice]]
          A = np.stack((indice, etiquette), axis=-1)
          B = np.c_[np.zeros(A.shape[0])+j,A].astype(int)
          train = np.concatenate((train,B))

      train = train[1:]
      all_train_interactions_arr.append(train)
      print("train", train.shape)

      # test
      # interactions + in test
      indice_P_t = np.c_[Ip[test_index],Jm[test_index], np.ones(len(test_index))].astype(int)

      # interactions - in test
      a = np.r_[np.c_[Inp,Jnm]] # all the zeros in the matrix (and NK ?)
      a1 = set(map(tuple, a))
      b = train[:,:2]   # all the interactions in the train
      b1 = set(map(tuple, b))
      indice_N_t = np.array(list(a1 - b1))#[:indice_P_t.shape[0],:] # we keep the same number of interactions - than interactions + in test, choosing the 0 in the matrix
      #print(len(indice_N_t))

      # add interactions np.nan in test

      if len(indice_N_t) == 0:
          # initialization
          indice_N_t = np.array([-1, -1]).reshape(1,2)

      c = np.r_[np.c_[Inkp,Jnkm]] # all the np.nan in the matrix

      if len(indice_N_t) < indice_P_t.shape[0]:
          # we add some interactions - in test to have the same number of interactions + and - in test choose in the np.nan in the matrix
          k = 0
          while len(indice_N_t) < indice_P_t.shape[0]+1:
              i = np.random.randint(0, len(c))
              if tuple(c[i]) not in b1:
                  indice_N_t = np.concatenate((indice_N_t, c[i].reshape(1,2)))
                  k += 1

      # we drop the first row of indice_N_t if is [-1, -1]
      if indice_N_t[0,0] == -1:
          indice_N_t = indice_N_t[1:,:]

      indice_N_t = indice_N_t[:len(indice_P_t),:]

      # we add the column of 0 for the etiquette
      indice_N_t = np.c_[indice_N_t, np.zeros(len(indice_N_t))].astype(int)
      test = np.r_[indice_P_t,indice_N_t]

      all_test_interactions_arr.append(test)
      print("test", test.shape)
      


  print("Train/test datasets prepared.")
  with open('data/train_arr_mol_orphan.data', 'wb') as f:
    pickle.dump(all_train_interactions_arr, f)
  with open('data/test_arr_mol_orphan.data', 'wb') as f:
    pickle.dump(all_test_interactions_arr, f)
  return all_train_interactions_arr, all_test_interactions_arr

def make_train_test_prot_orphan(df,nb_folds):
  """
    make train and test sets
    the proteins in the test set are not in the train set
  """

  # algo Matthieu corrected
  intMat = df.pivot(index='indfasta', columns="indsmiles", values='score').to_numpy(dtype=np.float16)

  # Set the different folds

  all_train_interactions_arr = []
  all_test_interactions_arr = []

  n_p,n_m = intMat.shape
  Ip, Jm = np.where(intMat==1)

  groups = np.array(Ip)
  group_kfold = GroupKFold(n_splits=5)

  nb_positive_inter = int(len(Ip))
  Inp, Jnm = np.where(intMat==0)
  Inkp, Jnkm = np.where(np.isnan(intMat))

  for train_index, test_index in group_kfold.split(range(nb_positive_inter), groups=groups):
      # 9' pour train

      Mm, bin_edges = np.histogram(Ip[train_index], bins = range(n_p+1)) # np.array with  #interactions for each protein of the train at the beginning

      Mp, bin_edges = np.histogram(Jm[train_index], bins = range(n_m+1)) # np.array with  #interactions for each drugs at the beginning (how manu time it can be chosen)

      train = np.zeros([1,3], dtype=int)

      nb_prot = len(list(set(Ip[train_index]))) # number of different prot in train
      for i in range(nb_prot):

          j = np.argmax(Mm) # choose protein with the maximum of interactions in the train

          indice_P = Jm[train_index][np.where(Ip[train_index]==j)[0]]  #np.array with index of interactions + in train
          indice_N = [k for k in Jm[train_index] if intMat[j][k]==0]
          indice_NK = [k for k in Jm[train_index] if np.isnan(intMat[j][k])] #np.array  with index of interactions not known

          indice_freq_mol = np.where(Mp>1)[0]  #drug's index with more than 2 interactions +
          indice_poss_mol = np.where(Mp == 1)[0]  #drug's index with 1 interaction +

          indice_freq_one_prot = np.intersect1d(indice_N, indice_freq_mol)
          indice_poss_one_prot = np.intersect1d(indice_N, indice_poss_mol)

          nb_positive_interactions = len(indice_P)
          nb_frequent_hitters_negative_interactions = len(indice_freq_one_prot)

          indice_freq_one_prot = np.intersect1d(indice_N, indice_freq_mol)
          indice_poss_one_prot = np.intersect1d(indice_N, indice_poss_mol)
          indice_freq_one_prot_NK = np.intersect1d(indice_NK, indice_freq_mol)
          indice_poss_one_prot_NK = np.intersect1d(indice_NK, indice_poss_mol)

          if len(indice_P) <= len(indice_freq_one_prot):
              # we shoot at random nb_positive_interactions in drugs with a lot of interactions
              indice_N_one_prot = np.random.choice(indice_freq_one_prot,
                                                  len(indice_P), replace = False)
          elif len(indice_P) <= len(indice_freq_one_prot) + len(indice_poss_one_prot):
              # we shoot at random nb_positive_interactions in drugs with a lot of interactions
              nb_negative_interactions_remaining = len(indice_P) - len(indice_freq_one_prot)
              indice_N_one_prot_poss = np.random.choice(indice_poss_one_prot,
                                                      nb_negative_interactions_remaining, replace = False )
              indice_N_one_prot = np.concatenate((indice_freq_one_prot,
                                              indice_N_one_prot_poss))
          elif len(indice_P) <= len(indice_freq_one_prot) + len(indice_poss_one_prot) + len(indice_freq_one_prot_NK):
              # we shoot at random nb_positive_interactions in drugs with a lot of interactions
              nb_negative_interactions_remaining = len(indice_P) - len(indice_freq_one_prot) - len(indice_poss_one_prot)
              indice_N_one_prot_poss = np.random.choice(indice_freq_one_prot_NK,
                                                      nb_negative_interactions_remaining, replace = False )
              indice_N_one_prot = np.concatenate((indice_freq_one_prot,
                                              indice_poss_one_prot, indice_N_one_prot_poss))
          else:
              # we shoot at random nb_positive_interactions in drugs with a lot of interactions
              nb_negative_interactions_remaining = len(indice_P) - len(indice_freq_one_prot) - len(indice_poss_one_prot) - len(indice_freq_one_prot_NK)
              #print("nb_negative_interactions_remaining", nb_negative_interactions_remaining) # pas de solution...
              #print(indice_poss_one_prot_NK.shape)
              indice_N_one_prot_poss = np.random.choice(indice_poss_one_prot_NK,
                                                      nb_negative_interactions_remaining, replace = False )
              indice_N_one_prot = np.concatenate((indice_freq_one_prot,
                                              indice_poss_one_prot, indice_freq_one_prot_NK, indice_N_one_prot_poss))

          Mp[indice_N_one_prot.astype(int)]-=1

          # this protein has been processed
          Mm[j] = 0

          indice = np.r_[indice_P,indice_N_one_prot].astype(int)
          etiquette = [x if not np.isnan(x) else 0 for x in intMat[j][indice]]
          A = np.stack((indice, etiquette), axis=-1)
          B = np.c_[np.zeros(A.shape[0])+j,A].astype(int)
          train = np.concatenate((train,B))

      train = train[1:]
      all_train_interactions_arr.append(train)
      print("train", train.shape)

      # test
      # interactions + in test
      indice_P_t = np.c_[Ip[test_index],Jm[test_index], np.ones(len(test_index))].astype(int)

      # interactions - in test
      a = np.r_[np.c_[Inp,Jnm]] # all the zeros in the matrix (and NK ?)
      a1 = set(map(tuple, a))
      b = train[:,:2]   # all the interactions in the train
      b1 = set(map(tuple, b))
      indice_N_t = np.array(list(a1 - b1))#[:indice_P_t.shape[0],:] # we keep the same number of interactions - than interactions + in test, choosing the 0 in the matrix
      #print(len(indice_N_t))

      # add interactions np.nan in test

      if len(indice_N_t) == 0:
          # initialization
          indice_N_t = np.array([-1, -1]).reshape(1,2)

      c = np.r_[np.c_[Inkp,Jnkm]] # all the np.nan in the matrix

      if len(indice_N_t) < indice_P_t.shape[0]:
          # we add some interactions - in test to have the same number of interactions + and - in test choose in the np.nan in the matrix
          k = 0
          while len(indice_N_t) < indice_P_t.shape[0]+1:
              i = np.random.randint(0, len(c))
              if tuple(c[i]) not in b1:
                  indice_N_t = np.concatenate((indice_N_t, c[i].reshape(1,2)))
                  k += 1

      # we drop the first row of indice_N_t if is [-1, -1]
      if indice_N_t[0,0] == -1:
          indice_N_t = indice_N_t[1:,:]

      indice_N_t = indice_N_t[:len(indice_P_t),:]

      # we add the column of 0 for the etiquette
      indice_N_t = np.c_[indice_N_t, np.zeros(len(indice_N_t))].astype(int)
      test = np.r_[indice_P_t,indice_N_t]

      all_test_interactions_arr.append(test)
      print("test", test.shape)
      


  print("Train/test datasets prepared.")
  with open('data/train_arr_prot_orphan.data', 'wb') as f:
    pickle.dump(all_train_interactions_arr, f)
  with open('data/test_arr_prot_orphan.data', 'wb') as f:
    pickle.dump(all_test_interactions_arr, f)
  return all_train_interactions_arr, all_test_interactions_arr


def perf_metrics(all_p_pred_arr,all_test_interactions_arr,threshold,b):
    all_perf_arr = []
    for p_pred, test_interactions in zip(all_p_pred_arr,all_test_interactions_arr):
        perf = []
        y_true = test_interactions[:,2]

        # for all threshold

        #Compute average precision (AP) from prediction scores. This score corresponds to the area under the precision-recall curve.
        perf.append(metrics.average_precision_score(y_true, p_pred))
        # Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores.
        perf.append(metrics.roc_auc_score(y_true, p_pred))

        # # for threshold choose in argument
        y_pred = (p_pred > threshold).astype(int)

        # Compute the accuracy
        perf.append(metrics.accuracy_score(y_true, y_pred) )

        # Compute the F score, the weighted harmonic mean of precision and recall. The beta parameter determines the weight of recall in the combined score. beta < 1 lends more weight to precision, while beta > 1 favors recall (beta -> 0 considers only precision, beta -> +inf only recall).
        perf.append( metrics.fbeta_score(y_true, y_pred, beta=b) )

        # Compute the F1 score, also known as balanced F-score or F-measure
        perf.append(metrics.f1_score(y_true, y_pred) )

        # Compute the recall
        perf.append(metrics.recall_score(y_true, y_pred) )

        # Compute the precision
        perf.append( metrics.precision_score(y_true, y_pred))



        all_perf_arr.append(perf)
    mean = np.mean(all_perf_arr, axis=0)
    std = np.std(all_perf_arr, axis=0)
    return mean, std,['AUPR', 'ROC AUC',"Accuracy", "Fbeta", "F1", "Recall", "Precision"]
