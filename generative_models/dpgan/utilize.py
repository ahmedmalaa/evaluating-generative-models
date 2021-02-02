# pylint: disable=unbalanced-tuple-unpacking
import pickle

from matplotlib.pylab import (
    mean, array, nonzero, count_nonzero, putmask, around, split, clip, unique, where, concatenate, random
)

from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, roc_auc_score


def data_readf(top):  # pylint: disable=unused-argument
    '''Read MIMIC-III data'''
    with open('/home/xieliyan/Dropbox/GPU/Data/MIMIC-III/patient_vectors_1071.pickle', 'rb') as f_: # Original MIMIC-III data is in GPU1
        MIMIC_ICD9 = pickle.load(f_) # dictionary, each one is a list
    MIMIC_data = []
    for value in MIMIC_ICD9: # dictionary to numpy array
        if mean(value) == 0.0: # skip all zero vectors, each patients should have as least one disease of course
            continue
        MIMIC_data.append(value) # amax(MIMIC_data): 540
    # MIMIC_data = age_filter(MIMIC_data) # remove those patients with age 18 or younger
    # MIMIC_data = binarize(array(MIMIC_data)) # binarize, non zero -> 1, average(MIMIC_data): , type(MIMIC_data[][]): <type 'numpy.int64'>
    # index, MIMIC_data = select_code(MIMIC_data, top) # should be done after binarize because we consider the frequency among different patients, select top codes and remove the patients that don't have at least one of these codes, see "applying deep learning to icd-9 multi-label classification from medical records"
    # MIMIC_data = MIMIC_data[:, index] # keep only those coordinates (features) correspondent to top ICD9 codes
    num_data = (array(MIMIC_data).shape)[0] # data number
    dim_data = (array(MIMIC_data).shape)[1] # data dimension
    return array(MIMIC_data), num_data, dim_data # (46520, 942) 46520 942 for whole dataset


def c2b(train, generated, adj):
    '''Set the number of 1 in generated data as multiple time of in training data, the rest is set to 0 (or not)'''

    if count_nonzero(generated) <= count_nonzero(train): # special case: number of 1 in generated is <= train, all nonzero in train = 1
        putmask(generated, generated > 0, 1.0)
        return generated

    p = float(count_nonzero(train))/train.size # percentage of nonzero elements
    g = sorted(generated.flatten(), reverse=True)
    idx = int(around(adj*p*len(g))) # with adjustment
    v = g[idx] # any value large than this set to 1, o.w. to 0
    putmask(generated, generated<=v, 0.0) # due to the property of putmask, must first set 0 then set 1
    putmask(generated, generated>v, 1.0)
    print("Nonzero element portion in training data and adjustment value are:")
    print(p, adj)
    print("Nonzero element portion in generated data after adjustment of c2b function:")
    print(float(count_nonzero(generated))/generated.size)
    return generated


def c2bcolwise(train, generated, adj):
    '''Set the number of 1 in each column in generated data the same as the same column in training data, the rest is set to 0.
    Network learn the joint distribution p(x1,...xd), then it should also learn the marginal distribution p(x1),...,p(xd), which
    is approximately the frequent of 1 (and 0) in each feature (coordinate) x1...xd, hence it make sense to do so. But
    by doing so we "force" the generated data have the same portion of 1 in each feature (coordinate) no matter how the network
    is trained (even not trained at all), this doesn't matters since features (coordinates) are dependent, p(x1,...xd) != p(x1)*...*p(xd)
    only setting the frequency of 1 in each feature (coordinate) is not enough, it also relies on the training of NN to learn the
    dependency among features (coordinates), i.e. conditional probability of x1...xd'''
    generated_new = [] # store new one
    s = train.sum(axis=0)
    print('Nonzero element in each feature (coordinate) in training data: ')
    print(list(map(int, s))) # not in scientific notation
    print("Adjustment value is: " + str(adj))
    for col in range(len(s)):
        col_train = train[:,col]
        col_generated = generated[:,col]
        if count_nonzero(col_generated) <= count_nonzero(col_train): # special case: number of 1 in generated is <= train, all nonzero in train = 1
            putmask(col_generated, generated > 0, 1.0)
            generated_new.append(col_generated)
            continue
        g = sorted(col_generated, reverse=True)
        idx = int(adj*s[col]) # with adjustment
        v = g[idx]
        putmask(col_generated, col_generated<=v, 0.0)
        putmask(col_generated, col_generated>v, 1.0)
        generated_new.append(col_generated)
    generated_new = array(generated_new).T
    print('Nonzero element in each feature (coordinate) in generated data: ')
    print(list(map(int, generated_new.sum(axis=0))))
    print('Portion of element that is match between training data and generated data')
    print(float(sum(train == generated_new))/(train.shape[0]*train.shape[1]))
    return generated_new


def splitbycol(dataType, _VALIDATION_RATIO, col, MIMIC_data):
    '''Separate training and testing for each dimension (col), if we fix column col as label,
    we need to take _VALIDATION_RATIO of data with label 1 and _VALIDATION_RATIO of data with label 0
    and merge them together as testing set and leave the rest. Then balance the rest as training set
    by keeping whomever (0 or 1) is smaller and random select same number from the other one.
    Finally return training and testing set'''
    if dataType == 'binary':
        MIMIC_data = clip(MIMIC_data, 0, 1)
    _, c = split(MIMIC_data, col) # get column col
    if (unique(c).size == 1): # skip column: only one class
        return [], []
    MIMIC_data_1 = MIMIC_data[nonzero(c), :][0]  # Separate data matrix by label, label==1
    MIMIC_data_0 = MIMIC_data[where(c == 0)[0], :]
    trainX_1, testX_1 = train_test_split(MIMIC_data_1, test_size=_VALIDATION_RATIO, random_state=0)
    trainX_0, testX_0 = train_test_split(MIMIC_data_0, test_size=_VALIDATION_RATIO, random_state=0)
    testX = concatenate((testX_1, testX_0), axis=0)
    if len(trainX_1) == len(trainX_0):
        trainX = concatenate((trainX_1, trainX_0), axis=0)
    elif len(trainX_1) < len(trainX_0):
        temp_train, temp_test = train_test_split(trainX_0, test_size=len(trainX_1), random_state=0)
        trainX = concatenate((trainX_1, temp_test), axis=0)
        # testX = concatenate((testX, temp_train), axis=0) # can't merge, test set is already done
    else:
        temp_train, temp_test = train_test_split(trainX_1, test_size=len(trainX_0), random_state=0)
        trainX = concatenate((trainX_0, temp_test), axis=0)
        # testX = concatenate((testX, temp_train), axis=0)
    if ((array(trainX).shape)[0] == 0 or (array(testX).shape)[0] == 0): # skip column: no data point in training or testing set
        return [], []
    return trainX, testX # <type 'numpy.ndarray'> <type 'numpy.ndarray'>



def gene_check(col, x_gene):
    '''check if each column (coordinate) has one class or not, balance the data set then output'''
    _, c = split(x_gene, col)  # get column col
    if (unique(c).size == 1):  # skip column: only one class
        return []
    x_gene_1 = x_gene[nonzero(c), :][0]
    x_gene_0 = x_gene[where(c == 0)[0], :]
    if len(x_gene_1) == len(x_gene_0):
        geneX = x_gene
    elif len(x_gene_1) < len(x_gene_0):
        temp_train, temp_test = train_test_split(x_gene_0, test_size=len(x_gene_1), random_state=0)
        geneX = concatenate((x_gene_1, temp_test), axis=0)
    else:
        temp_train, temp_test = train_test_split(x_gene_1, test_size=len(x_gene_0), random_state=0)
        geneX = concatenate((x_gene_0, temp_test), axis=0)
    if (array(geneX).shape)[0] == 0:
        return []
    return x_gene


def statistics(r, g, te, col):
    '''Column specific statistics (precision, recall(Sensitivity), f1-score, AUC)'''
    f_r, t_r = split(r, col)  # separate feature and target
    f_g, t_g = split(g, col)
    f_te, t_te = split(te, col)  # these 6 parts are all numpy array
    # t_g[t_g < 1.0] = 0  # hard decision boundary
    # t_g[t_g >= 0.5] = 1
    if (unique(t_r).size == 1) or (unique(t_g).size == 1):  # if only those coordinates correspondent to top codes are kept, no coordinate should be skipped, if those patients that doesn't contain top ICD9 codes were removed, more coordinates will be skipped
        return [], [], [], [], [], [], [], [], [], []
    model_r = linear_model.LogisticRegression()  # logistic regression, if labels are all 0, this will cause: ValueError: This solver needs samples of at least 2 classes in the data, but the data contains only one class: 0
    model_r.fit(f_r, t_r)
    label_r = model_r.predict(f_te) # decision boundary is 0
    model_g = linear_model.LogisticRegression()
    model_g.fit(f_g, t_g)
    label_g = model_r.predict(f_te)
    precision_r = precision_score(t_te, label_r) # precision
    precision_g = precision_score(t_te, label_g)
    recall_r = recall_score(t_te, label_r) # recall
    recall_g = recall_score(t_te, label_g)
    acc_r = accuracy_score(t_te, label_r) # accuracy
    acc_g = accuracy_score(t_te, label_g)
    f1score_r = f1_score(t_te, label_r)  # f1-score
    f1score_g = f1_score(t_te, label_g)
    auc_r = roc_auc_score(t_te, label_r) # AUC
    auc_g = roc_auc_score(t_te, label_g)

    return precision_r, precision_g, recall_r, recall_g, acc_r, acc_g, f1score_r, f1score_g, auc_r, auc_g


def dwp(r, g, te, db=0.5, C=1.0):
    '''Dimension-wise prediction & dimension-wise probability, r for real, g for generated, t for test, all without separated feature and target, all are numpy array'''
    rv_pre = []
    gv_pre = []
    rv_pro = []
    gv_pro = []
    for i in range(len(r[0])):
        print(i)
        f_r, t_r = split(r, i) # separate feature and target
        f_g, t_g = split(g, i)
        f_te, t_te = split(te, i) # these 6 are all numpy array
        t_g[t_g < db ] = 0  # hard decision boundary
        t_g[t_g >= db ] = 1
        if (unique(t_r).size == 1) or (unique(t_g).size == 1): # if only those coordinates correspondent to top codes are kept, no coordinate should be skipped, if those patients that doesn't contain top ICD9 codes were removed, more coordinates will be skipped
            print("skip this coordinate")
            continue
        model_r = linear_model.LogisticRegression(C=C) # logistic regression, if labels are all 0, this will cause: ValueError: This solver needs samples of at least 2 classes in the data, but the data contains only one class: 0
        model_r.fit(f_r, t_r)
        label_r = model_r.predict(f_te)
        model_g = linear_model.LogisticRegression(C=C)
        model_g.fit(f_g, t_g)
        label_g = model_g.predict(f_te)
        # print(label_r)
        # print(mean(model_r.coef_), count_nonzero(model_r.coef_), mean(model_g.coef_), count_nonzero(model_g.coef_)) # statistics of classifiers
        # rv.append(match(label_r, t_te)/(len(t_te)+10**(-10))) # simply match
        # gv.append(match(label_g, t_te)/(len(t_te)+10**(-10)))
        rv_pre.append(f1_score(t_te, label_r)) # F1 score
        gv_pre.append(f1_score(t_te, label_g))
        # reg = linear_model.LinearRegression() # least square error
        # reg.fit(f_r, t_r)
        # target_r = reg.predict(f_te)
        # reg = linear_model.LinearRegression()
        # reg.fit(f_g, t_g)
        # target_g = reg.predict(f_te)
        # rv.append(square(linalg.norm(target_r-t_te)))
        # gv.append(square(linalg.norm(target_g-t_te)))
        rv_pro.append(float(count_nonzero(t_r))/len(t_r))  # dimension-wise probability, see "https://onlinecourses.science.psu.edu/stat504/node/28"
        gv_pro.append(float(count_nonzero(t_g))/len(t_g))

    return rv_pre, gv_pre, rv_pro, gv_pro


def load_MIMICIII(dataType, _VALIDATION_RATIO, top):
    MIMIC_data, num_data, dim_data = data_readf(top)
    if dataType == 'binary':
        MIMIC_data = clip(MIMIC_data, 0, 1)
    trainX, testX = train_test_split(MIMIC_data, test_size=_VALIDATION_RATIO, random_state=0)
    return trainX, testX, dim_data


def fig_add_noise(List):
    '''adding noise to results to make them distinguishable on figure'''
    print(len(List))
    print(0.0001*random.randn(len(List)))
    List_new = List + 0.0001*random.randn(len(List))
    return List_new
