from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.ensemble import RandomForestClassifier
import random
import numpy as np
import pandas as pd
from openne import graph, node2vec, hope, lap, grarep
import math
from bionev.GAE.train_model import gae_model
from bionev.utils import *
import os
import time

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import MinMaxScaler
import array
import random
import numpy
from sklearn import linear_model


def get_LNS(feature_matrix, neighbor_num):
    feature_matrix = np.matrix(feature_matrix)
    iteration_max = 40  # same as 2018 bibm
    mu = 3  # same as 2018 bibm
    X = feature_matrix
    alpha = np.power(X, 2).sum(axis=1)
    distance_matrix = np.sqrt(alpha + alpha.T - 2 * X * X.T)
    row_num = X.shape[0]
    e = np.ones((row_num, 1))
    distance_matrix = np.array(distance_matrix + np.diag(np.diag(e * e.T * np.inf)))
    sort_index = np.argsort(distance_matrix, kind='mergesort')
    nearest_neighbor_index = sort_index[:, :neighbor_num].flatten()
    nearest_neighbor_matrix = np.zeros((row_num, row_num))
    nearest_neighbor_matrix[np.arange(row_num).repeat(neighbor_num), nearest_neighbor_index] = 1
    C = nearest_neighbor_matrix
    np.random.seed(0)
    W = np.mat(np.random.rand(row_num, row_num), dtype=float)
    W = np.multiply(C, W)
    lamda = mu * e
    P = X * X.T + lamda * e.T
    for q in range(iteration_max):
        Q = W * P
        W = np.multiply(W, P) / Q
        W = np.nan_to_num(W)
    return np.array(W)


def get_link_from_similarity(similarity_matrix, positive_num):
    row_num = similarity_matrix.shape[0]
    sort_index = np.argsort(-similarity_matrix, kind='mergesort') 
    nearest_neighbor_index = sort_index[:, :positive_num].flatten()
    nearest_neighbor_matrix = np.zeros((row_num, row_num))
    nearest_neighbor_matrix[np.arange(row_num).repeat(positive_num), nearest_neighbor_index] = 1
    return nearest_neighbor_matrix


def construct_net(train_LMI, positive_num):  
    miRNA_seq = np.loadtxt('mi_seq_L.csv', delimiter=',', dtype=float)
    lncRNA_seq = np.loadtxt('lnc_seq_L.csv', delimiter=',', dtype=float)

    mi_LNS_sim = get_LNS(miRNA_seq, int(len(miRNA_seq) * 0.8))
    lnc_LNS_sim = get_LNS(lncRNA_seq, int(len(lncRNA_seq) * 0.8))

    mi_intra_link = get_link_from_similarity(mi_LNS_sim, positive_num)
    lnc_intra_link = get_link_from_similarity(lnc_LNS_sim, positive_num)

    mat1 = np.hstack((lnc_intra_link, train_LMI))
    mat2 = np.hstack((train_LMI.T, mi_intra_link))
    return np.vstack((mat1, mat2))

def net2edgelist(lncRNA_miRNA_matrix_net):  
    none_zero_position = np.where(np.triu(lncRNA_miRNA_matrix_net) != 0)  
    none_zero_row_index = np.mat(none_zero_position[0], dtype=int).T
    none_zero_col_index = np.mat(none_zero_position[1], dtype=int).T
    none_zero_dgelist = np.array(np.hstack((none_zero_row_index, none_zero_col_index)))
    np.savetxt('L_lncRNA_miRNA_edgelist_L.txt', none_zero_dgelist, fmt="%d", delimiter=' ')

def get_dw_embedding_matrix(lncRNA_miRNA_matrix_net, graph1):
    model = node2vec.Node2vec(graph=graph1, path_length=80, num_paths=30, dim=30, dw=True)  
    vec = model.vectors
    matrix = np.zeros((len(lncRNA_miRNA_matrix_net), len(list(vec.values())[0])))
    for key, value in vec.items():
        matrix[int(key), :] = value
    return matrix

def get_hope_embedding_matrix(lncRNA_miRNA_matrix_net, graph1):
    model = hope.HOPE(graph=graph1, d=120)
    vec = model.vectors
    matrix = np.zeros((len(lncRNA_miRNA_matrix_net), len(list(vec.values())[0])))
    for key, value in vec.items():
        matrix[int(key), :] = value
    return matrix

def get_lap_embedding_matrix(lncRNA_miRNA_matrix_net, graph1):
    model = lap.LaplacianEigenmaps(graph1, rep_size=120)
    vec = model.vectors
    matrix = np.zeros((len(lncRNA_miRNA_matrix_net), len(list(vec.values())[0])))
    for key, value in vec.items():
        matrix[int(key), :] = value
    return matrix

def get_GraRep_embedding_matrix(lncRNA_miRNA_matrix_net, graph1):
    model = grarep.GraRep(graph=graph1, Kstep=1, dim=120)  
    vec = model.vectors
    matrix = np.zeros((len(lncRNA_miRNA_matrix_net), len(list(vec.values())[0])))
    for key, value in vec.items():
        matrix[int(key), :] = value
    return matrix

def get_embddeing_from_txt(txt_file):
    lines = []
    with open(txt_file, 'r') as f_wr:
        content = f_wr.readlines()
        for i in range(1, len(content)):
            emb = []
            num = content[i].split()
            node = num[0]
            for col in num[1:]:
                emb.append(float(col))
            line = int(node), np.array(emb)
            lines.append(line)
        emb_dict = dict(lines)
        return emb_dict
        
# modify your file path
def get_GAE_embedding_matrix(lncRNA_miRNA_matrix_net, hidden_unit=512):
    cmd_path1 = 'start cmd /k bionev --input L_lncRNA_miRNA_edgelist_L.txt --output GAE_emb_L.txt  --method GAE  --dimensions 120  --gae_model_selection gcn_vae  --epochs 500  --hidden %s'%(str(hidden_unit))
    os.system(cmd_path1)
    print('wait 35 second for  GAE_emb_L.txt')
    time.sleep(35)
    vec = get_embddeing_from_txt('GAE_emb_L.txt')

    matrix = np.zeros((len(lncRNA_miRNA_matrix_net), len(list(vec.values())[0])))
    for key, value in vec.items():
        matrix[int(key), :] = value
    return matrix

def get_individual_emb(train_lncRNA_miRNA_matrix, positive_num):
    lncRNA_miRNA_matrix_net = construct_net(train_lncRNA_miRNA_matrix, positive_num)
    net2edgelist(np.mat(lncRNA_miRNA_matrix_net))
    graph1 = graph.Graph()
    graph1.read_edgelist("L_lncRNA_miRNA_edgelist_L.txt")

    dw_lncRNA_miRNA_emb = get_dw_embedding_matrix(np.mat(lncRNA_miRNA_matrix_net), graph1)  #
    dw_lncRNA_len = train_lncRNA_miRNA_matrix.shape[0]
    lncRNA_emb_dw = np.array(dw_lncRNA_miRNA_emb[0:dw_lncRNA_len, 0:])  #
    miRNA_emb_dw = np.array(dw_lncRNA_miRNA_emb[dw_lncRNA_len::, 0:])

    hope_lncRNA_miRNA_emb = get_hope_embedding_matrix(np.mat(lncRNA_miRNA_matrix_net), graph1)  #
    hope_lncRNA_len = train_lncRNA_miRNA_matrix.shape[0]
    lncRNA_emb_hope = np.array(hope_lncRNA_miRNA_emb[0:hope_lncRNA_len, 0:])
    miRNA_emb_hope = np.array(hope_lncRNA_miRNA_emb[hope_lncRNA_len::, 0:])

    lap_lncRNA_miRNA_emb = get_lap_embedding_matrix(np.mat(lncRNA_miRNA_matrix_net), graph1)  #
    lap_lncRNA_len = train_lncRNA_miRNA_matrix.shape[0]
    lncRNA_emb_lap = np.array(lap_lncRNA_miRNA_emb[0:lap_lncRNA_len, 0:])
    miRNA_emb_lap = np.array(lap_lncRNA_miRNA_emb[lap_lncRNA_len::, 0:])

    GraRep_lncRNA_miRNA_emb = get_GraRep_embedding_matrix(np.mat(lncRNA_miRNA_matrix_net), graph1)  #
    GraRep_lncRNA_len = train_lncRNA_miRNA_matrix.shape[0]
    lncRNA_emb_GraRep = np.array(GraRep_lncRNA_miRNA_emb[0:GraRep_lncRNA_len, 0:])
    miRNA_emb_GraRep = np.array(GraRep_lncRNA_miRNA_emb[GraRep_lncRNA_len::, 0:])

    GAE_lncRNA_miRNA_emb = get_GAE_embedding_matrix(np.mat(lncRNA_miRNA_matrix_net), hidden_unit=512)   
    GAE_lncRNA_len = train_lncRNA_miRNA_matrix.shape[0]
    GAE_lncRNA_emb = np.array(GAE_lncRNA_miRNA_emb[0:GAE_lncRNA_len, 0:])
    GAE_miRNA_emb = np.array(GAE_lncRNA_miRNA_emb[GAE_lncRNA_len::, 0:])

    return [lncRNA_emb_dw, miRNA_emb_dw, lncRNA_emb_hope, miRNA_emb_hope, lncRNA_emb_lap, miRNA_emb_lap, lncRNA_emb_GraRep, miRNA_emb_GraRep, GAE_lncRNA_emb, GAE_miRNA_emb]


def get_train_data(train_lncRNA_miRNA_matrix, train_row, train_col, lnc_feature, mi_feature):
    train_feature = []  #
    train_label = []  #

    for num in range(len(train_row)):
        feature_vector = np.append(lnc_feature[train_row[num], :], mi_feature[train_col[num], :])
        train_feature.append(feature_vector)
        train_label.append(train_lncRNA_miRNA_matrix[train_row[num], train_col[num]])  

    train_feature = np.array(train_feature)
    train_label = np.array(train_label)
    return [train_feature, train_label]


def get_test_data(lncRNA_miRNA_matrix, lnc_feature, mi_feature, testPosition):
    test_feature = []
    test_label = []

    for num in range(len(testPosition)):
        feature_vector = np.append(lnc_feature[testPosition[num][0], :], mi_feature[testPosition[num][1], :])
        test_feature.append(feature_vector)
        test_label.append(lncRNA_miRNA_matrix[testPosition[num][0], testPosition[num][1]])  # 

    test_feature = np.array(test_feature)
    test_label = np.array(test_label)
    return [test_feature, test_label]


def get_Metrics(real_score, predict_score):
    sorted_predict_score = sorted(list(set(np.array(predict_score).flatten())))
    sorted_predict_score_num = len(sorted_predict_score)
    thresholdlist = []
    for i in range(999):
        threshold = sorted_predict_score[int(math.ceil(sorted_predict_score_num * (i + 1) / 1000) - 1)]
        thresholdlist.append(threshold)
    thresholds = np.matrix(thresholdlist)
    TN = np.zeros((1, len(thresholdlist)))
    TP = np.zeros((1, len(thresholdlist)))
    FN = np.zeros((1, len(thresholdlist)))
    FP = np.zeros((1, len(thresholdlist)))
    for i in range(thresholds.shape[1]):
        p_index = np.where(predict_score >= thresholds[0, i])
        TP[0, i] = len(np.where(real_score[p_index] == 1)[0])
        FP[0, i] = len(np.where(real_score[p_index] == 0)[0])
        n_index = np.where(predict_score < thresholds[0, i])
        FN[0, i] = len(np.where(real_score[n_index] == 1)[0])
        TN[0, i] = len(np.where(real_score[n_index] == 0)[0])
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    sen = TP / (TP + FN)
    recall = sen
    spec = TN / (TN + FP)
    precision = TP / (TP + FP)
    f1 = 2 * recall * precision / (recall + precision)
    max_index = np.argmax(f1)
    max_f1 = f1[0, max_index]
    max_accuracy = accuracy[0, max_index]
    max_recall = recall[0, max_index]
    max_spec = spec[0, max_index]
    max_precision = precision[0, max_index]
    return [max_f1, max_accuracy, max_recall, max_spec, max_precision]


def model_evaluate(real_score, predict_score):
    aupr = average_precision_score(real_score, predict_score)
    auc = roc_auc_score(real_score, predict_score)
    [f1,accuracy,recall,spec,precision] = get_Metrics(real_score,predict_score)
    return np.array([aupr, auc, f1, accuracy, recall, spec, precision])


def make_prediction(train_feature_matrix, train_label_vector, test_feature_matrix, seed):
    clf = RandomForestClassifier(random_state=seed, n_estimators=200, oob_score=True, n_jobs=-1)
    clf.fit(train_feature_matrix, train_label_vector)
    predict_y_proba = np.array(clf.predict_proba(test_feature_matrix)[:, 1])
    return predict_y_proba


def holdout_by_link(train_lncRNA_miRNA_matrix, ratio, seed, zero_row_index, zero_col_index):
    link_number = 0
    link_position = []
    nonLinksPosition = []  
    for i in range(0, train_lncRNA_miRNA_matrix.shape[0]):
        for j in range(0, train_lncRNA_miRNA_matrix.shape[1]):
            if train_lncRNA_miRNA_matrix[i, j] == 1:
                link_number = link_number + 1
                link_position.append([i, j])

    for k in range(len(zero_row_index)):
        nonLinksPosition.append([zero_row_index[k], zero_col_index[k]])
    no_link_number = len(zero_row_index)
    print('holdout_by_link     len(nonLinksPosition)', len(nonLinksPosition))

    print('holdout_by_link  link_number', link_number)
    link_position = np.array(link_position)
    index = np.arange(0, link_number)
    random.seed(seed)
    random.shuffle(index)
    train_index = index[(int(link_number * ratio) + 1):]
    test_index = index[0:int(link_number * ratio)]
    testLinkPosition = link_position[test_index]
    trainLinkPosition = link_position[train_index]

    no_link_position = np.array(nonLinksPosition)
    index2 = np.arange(0, no_link_number)
    random.seed(seed)
    random.shuffle(index2)
    train_index_2 = index2[(int(no_link_number * ratio) + 1):]
    test_index_2 = index2[0:int(no_link_number * ratio)]
    test_no_LinkPosition = no_link_position[test_index_2]
    train_no_LinkPosition = no_link_position[train_index_2]
    print('len(testLinkPosition)', len(testLinkPosition))
    print('len(test_no_LinkPosition)', len(test_no_LinkPosition))

    testPosition1 = np.vstack((test_no_LinkPosition, testLinkPosition))
    test_index_0 = [m for m in range(len(testPosition1))]
    random.seed(seed)
    random.shuffle(test_index_0)
    testPosition1_shuffle = [testPosition1[y] for y in test_index_0]

    trainPosition1 = np.vstack((train_no_LinkPosition, trainLinkPosition))
    train_index_0 = [m for m in range(len(trainPosition1))]
    random.seed(seed)
    random.shuffle(train_index_0)
    trainPosition1_shuffle = [trainPosition1[y] for y in train_index_0]
    print('len(trainPosition1_shuffle), len(testPosition1_shuffle)',
          len(trainPosition1_shuffle), len(testPosition1_shuffle))
    return trainPosition1_shuffle, testPosition1_shuffle


def ensemble_method(lncRNA_miRNA_matrix, train_row_shuffle, train_col_shuffle, testPosition, seed,
                    lncRNA_emb_dw, miRNA_emb_dw, lncRNA_emb_hope, miRNA_emb_hope, lncRNA_emb_lap, miRNA_emb_lap,
                    lncRNA_emb_GraRep, miRNA_emb_GraRep,GAE_lncRNA_emb, GAE_miRNA_emb
                    ):

    [dw_train_feature, dw_train_label] = get_train_data(lncRNA_miRNA_matrix, train_row_shuffle, train_col_shuffle, lncRNA_emb_dw, miRNA_emb_dw)
    [dw_test_feature, dw_test_label] = get_test_data(lncRNA_miRNA_matrix, lncRNA_emb_dw, miRNA_emb_dw, testPosition)  

    [hope_train_feature, hope_train_label] = get_train_data(lncRNA_miRNA_matrix, train_row_shuffle, train_col_shuffle, lncRNA_emb_hope, miRNA_emb_hope)
    [hope_test_feature, hope_test_label] = get_test_data(lncRNA_miRNA_matrix, lncRNA_emb_hope, miRNA_emb_hope, testPosition)

    [lap_train_feature, lap_train_label] = get_train_data(lncRNA_miRNA_matrix, train_row_shuffle, train_col_shuffle, lncRNA_emb_lap, miRNA_emb_lap)
    [lap_test_feature, lap_test_label] = get_test_data(lncRNA_miRNA_matrix, lncRNA_emb_lap, miRNA_emb_lap, testPosition)

    [GraRep_train_feature, GraRep_train_label] = get_train_data(lncRNA_miRNA_matrix, train_row_shuffle, train_col_shuffle, lncRNA_emb_GraRep,miRNA_emb_GraRep)
    [GraRep_test_feature, GraRep_test_label] = get_test_data(lncRNA_miRNA_matrix, lncRNA_emb_GraRep,miRNA_emb_GraRep, testPosition)

    [GAE_train_feature, GAE_train_label] = get_train_data(lncRNA_miRNA_matrix, train_row_shuffle, train_col_shuffle, GAE_lncRNA_emb, GAE_miRNA_emb)
    [GAE_test_feature, GAE_test_label] = get_test_data(lncRNA_miRNA_matrix, GAE_lncRNA_emb, GAE_miRNA_emb, testPosition)

    dw_prob = make_prediction(dw_train_feature, dw_train_label, dw_test_feature, seed)
    hope_prob = make_prediction(hope_train_feature, hope_train_label, hope_test_feature, seed)
    lap_prob = make_prediction(lap_train_feature, lap_train_label, lap_test_feature, seed)
    GraRep_prob = make_prediction(GraRep_train_feature, GraRep_train_label, GraRep_test_feature, seed)
    GAE_prob = make_prediction(GAE_train_feature, GAE_train_label, GAE_test_feature, seed)

    mul_emb_prob = [list(dw_prob), list(hope_prob), list(lap_prob), list(GraRep_prob), list(GAE_prob)]
    return mul_emb_prob

def LR_ensemble_method(lncRNA_miRNA_matrix, trainPosition1, testPosition, seed,
                       lncRNA_emb_dw, miRNA_emb_dw, lncRNA_emb_hope, miRNA_emb_hope, lncRNA_emb_lap, miRNA_emb_lap,
                       lncRNA_emb_GraRep, miRNA_emb_GraRep,
                       GAE_lncRNA_emb, GAE_miRNA_emb
                       ):

    train_row_shuffle = []
    train_col_shuffle = []
    for t in range(len(trainPosition1)):
        train_row_shuffle.append(trainPosition1[t][0])
        train_col_shuffle.append(trainPosition1[t][1])

    [dw_train_feature, dw_train_label] = get_train_data(lncRNA_miRNA_matrix, train_row_shuffle, train_col_shuffle, lncRNA_emb_dw, miRNA_emb_dw)
    [dw_test_feature, dw_test_label] = get_test_data(lncRNA_miRNA_matrix, lncRNA_emb_dw, miRNA_emb_dw, testPosition)

    [hope_train_feature, hope_train_label] = get_train_data(lncRNA_miRNA_matrix, train_row_shuffle, train_col_shuffle, lncRNA_emb_hope, miRNA_emb_hope)
    [hope_test_feature, hope_test_label] = get_test_data(lncRNA_miRNA_matrix, lncRNA_emb_hope, miRNA_emb_hope, testPosition)

    [lap_train_feature, lap_train_label] = get_train_data(lncRNA_miRNA_matrix, train_row_shuffle, train_col_shuffle, lncRNA_emb_lap, miRNA_emb_lap)
    [lap_test_feature, lap_test_label] = get_test_data(lncRNA_miRNA_matrix, lncRNA_emb_lap, miRNA_emb_lap, testPosition)

    [GraRep_train_feature, GraRep_train_label] = get_train_data(lncRNA_miRNA_matrix, train_row_shuffle, train_col_shuffle, lncRNA_emb_GraRep,miRNA_emb_GraRep)
    [GraRep_test_feature, GraRep_test_label] = get_test_data(lncRNA_miRNA_matrix, lncRNA_emb_GraRep,miRNA_emb_GraRep, testPosition)

    [GAE_train_feature, GAE_train_label] = get_train_data(lncRNA_miRNA_matrix, train_row_shuffle, train_col_shuffle, GAE_lncRNA_emb, GAE_miRNA_emb)
    [GAE_test_feature, GAE_test_label] = get_test_data(lncRNA_miRNA_matrix, GAE_lncRNA_emb, GAE_miRNA_emb, testPosition)

    dw_prob = make_prediction(dw_train_feature, dw_train_label, dw_test_feature, seed)
    hope_prob = make_prediction(hope_train_feature, hope_train_label, hope_test_feature, seed)
    lap_prob = make_prediction(lap_train_feature, lap_train_label, lap_test_feature, seed)
    GraRep_prob = make_prediction(GraRep_train_feature, GraRep_train_label, GraRep_test_feature, seed)
    GAE_prob = make_prediction(GAE_train_feature, GAE_train_label, GAE_test_feature, seed)

    mul_emb_prob = [list(dw_prob), list(hope_prob), list(lap_prob), list(GraRep_prob), list(GAE_prob)]
    return mul_emb_prob

def ensemble_scoring(real_matrix, multiple_matrix, testPosition, LR_cf1):  
    real_labels = []
    for i in range(0, len(testPosition)):
        real_labels.append(real_matrix[testPosition[i][0], testPosition[i][1]])

    all_test = []
    for i in range(0, len(testPosition)):
        vector=[]
        for indivadual_matrix in multiple_matrix:
            vector.append(indivadual_matrix[i])  
        all_test.append(vector)

    all_test_vec = np.array(all_test)
    print(all_test_vec.shape, all_test_vec.shape[0])

    all_pred = np.array(LR_cf1.predict_proba(all_test_vec)[:, 1])
    print(all_pred.shape)

    result_cf1=model_evaluate(np.array(real_labels), all_pred)

    return result_cf1


def get_LR_clf(real_matrix, multiple_matrix, testPosition):
    input_matrix = []
    output_matrix = []
    for i in range(0, len(testPosition)):
        output_matrix.append(real_matrix[testPosition[i][0], testPosition[i][1]])

    for m in range(len(multiple_matrix[0])):
        vector = []
        for j in range(0, len(multiple_matrix)):
            vector.append(multiple_matrix[j][m])
        input_matrix.append(vector)

    input_matrix = np.array(input_matrix)
    output_matrix = np.array(output_matrix)

    print('get_LR_clf    output_matrix.shape', output_matrix.shape)
    print('get_LR_clf    input_matrix.shape', input_matrix.shape)

    clf1 = linear_model.LogisticRegression(C=1.0, penalty='l2', tol=1e-6) 
    clf1.fit(input_matrix, output_matrix)
    return clf1

def cross_validation_experiment(lncRNA_miRNA_matrix, seed, k_folds):
    none_zero_position = np.where(lncRNA_miRNA_matrix != 0)
    none_zero_row_index = none_zero_position[0]
    none_zero_col_index = none_zero_position[1]

    zero_position = np.where(lncRNA_miRNA_matrix == 0)
    zero_row_index = zero_position[0]
    zero_col_index = zero_position[1]

    positive_randomlist = [i for i in range(len(none_zero_row_index))]
    random.seed(seed)
    random.shuffle(positive_randomlist)

    metric = np.zeros((1, 7))
    metric_csv = []
    size_of_cv = int(len(none_zero_row_index) / k_folds)

    print("seed = %d, evaluating lncRNA-miRNA......" % (seed))
    for k in range(k_folds):
        print("------cross validation (round = %d)------" % (k + 1))
        if k != k_folds - 1:
            positive_test = positive_randomlist[k * size_of_cv:(k + 1) * size_of_cv]
            positive_train = list(set(positive_randomlist).difference(set(positive_test)))
        else:
            positive_test = positive_randomlist[k * size_of_cv::]
            positive_train = list(set(positive_randomlist).difference(set(positive_test)))

        positive_train_row = none_zero_row_index[positive_train]
        positive_train_col = none_zero_col_index[positive_train]
        train_row = np.append(positive_train_row, zero_row_index)
        train_col = np.append(positive_train_col, zero_col_index)
        index3 = [v for v in range(len(train_row))]
        random.seed(seed)
        random.shuffle(index3)
        train_row_shuffle = [train_row[p] for p in index3]
        train_col_shuffle = [train_col[p] for p in index3]


        test_row = none_zero_row_index[positive_test]  
        test_col = none_zero_col_index[positive_test]
        train_lncRNA_miRNA_matrix = np.copy(lncRNA_miRNA_matrix)  
        train_lncRNA_miRNA_matrix[test_row, test_col] = 0  

        testPosition2 = []
        test_position_row = list(test_row) + list(zero_row_index)
        test_position_col = list(test_col) + list(zero_col_index)
        # shuffle
        test_index2 = [m for m in range(len(test_position_row))]
        random.seed(seed)
        random.shuffle(test_index2)
        test_position_row_shuffle = [test_position_row[j] for j in test_index2]
        test_position_col_shuffle = [test_position_col[j] for j in test_index2]
        
        if len(test_position_row) == len(test_position_col):
            for i in range(len(test_position_row)):
                testPosition2.append([test_position_row_shuffle[i], test_position_col_shuffle[i]])


        [lncRNA_emb_dw, miRNA_emb_dw, lncRNA_emb_hope, miRNA_emb_hope, lncRNA_emb_lap,miRNA_emb_lap, lncRNA_emb_GraRep, miRNA_emb_GraRep,
         GAE_lncRNA_emb, GAE_miRNA_emb] = get_individual_emb(copy.deepcopy(train_lncRNA_miRNA_matrix), 10)

        trainPosition1, testPosition1 = holdout_by_link(copy.deepcopy(train_lncRNA_miRNA_matrix), 0.25, seed, zero_row_index, zero_col_index)

        mul_emb_prob_for_ensemble = LR_ensemble_method(copy.deepcopy(lncRNA_miRNA_matrix), trainPosition1, testPosition1, seed,
                                                       lncRNA_emb_dw, miRNA_emb_dw, lncRNA_emb_hope, miRNA_emb_hope,
                                                       lncRNA_emb_lap, miRNA_emb_lap, lncRNA_emb_GraRep,
                                                       miRNA_emb_GraRep,
                                                       GAE_lncRNA_emb, GAE_miRNA_emb
                                                       )

        LR_clf1 = get_LR_clf(copy.deepcopy(lncRNA_miRNA_matrix), mul_emb_prob_for_ensemble, testPosition1)


        multiple_predict_matrix = ensemble_method(copy.deepcopy(lncRNA_miRNA_matrix),
                                                  train_row_shuffle, train_col_shuffle, testPosition2, seed,
                                                  lncRNA_emb_dw, miRNA_emb_dw, lncRNA_emb_hope, miRNA_emb_hope,
                                                  lncRNA_emb_lap, miRNA_emb_lap, lncRNA_emb_GraRep, miRNA_emb_GraRep,
                                                  GAE_lncRNA_emb, GAE_miRNA_emb
                                                  )

        ensemble_results = ensemble_scoring(copy.deepcopy(lncRNA_miRNA_matrix), multiple_predict_matrix, testPosition2, LR_clf1)

        print([round(i, 4) for i in ensemble_results])
        metric_csv.append(ensemble_results)
        metric += ensemble_results

    metric = metric / k_folds
    print([round(i, 4) for i in metric[0]])
    metric_csv.append(metric[0])
    df = pd.DataFrame(np.array(metric_csv),
                      index=['Round 1', 'Round 2', 'Round 3', 'Round 4', 'Round 5', 'Average'],
                      columns=['AUPR', 'AUC', 'F1', 'ACC', 'REC', 'SPEC', 'PRE'])
    df.to_csv('5_all_emb_L/'+'seed_' + str(seed) + '.csv')
    return metric


if __name__ == "__main__":
    k_folds = 5
    result_csv = []
    l_m_matrix = np.loadtxt('LMI_L.csv', delimiter=',', dtype=int)
    for seed in range(0, 21):
        # seq similarity top 10 convert to 1
        temp_result = cross_validation_experiment(l_m_matrix, seed, k_folds)
        result_csv.append(temp_result[0])

    df = pd.DataFrame(np.array(result_csv),
                      columns=['AUPR', 'AUC', 'F1', 'ACC', 'REC', 'SPEC', 'PRE'])
    df.to_csv('5_all_emb_L/20_seed.csv')
