# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 18:17:49 2022

@author: Yunrui QIU
"""

from IGME_MD_Analysis import IntegralGME
import numpy as np
import os
import scipy.linalg
import datetime
import warnings


def squared_difference(TPM, dimension, A_matrix, T_hat_matrix, end_frame=100, station_point=10, tpm_ref=None):
    eigenval, eigenvec = scipy.linalg.eig(TPM[station_point], right=False, left=True)
    eigenval = eigenval.real
    eigenvec = eigenvec.real
    tolerance = 1e-10
    mask = abs(max(eigenval) - eigenval) < tolerance
    station_pop = eigenvec[:, mask].T
    station_pop /= np.sum(station_pop)
    station_pop = np.diag(np.reshape(station_pop, dimension))
    TPM_prop_igme = np.zeros((end_frame, dimension, dimension))
    TPM_prop_igme[0] = np.dot(A_matrix, T_hat_matrix)
    error = np.sum(np.power(np.dot(station_pop, (tpm_ref[0] - TPM_prop_igme[0])), 2))

    for i in range(1, end_frame):
        TPM_prop_igme[i] = np.dot(TPM_prop_igme[i - 1], T_hat_matrix)
        error += np.sum(np.power(np.dot(station_pop, (tpm_ref[i] - TPM_prop_igme[i])), 2))
    error = np.sqrt(error / end_frame / dimension ** 2)
    return error

def rmse_on_training(TPM, dimension, A_matrix, T_hat_matrix, ini_point=10, end_point=100, station_point=10, tpm_ref=None):
    eigenval, eigenvec = scipy.linalg.eig(TPM[station_point], right=False, left=True)
    eigenval = eigenval.real
    eigenvec = eigenvec.real
    tolerance = 1e-10
    mask = abs(max(eigenval) - eigenval) < tolerance
    station_pop = eigenvec[:, mask].T
    station_pop /= np.sum(station_pop)
    station_pop = np.diag(np.reshape(station_pop, dimension))
    TPM_prop_igme = np.zeros((end_point, dimension, dimension))
    TPM_prop_igme[0] = np.dot(A_matrix, T_hat_matrix)
    error = 0
    for i in range(1, end_point):
        TPM_prop_igme[i] = np.dot(TPM_prop_igme[i - 1], T_hat_matrix)
        if i >= ini_point:
            error += np.sum(np.power(np.dot(station_pop, (tpm_ref[i] - TPM_prop_igme[i])), 2))
    error = np.sqrt(error / (end_point-ini_point) / dimension ** 2)
    return error



# FIP35 Macrostates
begin_time = datetime.datetime.now()
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings('ignore')
input_data = np.loadtxt('/home/yqiu78/workspace/Dataset/FIP35_Data/fip35_tpm_4states_row_norm_1ns_2us.txt', dtype=float)
TPM = np.loadtxt("/home/yqiu78/workspace/Dataset/FIP35_Data/fip35_tpm_4states_row_norm_1ns_2us.txt", dtype=float)
TPM = np.reshape(TPM, (len(TPM), int(np.sqrt(len(TPM[0]))), int(np.sqrt(len(TPM[0])))))
igme = IntegralGME(input_len=2000, dimension=4, delta_time=1)
igme.get_data(input_data=input_data)
igme.pre_set_data()
for k in range(3, 400):
    end_point = k
    delta_time = 1
    its_vb_rmse = np.zeros((end_point-2, 5))
    mfpt = np.zeros((end_point-2, 18))

    for i in range(0, end_point-2):
        igme.initial_for_gradient_descent(tau_ini=i, tau_end=end_point, initial_seeds='linear')
        print("***************fitting_range_"+str(i+1)+"__"+str(end_point)+"*************")
        A_matrix, T_hat_matrix = igme.gradient_descent_opt(learning_rate=1e-7, row_rate=1, detail_rate=0.1, momentum=0.8, epochs=1001, output_file=False)
        eigen_val, eigen_vec = np.linalg.eig(T_hat_matrix)
        eigen_val = eigen_val.real
        idx_order = np.argsort(eigen_val)
        its_vb_rmse[i, 0] = i+1
        its_vb_rmse[i, 1] = end_point
        mfpt[i, 0] = i+1
        mfpt[i, 1] = end_point
        its = -delta_time / np.log(eigen_val[idx_order[-2]])
        if not np.isinf(its):
            its_vb_rmse[i, 2] = -delta_time / np.log(eigen_val[idx_order[-2]])
        else:
            its_vb_rmse[i, 2] = its_vb_rmse[i-1, 2]
        its_vb_rmse[i, 3] = rmse_on_training(TPM=TPM, dimension=4, A_matrix=A_matrix, T_hat_matrix=T_hat_matrix, ini_point=i, end_point=end_point, station_point=80, tpm_ref=TPM)
        its_vb_rmse[i, 4] = squared_difference(TPM=TPM, dimension=4, A_matrix=A_matrix, T_hat_matrix=T_hat_matrix, end_frame=800, station_point=80, tpm_ref=TPM)
        mfpt_point = igme.mean_first_passage_time(T_hat_matrix, output_file=False)
        mfpt[i, 2:] = mfpt_point.flatten()
        print("********************variational bound for its and RMSE********************")
        print(its_vb_rmse[i])
    with open("./fip35_scan_results/its_rmse/"+str(end_point)+"ns_its_variational_bounds_fitting.txt", 'ab') as file:
        np.savetxt(file, its_vb_rmse, fmt='%.8e', delimiter='   ')
    with open("./fip35_scan_results/mfpt/"+str(end_point)+"ns_mfpt_scan.txt", 'ab') as file1:
        np.savetxt(file1, mfpt, fmt='%8e', delimiter=' ')
    end_time = datetime.datetime.now()
    print("The whole calculation time is: %.3f  mins" %((end_time-begin_time).seconds/60))



# Ala2 Macrostates
# =============================================================================
# begin_time = datetime.datetime.now()
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# warnings.filterwarnings('ignore')
# input_data = np.loadtxt('../Dataset/Ala2_Data/ala2-pccap-4states-0.1ps-50ps.txt', dtype=float)
# TPM = np.reshape(input_data, (len(input_data), int(np.sqrt(len(input_data[0]))), int(np.sqrt(len(input_data[0])))))
# igme = IntegralGME(input_len=500, dimension=4, delta_time=0.1)
# igme.get_data(input_data=input_data)
# igme.pre_set_data()
#
# for k in range(71, 91):
#     end_point = k
#     delta_time = 0.1
#     its_vb_rmse = np.zeros((end_point-2, 4))
#     for i in range(0, end_point-2):
#         igme.initial_for_gradient_descent(tau_ini=i, tau_end=end_point, initial_seeds='linear')
#         print("***************fitting_range_"+str(i+1)+"__"+str(end_point)+"*************")
#         A_matrix, T_hat_matrix = igme.gradient_descent_opt(learning_rate=1e-6, row_rate=5, detail_rate=3, momentum=0.6, epochs=1001,
#                                                        output_file=False)
#         eigen_val, eigen_vec = np.linalg.eig(T_hat_matrix)
#         eigen_val = eigen_val.real
#         idx_order = np.argsort(eigen_val)
#         its_vb_rmse[i, 0] = i+1
#         its_vb_rmse[i, 1] = end_point
#         its = -delta_time / np.log(eigen_val[idx_order[-2]])
#         if not np.isinf(its):
#             its_vb_rmse[i, 2] = -delta_time / np.log(eigen_val[idx_order[-2]])
#         else:
#             its_vb_rmse[i, 2] = its_vb_rmse[i-1, 2]
#         its_vb_rmse[i, 3] = squared_difference(TPM=TPM, dimension=4, A_matrix=A_matrix, T_hat_matrix=T_hat_matrix, end_frame=500, tpm_ref=TPM)
#         print("********************variation bound for its and RMSE********************")
#         print(its_vb_rmse[i])
#     with open("./ala2_results/"+str(end_point)+"ns_its_variational_bounds_fitting.txt", 'ab') as file:
#         np.savetxt(file, its_vb_rmse, fmt='%.8e', delimiter='   ')
#     end_time = datetime.datetime.now()
#     print("The whole calculation time is: %.3f  mins" %((end_time-begin_time).seconds/60))
# =============================================================================

