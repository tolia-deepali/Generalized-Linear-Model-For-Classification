import time
from matplotlib import pyplot
import random
import sys
import numpy as np


# Read csv files
def parse_document(phi, label):
    phi_arr = np.genfromtxt(phi, delimiter=',')
    label_arr = np.genfromtxt(label)
    return phi_arr, label_arr


# Divide data in training and test set
def data_partition(phi, label):
    index_list = [*range(0, len(phi), 1)]
    random.shuffle(index_list)
    arr_length = round(len(phi) / 3)
    one_third = index_list[0:arr_length]
    two_third = index_list[arr_length:]
    ptest = np.take(phi, one_third, axis=0)
    ptrain = np.take(phi, two_third, axis=0)
    ttrain = np.take(label, two_third, axis=0)
    ttest = np.take(label, one_third, axis=0)
    return ptrain, ttrain, ptest, ttest


# Sigmoid Function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Calculate y value
def y_calc(w_curr, pdata, lv_class):
    if lv_class == 'bayesian':
        return sigmoid(np.inner(w_curr.T, pdata))
    elif lv_class == 'poisson':
        return np.exp(np.inner(w_curr.T, pdata))
    elif lv_class == 'ordinal':
        s = 1
        a =[]
        for i in range(len(pdata)):
            a.append(np.inner(w_curr.T, pdata[i]))
        phi_j = [-2, -1, 0, 1, float("inf")]
        yij = []
        for i in range(len(pdata)):
            temp_row = []
            for j in range(0,5):
                if j != 4:
                    temp_row.append(sigmoid(s * (phi_j[j] - np.array(a[i]))))
                else:
                    temp_row.append(1)
            yij.append(temp_row)
        return np.array(yij)


# Calculate r value
def r_calc(y, tdata, lv_class):
    if lv_class == 'bayesian':
        return y - (y * y)
    elif lv_class == 'poisson':
        return y
    elif lv_class == 'ordinal':
        s =1
        r = []
        for k in range(len(y)):
            col = int(tdata[k])
            r.append(s * s * ((y[k][col-1] * (1 - y[k][col-1])) + (y[k][col-2] * (1 - y[k][col-2]))))
        return np.array(r).T


# Calculate MAP solution wmap
def wmap_calc(alpha,pdata, tdata, lv_class):
    row, col = pdata.shape
    w_curr = np.zeros(col)
    #alpha = 10
    s = 1
    calc1 = np.dot(-alpha, np.identity(col))
    for i in range(100):
        y = y_calc(w_curr, pdata, lv_class)
        if lv_class != 'ordinal':
            r = r_calc(y, tdata, lv_class)
            R = np.diag(r)
            d = np.subtract(tdata, y)
        else:
            r = []
            d = []
            for k in range(len(y)):
                col = int(tdata[k])
                r.append(s * s * ((y[k][col - 1] * (1 - y[k][col - 1])) + (y[k][col - 2] * (1 - y[k][col - 2]))))
                d.append(y[k][col - 1] + y[k][col - 2] - 1)
            R = np.zeros((len(pdata), len(pdata)))
            np.fill_diagonal(R, np.array(r).T)
        calc = np.dot(R, pdata)
        calc2 = np.dot(pdata.T, calc)
        prod1 = np.linalg.inv(np.subtract(calc1, calc2))
        prod2 = np.subtract(np.dot(pdata.T, d), np.dot(alpha, w_curr))
        w_next = np.subtract(w_curr, np.dot(prod1, prod2))
        if i != 0:
            w = np.linalg.norm(w_next - w_curr) / np.linalg.norm(w_curr)
            if w < 0.001:
                return w_next
        w_curr = w_next
    return w_curr


# Calculate ytest
def ytest_calc(wmap, ptest, lv_class):
    if lv_class == 'bayesian':
        return sigmoid(np.inner(wmap.T, ptest))
    elif lv_class == 'poisson':
        return np.exp(np.inner(wmap.T, ptest))
    elif lv_class == 'ordinal':
        a = np.inner(wmap.T, ptest)
        phi_j = [-2, -1, 0, 1, float("inf")]
        yj = []
        s = 1
        for j in range(0,5):
            yj.append(sigmoid(s * (phi_j[j] - a)))
        return np.array(yj).T



# Predicting label values
def prediction(ytest):
    predict = []
    if lv_class == 'bayesian':
        for val in ytest:
            if val < 0.5:
                predict.append(0)
            elif val >= 0.5:
                predict.append(1)
        return predict
    elif lv_class == 'poisson':
        return np.floor(ytest)
    elif lv_class == 'ordinal':
        for j in range(len(ytest)):
            predict.append(ytest[j] - ytest[j - 1])
        return predict


# Bayesian Model Selection
def bms(ptrain, ttrain):
    alpha_new = np.random.randint(1, 100)
    beta_new = np.random.randint(1, 100)
    n, c = ptrain.shape
    ptrain_tr = ptrain.transpose()
    prod1 = ptrain_tr.dot(ptrain)
    l_matrix = beta_new * prod1
    i_dim, c = prod1.shape
    # eigen value of lambda matrix
    l, v = np.linalg.eig(l_matrix)
    for i in range(1, 100):
        alpha_old = alpha_new
        beta_old = beta_new
        # calculate Gamma
        gamma = sum(l / np.add(alpha_old, l))
        # calculate Sn
        sn = np.linalg.inv(np.add((alpha_old * np.identity(i_dim)), (beta_old * prod1)))
        # calculate Mn
        mn = np.dot(beta_old * sn, (ptrain_tr.dot(ttrain)))
        mnt = mn.transpose()
        # calculate alpha
        alpha_new = gamma / np.dot(mnt, mn)
        # calculate beta
        beta_new = (n - gamma) / (sum((np.subtract(ptrain.dot(mnt), ttrain)) ** 2))
        # Convergence condition for alpha beta : 10^-7
        if abs((alpha_new - alpha_old)) < 10 ** -7 and abs((beta_new - beta_old)) < 10 ** -7:
            break
    l_final = alpha_new / beta_new
    return l_final


# Generalised linear model
def GLM(ptrain, ttrain, ptest, ttest, lv_class, lv_alpha):
    arr_length = len(ptrain)
    tarr_length = len(ttrain)
    error_list = []
    mean_list = []
    sd_list = []
    alpha_list = []
    end_time = []
    for i in range(1, 11):
        data_per = i / 10
        pdata = ptrain[0:round(data_per * arr_length)]
        tdata = ttrain[0:round(data_per * tarr_length)]
        mean = np.mean(tdata)
        sd = np.std(tdata)
        mean_list.append(mean)
        sd_list.append(sd)
        #alpha value
        if lv_alpha != 'ten':
            alpha = bms(pdata, tdata)
            alpha_list.append(alpha)
        else:
            alpha = 10
        start_time = time.time()
        wmap = wmap_calc(alpha,pdata, tdata, lv_class)
        end_time.append(time.time() - start_time)
        ytest = ytest_calc(wmap, ptest, lv_class)
        predict = prediction(ytest)
        if lv_class != 'ordinal':
            error = np.abs(np.subtract(predict, ttest))
        else:
            t_hat = np.argmax(predict, axis=1)
            error = np.abs(np.subtract(t_hat, ttest))
        error_list.append(np.mean(error))
    if lv_alpha!='ten':
        return mean_list, error_list,np.average(end_time),np.average(alpha_list)
    else:
        return mean_list, error_list,np.average(end_time),np.average(alpha_list)



# Main Function
if __name__ == "__main__":
    lv_phi = sys.argv[1]
    lv_label = sys.argv[2]
    lv_class = sys.argv[3]
    lv_alpha = sys.argv[4]
    phi, label = parse_document(lv_phi, lv_label)
    row = len(phi)
    phi = np.c_[np.ones(row), phi]
    error_step = []
    sd_step = []
    time_i = []
    a_step = []
    start_time = time.time()
    for i in range(1, 31):
        ptrain, ttrain, ptest, ttest = data_partition(phi, label)
        start_time_i = time.time()
        mean, error_value, wmap_time, a_list = GLM(ptrain, ttrain, ptest, ttest, lv_class,lv_alpha)
        if lv_alpha != 'ten':
            a_step.append(a_list)
        error_step.append(error_value)
        time_i.append((time.time() - start_time_i))
    print("--- %s seconds ---" % (time.time() - start_time))
    print("Mean",np.average(time_i))
    print("Max",np.max(time_i))
    print("Min",np.min(time_i))
    print("Wmap",np.average(wmap_time))
    if lv_alpha != 'ten':
        print("Average Alpha value is:", np.average(a_step))
    np.reshape(error_step, (30,10))
    avg = np.mean(np.array(error_step), axis=0)
    ER = (avg).tolist()
    x = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    SD_ER = np.std(error_step)
    pyplot.errorbar(x, ER, SD_ER, label = lv_phi, ecolor='Red')
    pyplot.xlabel("Data Size")
    pyplot.ylabel("Error Rate")
    pyplot.legend()
    pyplot.show()
    xt = [xr for xr in range(1,31)]
    pyplot.plot(xt, time_i, label='time per iteration'+lv_phi)
    pyplot.xlabel("Iterations")
    pyplot.ylabel("Time")
    pyplot.legend()
    pyplot.show()
