import random
import xlrd
import numpy as np
import math
import matplotlib.pyplot as plt


def get_data(filename):
    """
    从文件中获取正态分布的数据集，并混合
    :param filename:
    :return:
    """
    file = xlrd.open_workbook(filename)
    sheet = file.sheet_by_index(0)
    male = []
    female = []
    for i in range(1, 201):
        male.append(sheet.cell(i, 1).value)
    for j in range(1, 101):
        female.append(sheet.cell(j, 4).value)
    print('male: ', male)
    print('female: ', female)
    people = []
    people.extend(male)
    people.extend(female)
    # print('people: ', people)
    random.seed(10086)
    random.shuffle(people)
    return people


def Normal_data_generator():
    """
    生成正态分布的数据集
    :return:
    """
    male = np.random.normal(175, 15, 200)
    female = np.random.normal(165, 10, 100)
    print('male: ', male)
    print('female: ', female)
    people = []
    people.extend(male)
    people.extend(female)
    random.seed(10086)
    random.shuffle(people)
    return people


def init_parameters(pi, mu1, sig1, mu2, sig2):
    """
    整合参数
    :param pi: 男生比例
    :param mu1: 男生均值
    :param sig1: 男生标准差
    :param mu2: 女生均值
    :param sig2: 女生标准差
    :return: 整合的参数
    """
    parameters = [pi, mu1, sig1, 1 - pi, mu2, sig2]
    return parameters


def cal_expectation_step(data, para):
    """
    计算
    :param data:
    :param para:
    :return:
    """
    N_M = cal_norm_p(data, para[1], para[2])
    N_F = cal_norm_p(data, para[4], para[5])
    # print('N_M:', N_M)
    den = para[0] * N_M + para[3] * N_F
    gamma_M = para[0] * N_M / den
    gamma_F = para[3] * N_F / den
    probability = np.vstack((gamma_M, gamma_F))
    return probability


def cal_norm_p(dat, mu, sig):
    """
    计算数据对应正态分布的概率值
    :param dat:
    :param mu:
    :param sig:
    :return:
    """
    arr = dat - mu
    norm = 1 / (math.sqrt(2 * math.pi) * sig) * np.exp(-arr * arr / (2 * sig * sig))
    return norm


def maximum_step(data, prob):
    """
    估计下一次用于迭代的混合概率和正态概率值
    :param data:
    :param prob:
    :return:
    """
    num = len(prob[0])
    N_mk = 0
    N_fk = 0
    for i in range(num):
        N_mk += prob[0][i]
        N_fk += prob[1][i]
    mu_m = np.dot(prob[0], data) / N_mk
    mu_f = np.dot(prob[1], data) / N_fk
    pi_m = N_mk / num
    pi_f = N_fk / num
    sig_m = np.dot(prob[0], (data - mu_m) * (data - mu_m)) / N_mk
    sig_f = np.dot(prob[1], (data - mu_f) * (data - mu_f)) / N_fk
    parameters = [pi_m, mu_m, np.sqrt(sig_m), pi_f, mu_f, np.sqrt(sig_f)]
    return parameters


def cal_likelihood(data, para):
    """
    计算似然函数值
    :param data:
    :param para:
    :return:
    """
    likelihood = 0
    num = len(data)
    for i in range(num):
        norm_m = 1 / (math.sqrt(2 * math.pi) * para[2]) * np.exp(
            -(data[i] - para[1]) * (data[i] - para[1]) / (2 * para[2] * para[2]))
        norm_f = 1 / (math.sqrt(2 * math.pi) * para[5]) * np.exp(
            -(data[i] - para[1]) * (data[i] - para[4]) / (2 * para[5] * para[5]))
        likelihood += np.log(para[0] * norm_m + para[3] * norm_f)
    return likelihood


def judge_end(before, after):
    error = abs(after - before)
    return error


if __name__ == "__main__":
    # data = get_data('./data/data.xlsx')
    # print('打乱后的: ', data)
    data = Normal_data_generator()
    data = np.array(data)
    para = init_parameters(0.5, 180, 20, 155, 15)
    l_theta = cal_likelihood(data, para)
    print('初始化的参数：', para, '%男生比例，男生均值，男生方差，女生比例，女生均值，女生方差')
    print(0, ':', l_theta)
    i = 0
    acc_theta = []
    while True:
        acc_theta.append(l_theta)
        pre_l = l_theta
        prob = cal_expectation_step(data, para)
        # print(prob[0])
        para = maximum_step(data, prob)
        print('第%s次迭代参数：' % (i + 1), para)
        l_theta = cal_likelihood(data, para)
        print(i + 1, ':', l_theta)
        i = i + 1
        c = judge_end(pre_l, l_theta)
        if c < 0.0001 or i > 2000:
            break
    length = len(acc_theta)
    x = np.linspace(1, length, length, endpoint=True)
    plt.plot(x, acc_theta)
    plt.show()
