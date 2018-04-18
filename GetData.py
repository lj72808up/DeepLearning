# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
def getData():
    data = pd.read_csv("../data/census.csv")
    income_raw = data['income']
    features_raw = data.drop('income', axis=1)

    # 独热编码,只取收入大于50k作为输出字段
    income = pd.get_dummies(income_raw).iloc[:, 1:]

    # 处理取值范围很大的特征
    skewed = ['capital-gain', 'capital-loss']
    features_raw[skewed] = data[skewed].apply(lambda x: np.log(x + 1))
    features = pd.get_dummies(features_raw)

    # PCA降低独热编码后的特征数量
    # from sklearn.decomposition import PCA
    # pca = PCA(n_components=30)
    # features = pca.fit_transform(features)

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(features, income, test_size=0.2, random_state=0,
                                                        stratify=income)
    # 将'X_train'和'y_train'进一步切分为训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0, stratify=y_train)
    return X_train, X_val, X_test, y_train, y_val, y_test