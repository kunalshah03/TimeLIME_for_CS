from othertools import *
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
# from XTREE import XTREE
import random
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings("ignore")

def TL(name,par,rules,smote=False,act=False):
    start_time = time.time()
    files = [name[0], name[1], name[2]]
    freq = [0] * 18
    deltas = []
    for j in range(0, len(files) - 2):
        df1 = prepareData(files[j])
        df2 = prepareData(files[j + 1])
        for i in range(1, 19):
            col1 = df1.iloc[:, i]
            col2 = df2.iloc[:, i]
            deltas.append(hedge(col1, col2))
    deltas = sorted(range(len(deltas)), key=lambda k: deltas[k], reverse=True)

    actionable = []
    for i in range(0, len(deltas)):
        if i in deltas[0:5]:
            actionable.append(1)
        else:
            actionable.append(0)
    print("actionable")
    print(actionable)
    df1 = prepareData(name[0])
    df2 = prepareData(name[1])
    df3 = prepareData(name[2])
    bug1 = bugs(name[0])
    bug2 = bugs(name[1])
    bug3 = bugs(name[2])

    df11 = df1.iloc[:, 1:]
    df22 = df2.iloc[:, 1:]
    df33 = df3.iloc[:, 1:]

    df1n = norm(df11, df11)
    df2n = norm(df22, df22)
    df3n = norm(df22, df33)

    X_train1 = df1n.iloc[:, :-1]
    y_train1 = df1n.iloc[:, -1]
    X_test1 = df2n.iloc[:, :-1]
    y_test1 = df2n.iloc[:, -1]
    X_test2 = df3n.iloc[:, :-1]
    y_test2 = df3n.iloc[:, -1]

    score = []
    bugchange = []
    size = []
    score2 = []
    records = []
    matrix = []
    seen = []
    seen_id = []
    par = 18
    clf1 = RandomForestRegressor()
    if smote:
        sm = SMOTE()
        X_train1_s, y_train1_s = X_train1, y_train1
        clf1.fit(X_train1_s, y_train1_s)
        explainer = lime.lime_tabular.LimeTabularExplainer(training_data=pd.DataFrame(X_train1_s).values, training_labels=y_train1_s,
                                                           feature_names=df11.columns,
                                                           discretizer='entropy', feature_selection='lasso_path',
                                                           mode='regression')
    else:
        clf1.fit(X_train1, y_train1)

        explainer = lime.lime_tabular.LimeTabularExplainer(training_data=X_train1.values, training_labels=y_train1,
                                                           feature_names=df11.columns,
                                                           discretizer='entropy', feature_selection='lasso_path',
                                                           mode='regression')


    for i in range(0, len(y_test1)):
        for j in range(0, len(y_test2)):
            actual = X_test2.values[j]
            if True:
                ins = explainer.explain_instance(data_row=X_test1.values[i], predict_fn=clf1.predict,
                                                    num_features=18,
                                                    num_samples=5000)
                ind = ins.local_exp[1]
                temp = X_test1.values[i].copy()
                if act:
                    tem, plan, rec = flip(temp, ins.as_list(), ind, clf1, df1n.columns, par,
                                            actionable=actionable)
                else:
                    tem, plan, rec = flip(temp, ins.as_list(), ind, clf1, df1n.columns, par, actionable=None)
                if act:
                    if rec in seen_id:
                        supported_plan_id = seen[seen_id.index(rec)]
                        print("Seen.")
                    else:
                        supported_plan_id = find_supported_plan(rec, rules, top=5)
                        seen_id.append(rec.copy())
                        seen.append(supported_plan_id)
                        print("Not seen.", rec)
                        print("seen_id", seen_id)

                    for k in range(len(rec)):
                        if rec[k] != 0:
                            if (k not in supported_plan_id) and ((0 - k) not in supported_plan_id):
                                plan[k][0], plan[k][1] = tem[k] - 0.05, tem[k] + 0.05
                                rec[k] = 0
                score.append(overlap(plan, actual))
                size.append(size_interval(plan))
                score2.append(len([n for n in rec if n != 0]))
                records.append(rec)
                tp, tn, fp, fn = abcd(temp, plan, actual, rec)
                print("")
                matrix.append([tp, tn, fp, fn])

                bugchange.append(bug3[j] - bug2[i])  # negative if reduced #bugs, positive if added
            break
    print("Runtime:", time.time() - start_time)
    print(name[0], par)
    print('>>>')
    print('>>>')
    print('>>>')
    return score, bugchange, size, score2, records, matrix


def historical_logs(name, par, explainer=None, smote=False, small=0.05, act=False):
    start_time = time.time()
    files = [name[0], name[1], name[2]]
    freq = [0] * 18
    deltas = []
    for j in range(0, len(files) - 2):
        df1 = prepareData(files[j])
        df2 = prepareData(files[j + 1])
        for i in range(1, 19):
            col1 = df1.iloc[:, i]
            col2 = df2.iloc[:, i]
            deltas.append(hedge(col1, col2))
    deltas = sorted(range(len(deltas)), key=lambda k: deltas[k], reverse=True)

    actionable = []
    for i in range(0, len(deltas)):
        if i in deltas[0:5]:
            actionable.append(1)
        else:
            actionable.append(0)
    print("actionable")
    print(actionable)
    df1 = prepareData(name[0])
    df2 = prepareData(name[1])
    df3 = prepareData(name[2])
    bug1 = bugs(name[0])
    bug2 = bugs(name[1])
    bug3 = bugs(name[2])
    df11 = df1.iloc[:, 1:]
    df22 = df2.iloc[:, 1:]
    df33 = df3.iloc[:, 1:]

    df1n = norm(df11, df11)
    df2n = norm(df22, df22)
    df3n = norm(df22, df33)

    X_train1 = df1n.iloc[:, :-1]
    y_train1 = df1n.iloc[:, -1]
    X_test1 = df2n.iloc[:, :-1]
    y_test1 = df2n.iloc[:, -1]
    X_test2 = df3n.iloc[:, :-1]
    y_test2 = df3n.iloc[:, -1]

    old_change = []
    new_change = []
    par = 0
    clf1 = RandomForestRegressor()
    if smote:
        sm = SMOTE()

        X_train1_s, y_train1_s = X_train1, y_train1
        clf1.fit(X_train1_s, y_train1_s)
        explainer = lime.lime_tabular.LimeTabularExplainer(training_data=pd.DataFrame(X_train1_s).values, training_labels=y_train1_s,
                                                           feature_names=df11.columns,
                                                           discretizer='entropy', feature_selection='lasso_path',
                                                           mode='regression', sample_around_instance=True)
    else:
        clf1.fit(X_train1, y_train1)
        explainer = lime.lime_tabular.LimeTabularExplainer(training_data=X_train1.values, training_labels=y_train1,
                                                           feature_names=df11.columns,
                                                           discretizer='entropy', feature_selection='lasso_path',
                                                           mode='regression', sample_around_instance=True)
    for i in range(0, len(y_test1)):
        for j in range(0, len(y_test2)):
            if True:
                actual = X_test2.values[j]
                ins = explainer.explain_instance(data_row=X_test1.values[i], predict_fn=clf1.predict,
                                                 num_features=18,
                                                 num_samples=5000)
                ind = ins.local_exp[1]
                temp = X_test1.values[i].copy()
                if act:
                    tem, plan, rec = flip(temp, ins.as_list(), ind, clf1, df1n.columns, 0, actionable=actionable)
                else:
                    tem, plan, rec = flip(temp, ins.as_list(), ind, clf1, df1n.columns, 0, actionable=None)
                o = track1(plan[:-1], temp)
                n = track1(plan[:-1], actual)
                old_change.append(o)
                new_change.append(n)

                break
    print("Runtime:", time.time() - start_time)
    print(name[0], par)
    print('>>>')
    print('>>>')
    print('>>>')
    return old_change, new_change
