
# # HR Analytics

# This notebook addresses:
# 1. Why are company's best and most experienced employees leaving prematurely?
# 2. Try to predict which valuable employee will leave next.
# 3. Build a system which may assist in staff retention
#
# The HR analytics dataset is obtained from Kaggle [Human Resources Analytics](https://www.kaggle.com/ludobenistant/hr-analytics) dataset.
#
# It shows details for last 5 years and contains the following fields:
#
# * Employee satisfaction level, (range 0 to 1)
# * Last evaluation, (range 0 to 1)
# * Number of projects
# * Average monthly, (in hours)
# * Time spent at the company, (in years)
# * Number of work accident
# * Promotions in the last 5 years
# * Sales (or job function)
# * Salary (low, medium or high)
# * Whether the employee has left the company or not


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import tree
import copy
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from flask import Flask, render_template,request, redirect, url_for
app = Flask(__name__)

@app.route('/')
def main():
    return render_template('index.html')

def preprocess():

    employees = pd.read_csv('HR_comma_sep.csv')
    employees['salary'] = pd.factorize(employees['salary'])[0]
    employees['sales'] = pd.factorize(employees['sales'])[0]
    # To separate label and features in data
    labels = np.where(employees['left'] == 1, 1, 0)
    features = employees.drop('left', axis = 1).as_matrix().astype(np.float)
    label_names = np.unique(labels)
    feature_names = list(employees.axes[1])

    # Splitting dataset into train and test data
    X_train, X_test, Y_train, Y_test = train_test_split(features, labels, test_size=0.1, random_state=0)
    clf = DecisionTreeClassifier(max_depth=5, min_impurity_split=1e-02)
    clf = clf.fit(X_train, Y_train)

    # To separate label and features in data
    X_train_ = X_train[:, 1:]
    X_test_ = X_test[:, 1:]
    y_train_ = X_train[:, 0]
    y_test_ = X_test[:, 0]

    X_ = np.vstack((X_train_, X_test_))
    y_ = np.append(y_train_, y_test_)

    return clf, X_train_, y_train_

def genNewSamples(x, idx, xmin, xmax, num):
    """
    x - actual sample array
    idx - idx of sample to be changed
    (xmin, xmax) - sample to change
    num - number of points to pick from uniform distribution
    """
    x_ = []
    for newx in np.arange(xmin, xmax, (xmax-xmin)*1.0/num):
        y = copy.deepcopy(x)
        y[idx] = newx
        x_.append(y)

    return np.vstack([[x] for x in x_])

# Check how to improve the satisfaction level of employee for those who are leaving
@app.route('/improveSatisfaction', methods=['POST','GET'])
def improveSatisfaction():
    print "inside function"
    print request.method
    if request.method == 'POST':
        X_test_ = []
        X_test_.append(float(request.form['satisfaction_level']))
        X_test_.append(float(request.form['last_evaluation']))
        X_test_.append(float(request.form['number_project']))
        X_test_.append(float(request.form['average_montly_hours']))
        X_test_.append(float(request.form['time_spend_company']))
        X_test_.append(float(request.form['Work_accident']))
        X_test_.append(float(request.form['promotion_last_5years']))
        X_test_.append(float(request.form['department']))
        X_test_.append(float(request.form['salary']))
        print X_test_
        print "\n\n"
        suggestions =[]
        leavingThreshold = 0.9
        # get the probavbilty of leaving
        clf, X_train_, y_train_ = preprocess()
        leaving = clf.predict_proba(X_test_)
        leaving = leaving[0][1]
        print "leaving"
        print leaving
        # if person is leaving, try to improve the satisfaction level
        if leaving > leavingThreshold:
            print '\nThe employee will leave ! - ', X_test_
            # generate new samples by changing
            x_test_rec_ = []
            # number of projects
            x_test_rec_.append(genNewSamples(X_test_, 2, 2, 8, 6))
            # salary
            x_test_rec_.append(genNewSamples(X_test_, 8, 0, 3, 3))
            # number of hours
            '''
            s = 'average_montly_hours'
            xmin = retention_profile_mean[s] - 3.0*retention_profile_std[s]
            xmax = retention_profile_mean[s] + 3.0*retention_profile_std[s]
            x_test_rec_.append(genNewSamples(X_test_, 3, xmin, xmax, 10))
            '''

            # predict the new satisfaction level for updated sample set
            x_test_rec_ = np.vstack(x for x in x_test_rec_)
            regres= RandomForestRegressor()
            regres.fit(X_train_,y_train_)
            y_test_rec_ = regres.predict(x_test_rec_[:,1:])

            # generate new x_test with this predicted satisfaction level and
            # check if the person is leaving
            x_test_rec = np.hstack((np.asarray([y_test_rec_]).T, x_test_rec_[:,1:]))
            y_test_rec= clf.predict_proba(x_test_rec)
            y_test_rec = y_test_rec[:, 0]
            print "\n\n"
            print y_test_rec
            print 'The employee will not leave for following conditions:'
            for idx, y_test in enumerate(y_test_rec):
                if y_test > leavingThreshold:
                    print x_test_rec[idx]
                    if x_test_rec[idx][2]>X_test_[2]:
                        text = "Increase the employee's number of projects by : "
                        text+= str(x_test_rec[idx][2]-X_test_[2])
                        text+= " to get satisfaction level of "+ str(x_test_rec[idx][0]) + "\n"
                        suggestions.append(text)
                    if x_test_rec[idx][8]!=X_test_[8]:
                        text = "Make employee's salary : " + str(x_test_rec[idx][8])
                        text += " to get satisfaction level of "+ str(x_test_rec[idx][0])
                        suggestions.append(text)
        else:
            response = "The employee will not leave! No action required. "
            suggestions.append(response)

        return render_template('index.html', test = X_test_, suggestions = suggestions)

if __name__ == "__main__":
    app.run(host='0.0.0.0')

#test = [   0.34,    0.57,    2,    141,      3,      0,      0,      7,      1  ]

#print improveSatisfaction(test)
