import numpy as np # linear algebra
from sklearn.model_selection import train_test_split
import pandas as pd # data processing
import sklearn as sk #data sets
from scipy.sparse import csc_matrix
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import math






if __name__ == "__main__":
    #Code first needs to red in the training data sets.
    #There are 2 training sets of 1000000 lines
    #File 1) indeed_data_science_exercise/train_features_2013-03-07.csv
    #giving the fetures we will use to estimate the salary
    #File 2) indeed_data_science_exercise/train_salary_2013-03-07.csv
    #giving the salaries for each position, linked to the first file by
    #jobId
    #
    #Of the test set the first thing we will do is divide it into 2 sets
    #1) 75% of the data for training
    #2) 25% for testing the performance of our algorithm
    #
    #We can then run our trained algorithm over the
    #indeed_data_science_exercise/test_features_2013-03-07.csv
    #nd predict the salaries
    #


    #Read in data sets using pandas
    testFeatureFile = pd.read_csv("indeed_data_science_exercise/train_features_2013-03-07.csv")
    testSalariesFile = pd.read_csv("indeed_data_science_exercise/train_salaries_2013-03-07.csv")
    #test file
    FeatureFile = pd.read_csv("indeed_data_science_exercise/test_features_2013-03-07.csv") 
    
    #select the training set by using the mask variable
    #Will only roughly split the data 20%/80%
    #as training on 80% of the data would take a prohibitive amount of time
    trainFeature, testFeature, trainSalaries, testSalaries = train_test_split(testFeatureFile,testSalariesFile, test_size=0.8, random_state=42)
    trainFeature = pd.merge(trainFeature,trainSalaries)
    #trainFeature = trainFeature[trainFeature.astype(str).ne('NONE').all(1)]


    #What fields do we have?
    #jobId,companyId,jobType,degree,major,industry,yearsExperience,
    #milesFromMetropolis
    #
    #not used for classification
    #jobId is basically just an integer and not used for the classification
    #
    #Used for classification
    #companyId is also basically an integer
    #jobType text
    #degree text
    #major text
    #industry text
    #yearsExperience integer
    #milesFromMetropolis integer

    #Text based columns get a vectorizer
    #The text in these columns are very simple
    #One word or two words separatd by an _
    #Let''s just make the ngram range just 1 word
    #The vocbulary is small as well

    #print trainFeature["jobType"] != "None" #or trainFeature["major"] != "None" or trainFeature["industry"] != "None"
    trainFeature["jobType"] = trainFeature["jobType"].astype('category')
    trainFeature["degree"] = trainFeature["degree"].astype('category') 
    trainFeature["major"] = trainFeature["major"].astype('category')
    trainFeature["industry"] = trainFeature["industry"].astype('category')
    trainFeature["jobType"] = trainFeature["jobType"].cat.codes
    trainFeature["degree"] = trainFeature["degree"].cat.codes
    trainFeature["major"] = trainFeature["major"].cat.codes
    trainFeature["industry"] = trainFeature["industry"].cat.codes
    testFeature["jobType"] = testFeature["jobType"].astype('category')
    testFeature["degree"] = testFeature["degree"].astype('category') 
    testFeature["major"] = testFeature["major"].astype('category')
    testFeature["industry"] = testFeature["industry"].astype('category')
    testFeature["jobType"] = testFeature["jobType"].cat.codes
    testFeature["degree"] = testFeature["degree"].cat.codes
    testFeature["major"] = testFeature["major"].cat.codes
    testFeature["industry"] = testFeature["industry"].cat.codes
   
    #test data
    FeatureFile["jobType"] = FeatureFile["jobType"].astype('category')
    FeatureFile["degree"] = FeatureFile["degree"].astype('category') 
    FeatureFile["major"] = FeatureFile["major"].astype('category')
    FeatureFile["industry"] = FeatureFile["industry"].astype('category')
    FeatureFile["jobType"] = FeatureFile["jobType"].cat.codes
    FeatureFile["degree"] = FeatureFile["degree"].cat.codes
    FeatureFile["major"] = FeatureFile["major"].cat.codes
    FeatureFile["industry"] = FeatureFile["industry"].cat.codes
    
    companyIdInt = trainFeature["companyId"].str.extract('(\d+)').astype(int)
    #replce the Text based company ID with a numberic column
    trainFeature['companyId'] = companyIdInt
    companyIdInt = testFeature["companyId"].str.extract('(\d+)').astype(int)
    #replce the Text based company ID with a numberic column
    testFeature['companyId'] = companyIdInt


    #test data
    companyIdInt = FeatureFile["companyId"].str.extract('(\d+)').astype(int)
    FeatureFile['companyId'] = companyIdInt
    
    #Get the salaries out of the features dataFrame
    trainSalaries2 = trainFeature.pop('salary').to_frame()
    #plot salaries and log10(salaries)
    #which is more normal?
    #some basic stats
    print "Let's get some basic stats first"
    print trainFeature.describe()
    print trainSalaries.describe()
    n1, bins1, patches1 = plt.hist(trainSalaries['salary'], 25, normed=1,
                                facecolor='green', alpha=0.75)
    plt.savefig("salary.pdf")
    #there is a small tail towards higher salaries, but reasonably normal
    
    #Regression Trees expect the targetdata to be normally distributed
    #trainSalaries2['salary'].apply(lambda x: math.log10(x))
    #n2, bins2, patches2 = plt.hist(trainSalaries['salary'], 25, normed=1, facecolor='green', alpha=0.75)
    #there is a small tail towards higher salaries, but reasonably normal
    #log10 does not change this.

    #we will train to the salaries, not log10 of the salaries
    #plt.show()
    
    #we are now ready to try analysing this data
    #As a first test we will leave the 'None' in as a category
    #For education level this is likely a good catgory, but will need
    #to thnk about others


####################################
    
    revColumnsTrain = ['companyId','jobType','major','degree',
                  'industry','yearsExperience','milesFromMetropolis']
    revColumnsTarget = ['salary']
    X = trainFeature[revColumnsTrain]
    y = trainSalaries2[revColumnsTarget]
    X_test = testFeature[revColumnsTrain]
    y_test = testSalaries[revColumnsTarget]
    X_testFeatures = FeatureFile[revColumnsTrain]

    #some code to tune parameters for the RandomForestRegressor

    # estimator = RandomForestRegressor()
    # param_grid = {
    #     "n_estimators"      : [10,50,100,500],
    #     "min_samples_split" : [2,4,8]
    # }
    # for i in range(3,8):
    #     print "For the k = ",i," best features:"
    #     X_new = SelectKBest(f_regression, k=i).fit_transform(X, y.values.ravel())     
    #     grid = GridSearchCV(estimator, param_grid, verbose=1, n_jobs=-1, cv=5)
    #     grid.fit(X_new, y.values.ravel())
    #     print "Best Score ",grid.best_score_
    #     print "Best Param. ",grid.best_params_
    

    #initial test with varying n_estimators
    #minimising the absolute error

    #let's define a range of estimators
    #     estimator_range = range(5,2000,100)
    #     RMSE_scores = []
    #     print "X_new ",X_new
    #     print trainFeature
    #     #print y.values.ravel()
    #     for estimator in estimator_range:
    #        print "Estimators = ",estimator
    #        rf = RandomForestRegressor(n_estimators=estimator,random_state=42)    
    #        MSE_scores=cross_val_score(rf,X_new,y.values.ravel(),scoring='neg_mean_absolute_error')
    #        #RMSE_scores.append(np.mean(np.sqrt(-MSE_scores)))
    #        print MSE_scores 
    #        RMSE_scores.append(np.mean(-MSE_scores))
    #        print "Estimator ",estimator," RMSE ",RMSE_scores[-1]
    
       
    #100 seems ok
    #run over the training data
    rfreg = RandomForestRegressor(n_estimators=100,min_samples_split=8,random_state=1)
    rfreg.fit(X,y.values.ravel())
    MSE_scores=cross_val_score(rfreg,X,y.values.ravel(),scoring='neg_mean_absolute_error')
    print MSE_scores
    
    importances = rfreg.feature_importances_
    print revColumnsTrain
    print importances
    
    y_pred = rfreg.predict(X_test)

    y_testFeatures = rfreg.predict(X_testFeatures)
    print "Test Data predictions\n",y_testFeatures
    
    y_pred_reshaped = y_pred.reshape(y_pred.shape[0],1)
    print y_pred_reshaped,y_test.values
    #men absolute error
    AE = (y_pred_reshaped - y_test.values)
    n3,bins3,patches3 = plt.hist(AE,25,facecolor='green', alpha=0.75)
    plt.savefig("AE2.pdf")
    print np.argwhere(np.isinf(y_pred_reshaped))
    print np.argwhere(np.isinf(y_test.values))
    #relE = (y_pred_reshaped - y_test.values)/y_test.values
    relE = np.divide(y_pred_reshaped - y_test.values,y_test.values,where=y_test.values!=0)
    print np.argwhere(np.isinf(relE))
    print 'relE ',relE,relE.shape,y_test.values.shape
    relE2 = relE[~np.isinf(relE)]
    relE3 =  relE2[~np.isnan(relE2)]
    #n4,bins4,patches4 = plt.hist(relE3,25,(-2,2),facecolor='green', alpha=0.75)
    n4,bins4,patches4 = plt.hist(relE3,facecolor='green', alpha=0.75)
    print "hist out ",n4,bins4,patches4
    plt.savefig("relE.pdf")
    #plt.show()
    print AE


    #run over the test data
    #predict the output salaries

    fobj = open("test_salaries.csv", 'w')
    fobj.write("jobId,salary\n")
    for jobid,salary in zip(FeatureFile['jobId'].values,y_testFeatures.astype(int)):
        fobj.write('{},{}\n'.format(jobid,salary))
        #fobj.write"%s,%i"%(jobid,salary)
    fobj.close()
