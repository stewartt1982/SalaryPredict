import numpy as np # linear algebra
from sklearn.model_selection import train_test_split
import pandas as pd # data processing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt







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


    #Read in data sets using pandas.
    #Training, validation and finaly for predictions
    testFeatureFile = pd.read_csv("indeed_data_science_exercise/train_features_2013-03-07.csv")
    testSalariesFile = pd.read_csv("indeed_data_science_exercise/train_salaries_2013-03-07.csv")
    #test file for ue in prediction of the salary
    FeatureFile = pd.read_csv("indeed_data_science_exercise/test_features_2013-03-07.csv") 
    
    #select the training set by using the mask variable
    #Will only roughly split the data 20%/80%
    #as training on 80% of the data would take a prohibitive amount of time
    trainFeature, testFeature, trainSalaries, testSalaries = train_test_split(testFeatureFile,testSalariesFile, test_size=0.65, random_state=42)
    trainFeature = pd.merge(trainFeature,trainSalaries)

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


    #What followings is some terrible data wranging
    #I need more experience with categorical data
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
    #there is a small tail towards higher salaries, but reasonably normal
    #log10 does not change this.
    #we will train to the salaries, not log10 of the salaries
    
    #we are now ready to try analysing this data

####################################
    
    revColumnsTrain = ['companyId','jobType','major','degree',
                       'industry','yearsExperience','milesFromMetropolis']
    revColumnsTarget = ['salary']
    X = trainFeature[revColumnsTrain]
    y = trainSalaries2[revColumnsTarget]
    X_test = testFeature[revColumnsTrain]
    y_test = testSalaries[revColumnsTarget]
    X_testFeatures = FeatureFile[revColumnsTrain]
           
    #n_estimators = 100 seems ok during initial optimistion
    #run over the training data
    rfreg = RandomForestRegressor(n_estimators=100,min_samples_split=8,random_state=1)
    rfreg.fit(X,y.values.ravel())
    MAE_scores=cross_val_score(rfreg,X,y.values.ravel(),scoring='neg_mean_absolute_error')
    print "\nMAE mean over training data\n", np.mean(-MAE_scores)
    
    importances = rfreg.feature_importances_
    print "\nFeature categories\n",revColumnsTrain
    print "\nWhich features are most important\n",importances

    #Using test data predict the salaries
    y_pred = rfreg.predict(X_test)
    y_pred_reshaped = y_pred.reshape(y_pred.shape[0],1)
    
    #Now predict the salaries using the test data without known salaries
    y_testFeatures = rfreg.predict(X_testFeatures)
    
    #men absolute error for the test data
    AE = (y_pred_reshaped - y_test.values)
    n3,bins3,patches3 = plt.hist(AE,25,facecolor='green', alpha=0.75)
    plt.savefig("AE.pdf")

    #run over the test data
    #predict the output salaries

    fobj = open("test_salaries.csv", 'w')
    fobj.write("jobId,salary\n")
    for jobid,salary in zip(FeatureFile['jobId'].values,y_testFeatures.astype(int)):
        fobj.write('{},{}\n'.format(jobid,salary))
        #fobj.write"%s,%i"%(jobid,salary)
    fobj.close()
