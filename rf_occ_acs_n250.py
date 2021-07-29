#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
python script to use nlp tools for occupation autocoding

Created on Wed Jun  2 13:33:11 2021

@author: wilki341
"""

# import modules
import os
import re
import json
import pandas as pd
import numpy as np

from matplotlib import pyplot
from sklearn import preprocessing

from sklearn.model_selection import train_test_split

import string
import nltk

stopwords = nltk.corpus.stopwords.words( 'english' )
wn = nltk.WordNetLemmatizer()
ps = nltk.PorterStemmer()

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer



# set working directory
os.chdir('/projects/users/wilki341')


# import data files
# NOTE: acs is a random sample (with replacemet) so each occ has 500 records
#onet = pd.read_json( 'cenocc_onet.json' )
acs = pd.read_csv( 'acs_occ_sample_n250.csv' )

'''
# make sure 'occ' is character and add leading zeroes
def leadingzeroes( code ):
    if len( code ) == 4:
        return code
    else:
        code = '{0}{1}'.format( '0' * ( 4 - len( code ) ) , code )
        return code

acs['OCC'] = acs['OCC'].apply(lambda x: leadingzeroes(str(x)))
'''

'''
PIPELINE
1. load raw text
2. clean text (e.g., remove stopwords/punct, tokenize, lemmatize)
3. vectorize
4. feature engineering
5. fit model
'''

# [1] LOAD DATA, MERGE WITH EXTERNAL SOURCES, ENGINEER FEATURES
# NOTE: combine onet with write-ins to create large text field for cleaning
# (a) prep onet occ code for mergeing (i.e., make string and add leading zeroes)
'''
onet['cenocc'] = onet['cenocc'].apply(lambda x: leadingzeroes(str(x)))

# (b) combine onet with acs
acsonet = pd.merge( acs ,
                    onet[['cenocc', 'description', 'tasks' ]] ,
                    how = 'left' ,
                    left_on = 'OCC' ,
                    right_on = 'cenocc' )

# (c) check whether any occupations did not match
noonet = acsonet[pd.isnull(acsonet.cenocc)][['OCC','ocw1','ocw2']]
noonet_unique = list( set( noonet['OCC'].tolist() ) )

print( 'number of records with no onet match:' , len( noonet ) ,
       '({0}%)'.format(round(100*float(len(noonet)/len(acs)),1)) )

print( 'number of unique OCC codes with no onet match:' , len( noonet_unique ) )

# (d) create a super string by combining {ocw1, ocw2, description, tasks}
'''
# (e) add in msa codes and population estimates
# (e1) import msa population file
msadf = pd.read_json( 'msapop.json' )

'''
# for fun, plot distribution of city sizes in 2019
pyplot.hist(msadf[msadf.year==2019][['popest']])
pyplot.title('MSA population distribution, 2019')
pyplot.show()
'''

# (e2) drop cbsa_title
msadf.drop(columns='cbsa_title', inplace=True, axis=1)

# (e2) merge msa indicators and population estimates to acsonet
acsdat = pd.merge( acs ,
                   msadf ,
                   how = 'left' ,
                   left_on = ['ST', 'CTY', 'year'] ,
                   right_on = ['stfips', 'ctyfips', 'year'] )

# (e3) drop stfips and ctyfips from file
acsdat.drop(columns=['stfips','ctyfips'], axis=1, inplace=True)

# (e4) define 'urban' as being in a cbsa. check sensibility of assumption
# NOTE: check the rate of missing (null) cbsa values. exepct 80%+/- urban
print( 'percent of rural (non-urban) records:' ,
       round( acsdat['cbsa'].isnull().sum() / len(acsdat) * 100.0 , 1 ) )

acsdat['urban'] = np.where(acsdat['cbsa'].isnull() , 0 , 1 )

# (e5) create msa size categorical variable
# NOTE: non-urban records will be NaN
msasz_bins = [0, 99999, 249999, 499999, 999999,
              2499999, 4999999, msadf.popest.max()]

msasz_categ = pd.cut(acsdat['popest'],
                     bins = msasz_bins ,
                     labels = range(1,len(msasz_bins)))

acsdat.insert(acsdat.shape[-1], 'msasize', msasz_categ)

# set NaN msasize to zero (these are non-urban cases)
acsdat['msasize'] = np.where( acsdat['msasize'].isnull() , 0 ,
                              acsdat['msasize'] )

# (e6) create dummy for puerto rico
# NOTE: we don't use pr msas and pr is a bit unique (e.g., spanish write-ins)
acsdat['pr'] = np.where( acsdat.ST == 72 , 1 , 0 )


# feature engineering
# first, clean up
#del acs


# (f) missing earnings
# NOTE: missing if UWAG.isnull() and USEM.isnull()
acsdat['missearn'] = np.where( acsdat.USEM.isnull() & acsdat.UWAG.isnull() ,
                               1 , 0 )

# (g) due to missing earnings, use indicators for percentile in distribution
# NOTES: paramters: 
#        tabname = dataframe object name (as string), 
#        variable = variable to summarize (as string),
#        quantiles = list of quantiles;
#
#        returns a list of values corresponding to the pth percentile.
#        requies the numpy and pandas packages. returns values for non-missing
#        (i.e., isnull() == False) records 
def getquantiles( tabname , variable , quantiles ):
    if isinstance(quantiles,(list)):
        quant_values = [np.percentile(globals()[tabname][globals()[tabname][variable].isnull() == False][[variable]], x) for x in quantiles]
        return quant_values
    else:
        raise ValueError( 'parameter quantiles must be a list' )

dec_labels = [5]
dec_labels += [x for x in range(10,100,10)]
dec_labels += [95, 99]

dec_values = getquantiles( 'acsdat' , 'UWAG' , dec_labels )

dec_labels += [100]
dec_values += [acsdat['UWAG'].max()]

dec_categ = pd.cut(acsdat['UWAG'],
                   bins = [0] + dec_values ,
                   labels = dec_labels )

dec_categ = dec_categ.cat.add_categories(0).fillna(0)

acsdat.insert(acsdat.shape[-1], 'uwagquantile', dec_categ)


# (h) female binary indicator
acsdat['female'] = np.where( acsdat.AGE == 2, 1, 0)


# (i) education categorical variable
ed_bins = [0, 15, 17, 19, 20, 21, 22, 23, 24]
ed_labels = ['lesshs', 'hsged', 'somecol', 'col2y', 
             'col4y', 'mast', 'prof', 'doct' ]

ed_categ = pd.cut(acsdat['SCHL'],
                  bins = ed_bins ,
                  labels = ed_labels )

acsdat.insert(acsdat.shape[-1], 'edcat', ed_categ)


# (j) indicators for self-employment earnings
# (j1) has non-missing USEM
acsdat['hassem'] = np.where( acsdat['USEM'].isnull() , 0 , 1 )

# (j2) explore the distribution of USEM
acsdat[acsdat.hassem==1][['USEM']].describe()

'''
pyplot.hist(acsdat[(acsdat.hassem==1) & (acsdat.USEM > 0) & (acsdat.USEM <100000)][['USEM']])
pyplot.title('USEM distribution')
pyplot.show()
'''

# (j3) break up USEM into quantiles and create categorical variable
# define a function to return percentiles from a list of integers
decsem_labels = [x for x in range(10, 100, 10)]

decsem_values = getquantiles( 'acsdat' , 'USEM' , decsem_labels )

decsem_labels += [100]
decsem_values += [acsdat['USEM'].max()]

decsem_categ = pd.cut(acsdat['USEM'],
                      bins = [acsdat['USEM'].min()] + decsem_values ,
                      labels = decsem_labels )

decsem_categ = decsem_categ.cat.add_categories(0).fillna(0)

acsdat.insert(acsdat.shape[-1], 'usemquantile', decsem_categ)


# (k) normalized age variable
# NOTE: use MinMaxScaler to normalize
min_max_scaler = preprocessing.MinMaxScaler()

acsdat['minmaxage'] = min_max_scaler.fit_transform(acsdat[['AGE']])

'''
pyplot.hist(acsdat['minmaxage'])
pyplot.title('AGE distribution (MinMaxScaler())')
pyplot.show()
'''

# (l) convert categorical variables to dummies
# store list of feature columns names to be used later
feature_list = ['pr', 'female', 'missearn', 'urban', 'hassem', 'minmaxage']

# (l1) class of worker
cow_vals = pd.DataFrame(acsdat.COW.unique().tolist(), columns = ['COW'])
cow_vals.sort_values(axis=0, by='COW', inplace=True )
cow_vals.reset_index(inplace=True, drop=True)

cow_dum = pd.get_dummies(cow_vals, prefix=['cow'], columns=['COW'])

cow_vals = cow_vals.join(cow_dum)

feature_list += list( cow_dum.columns )

acsdat = acsdat.merge(cow_vals, how='left', on='COW')

# (l2) interview mode
mode_vals = pd.DataFrame(acsdat.MODE.unique().tolist(), columns = ['MODE'])
mode_vals.sort_values(axis=0, by='MODE', inplace=True )
mode_vals.reset_index(inplace=True, drop=True)

mode_dum = pd.get_dummies(mode_vals, prefix=['mode'], columns=['MODE'])

feature_list += list( mode_dum.columns )

mode_vals = mode_vals.join(mode_dum)

acsdat = acsdat.merge(mode_vals, how='left', on='MODE')

# (l3) msasize
msasz_vals = pd.DataFrame(acsdat.msasize.unique().tolist(), columns = ['msasize'])
msasz_vals.sort_values(axis=0, by='msasize', inplace=True )
msasz_vals.reset_index(inplace=True, drop=True)

msasz_vals['msasize'] = msasz_vals['msasize'].apply(lambda x: int(x))

msasz_dum = pd.get_dummies(msasz_vals, prefix=['msasz'], columns=['msasize'])

msasz_vals = msasz_vals.join(msasz_dum)
msasz_vals.drop(columns='msasz_0', inplace=True, axis=1)

feature_list += list( msasz_dum.columns )[1:]

acsdat = acsdat.merge(msasz_vals, how='left', on='msasize')


# (l4) education categories
ed_vals = pd.DataFrame(acsdat.edcat.unique().tolist(), columns = ['edcat'])
ed_vals.sort_values(axis=0, by='edcat', inplace=True )
ed_vals.reset_index(inplace=True, drop=True)

ed_dum = pd.get_dummies(ed_vals, prefix=['ed'], columns=['edcat'])

ed_vals = ed_vals.join(ed_dum)

feature_list += list( ed_dum.columns )

acsdat = acsdat.merge(ed_vals, how='left', on='edcat')


# (l5) wage quantiles
wag_vals = pd.DataFrame(acsdat.uwagquantile.unique().tolist(), columns = ['uwagquantile'])
wag_vals.sort_values(axis=0, by='uwagquantile', inplace=True )
wag_vals.reset_index(inplace=True, drop=True)

wag_dum = pd.get_dummies(wag_vals, prefix=['uwagq'], columns=['uwagquantile'])

wag_vals = wag_vals.join(wag_dum)

wag_vals.drop(columns='uwagq_0', axis=1, inplace=True)

feature_list += list( wag_dum.columns )[1:]

acsdat = acsdat.merge(wag_vals, how='left', on='uwagquantile')


# (l6) self-employment earnings quantiles
sem_vals = pd.DataFrame(acsdat.usemquantile.unique().tolist(), columns = ['usemquantile'])
sem_vals.sort_values(axis=0, by='usemquantile', inplace=True )
sem_vals.reset_index(inplace=True, drop=True)

sem_dum = pd.get_dummies(sem_vals, prefix=['usemq'], columns=['usemquantile'])

sem_vals = sem_vals.join(sem_dum)

sem_vals.drop(columns='usemq_0', axis=1, inplace=True)

feature_list += list( sem_dum.columns )[1:]

acsdat = acsdat.merge(sem_vals, how='left', on='usemquantile')


# [2] clean text, tokenize, lemmatize
# (a) function to clean (remove punct and drop stopwords)s and tokenize
# NOTE: paramter 'lemma' will lemmatize the tokens; default is True
def cleantext( text , lemma = False ):
    text = ''.join([x.lower() for x in text if x not in string.punctuation])
    tokens = re.split( '\W+' , text , flags = re.I )
    text = [x for x in tokens if x not in stopwords]
    
    if lemma:
        text = [wn.lemmatize(word) for word in text]
        
    return text



# [3] vectorize
'''
# (a) count vectorizer
count_vect = CountVectorizer(analyzer=cleantext)
X_counts = count_vect.fit_transform(acsdat['occtext'].apply(lambda x: str(x)))

# (b) n-grams
# NOTE: try with n=2 and n=3
ng2_vect = CountVectorizer(ngram_range=(2,2))
ng3_vect = CountVectorizer(ngram_range=(3,3))

def removepunct( text ):
    text = ''.join([x for x in text if x not in string.punctuation])
    return text

acsonet['occnopunct'] = acsonet['occtext'].apply(lambda x: removepunct(str(x)))

X_ng2 = ng2_vect.fit_transform(acsonet['occnopunct'])
X_ng3 = ng3_vect.fit_transform(acsonet['occnopunct'])

print( 'shape of X_ng2:' X_ng2.shape )
print( 'shape of X_ng3:' X_ng3.shape )
'''


# clean up to save memory
del msasz_categ, ed_categ, decsem_categ, dec_categ



# prep text
# function to create composite text string from multiples fields
def combinetext(ocw1, ocw2):
    text_fields = [x for x in [ocw1, ocw2] if not pd.isnull(x)]
    text = ' '.join( text_fields )
    return text

acsdat['occtext'] = np.vectorize(combinetext)(acsdat.ocw1, acsdat.ocw2)


'''
# (a) n-grams
# NOTE: try with n=(1,3)
ngram_vect = CountVectorizer(ngram_range=(2,2))

X_ngram = ngram_vect.fit_transform(acsdat['occnopunct'])
# (a) count vectorizer
count_vect = CountVectorizer(analyzer=cleantext)

X_count = count_vect.fit_transform(acsdat['occtext'])
'''


# (b) tf-idf
# (b1) clean and lemmatize text

# (b2) create dummy function to allow tfidf vectorizer to use cleaned text
# NOTE: this allows us to use our lemmatized text
def dummy_fun(doc):
    return doc

# (b3) apply tf-idf vectorizer
tfidf_vect = TfidfVectorizer(
    analyzer = 'word' ,
    tokenizer = dummy_fun ,
    preprocessor = dummy_fun ,
    token_pattern = None )

'''
tfidf_vect = TfidfVectorizer(analyzer=cleantext)

X_tfidf = tfidf_vect.fit_transform(acsdat['occtext'])
'''

acsdat['occwi_clean'] = acsdat['occtext'].apply(lambda x: cleantext(str(x)))

X_tfidf = tfidf_vect.fit_transform(acsdat['occwi_clean'])

# [3] create analysis data set(s)
y_labels = acsdat['OCC']

X_feat_tfidf = pd.concat([acsdat[feature_list], 
                          pd.DataFrame(X_tfidf.toarray())], 
                         axis=1)

'''
X_feat_count = pd.concat([acsdat[feature_list], 
                          pd.DataFrame(X_count.toarray())], 
                         axis=1)
'''

#del acsdat

# (3a) split data into training, test, and validation files
X_train, X_test, y_train, y_test = train_test_split(X_feat_tfidf ,
                                                    y_labels ,
                                                    test_size=0.4 ,
                                                    random_state=123 )

X_val, X_test, y_val, y_test = train_test_split(X_test ,
                                                y_test ,
                                                test_size=0.5 ,
                                                random_state=123 )

'''
# (3b) save data to disk for later use
for i in ['X', 'y']:
    for j in ['train', 'test']:
        tabname = '{0}_{1}'.format( i , j )
        globals()[tabname].to_csv('{}.csv'.format(tabname), index=False)

del i, j
'''
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

def print_results( results ):
    print( 'BEST PARAMS:{}\n'.format(results.best_params_))
    
    means = results.cv_results_['mean_test_score']
    stds = results.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, results.cv_results_['params']):
        print('{} (+/-{}) for {}'.format(round(mean, 3),
                                         round(std*2, 3),
                                         params))

'''
# basic randomforest
rf = RandomForestClassifier(n_estimators=500,
                            max_depth=20, 
                            n_jobs=-1)
rf_model = rf.fit(X_train, y_train.values.ravel())

y_pred_test = rf_model.predict(X_test)
y_pred_val = rf_model.predict(X_val)

probs_test = rf_model.predict_proba(X_test)
probs_val = rf_model.predict_proba(X_val)

y_probs_test = [max(x) for x in probs_test]
y_probs_val = [max(x) for x in probs_val]
'''
from sklearn.metrics import accuracy_score
'''
accuracy_score(y_test,y_pred_test)
accuracy_score(y_val,y_pred_val)


# naive bayes
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
gnb_model = gnb.fit(X_train, y_train)

y_pred_gnb_test = gnb_model.predict(X_test)
y_pred_gnb_val = gnb_model.predict(X_val)

probs_gnb_test = gnb_model.predict_proba(X_test)
probs_gnb_val = gnb_model.predict_proba(X_val)

y_probs_gnb_test = [max(x) for x in probs_gnb_test]
y_probs_gnb_val = [max(x) for x in probs_gnb_val]

accuracy_score(y_test,y_pred_gnb_test)
accuracy_score(y_val,y_pred_gnb_val)

# kneighbors classifier
from sklearn.neighbors import KNeighborsClassifier

knc = KNeighborsClassifier(n_jobs=-1)
knc_model = knc.fit(X_train, y_train)
y_pred_knc = knc_model.predict(X_test)
'''

# gridsearchcv
# (a) random forest
rf = RandomForestClassifier()

parameters = {
    'n_estimators' : [20, 600, 900] ,
    'max_depth' : [5, 25, 50, 100]
    }

cv = GridSearchCV(rf, parameters)
cv.fit(X_train, y_train.values.ravel())

print( 'cv results for RandomForestClassifier:\n' )
print_results(cv)

joblib.dump(cv.best_estimator_, 'mdl_occ_best_rf_tfidf_lemma.pkl')

# (b) gradient boosting
# NOTE: use same paramters as random forest
'''
gb = GradientBoostingClassifier()

cv = GridSearchCV(gb, parameters)
cv.fit(X_train, y_train.values.ravel())

print( 'cv results for GradientBoostingClassifier:\n' )
print_results(cv)

joblib.dump(cv.best_estimator_, 'mdl_occ_best_gb_tfidf_lemma.pkl')
'''

'''
estimator = best_rf.estimators_[0]

from sklearn.tree import export_graphviz

export_graphviz(
    estimator ,
    out_file = 'best_rf_tree0.dot' ,
    feature_names = X_train.columns ,
    rounded = True , precision = 2 , proportion = False, filled = True
    )

from subprocess import call
'''

call(['dot', '-Ttif' , 'best_rf_tree0.dot', '-o', 'best_rf_tree0.tif', '-Gdpi=600'])