
"""
import time
start_time = time.time()
print time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(start_time))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize, ngrams
import xgboost as xgb
from sklearn.metrics import log_loss
from collections import Counter
from xgboost import plot_importance
from collections import defaultdict


#load data
train_df = pd.read_csv('new_train.csv')
test_df = pd.read_csv('new_test.csv')

####### data processing: replace abbreviations
## global variable
eng_stopwords = set(stopwords.words('english'))
stops = eng_stopwords
eps = 5000
weights = {}

#Magic Feature
ques = pd.concat([train_df[['question1','question2']], test_df[['question1','question2']]], axis=0).reset_index(drop='index')
q_dict = defaultdict(set)
for i in range(ques.shape[0]):
    q_dict[ques.question1[i]].add(ques.question2[i])
    q_dict[ques.question2[i]].add(ques.question1[i])

print('done process data')



####### featuring 
# word share
def normalized_word_share(row):
        w1 = set(map(lambda word: word.lower().strip(), str(row['question1']).split(" ")))
        w2 = set(map(lambda word: word.lower().strip(), str(row['question2']).split(" ")))    
        return float(len(w1 & w2))/(len(w1) + len(w2)) ######### for python 2.X

# tfidf word share
def get_weight(count, eps=10000, min_count=1):
    if count < min_count:
        return 0.0
    else:
        return 1.0 / (count + eps)

def tfidf_word_share(row, weights):
    out_list = []
    q1words_stops = {}
    q2words_stops = {}

     # tfidf_word_share_stops
    for word in str(row['question1']).lower().split():
        if word not in stops:
            q1words_stops[word] = 1
    for word in str(row['question2']).lower().split():
        if word not in stops:
            q2words_stops[word] = 1
    if len(q1words_stops) == 0 or len(q2words_stops) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        out_list.extend([0])
    else:
        shared_weights_stops = [weights.get(w, 0) for w in q1words_stops.keys() if w in q2words_stops] + [weights.get(w, 0) for w in q2words_stops.keys() if w in q1words_stops]
        total_weights_stops = [weights.get(w, 0) for w in q1words_stops] + [weights.get(w, 0) for w in q2words_stops]
        R_stops = np.sum(shared_weights_stops) / np.sum(total_weights_stops)
        out_list.extend([R_stops])
                     
    return out_list

#stop word ratio: stops / lens
def stop_ratio(row,flag=1):
    if flag == 1:
        ques = set(str(row['question1']).lower().split())
    else:
        ques = set(str(row['question2']).lower().split())
    
    if len(ques) == 0:
        return 0
    
    qstops = ques.intersection(stops)
    return float(len(qstops))/len(ques)


def grams_select(row):
    que1 = str(row['question1'])
    que2 = str(row['question2'])
    out_list = []
    # get unigram features #
    unigrams_que1 = [word for word in que1.lower().split() if word not in eng_stopwords]
    unigrams_que2 = [word for word in que2.lower().split() if word not in eng_stopwords]
    common_unigrams_len = len(set(unigrams_que1).intersection(set(unigrams_que2)))
    common_unigrams_ratio = float(common_unigrams_len) / max(len(set(unigrams_que1).union(set(unigrams_que2))),1)
    diff_unigrams = abs(len(set(unigrams_que1)) - len(set(unigrams_que2)))
    out_list.extend([common_unigrams_len, common_unigrams_ratio])

    # get bigram features #
    bigrams_que1 = [i for i in ngrams(unigrams_que1, 2)]
    bigrams_que2 = [i for i in ngrams(unigrams_que2, 2)]
    common_bigrams_len = len(set(bigrams_que1).intersection(set(bigrams_que2)))
    common_bigrams_ratio = float(common_bigrams_len) / max(len(set(bigrams_que1).union(set(bigrams_que2))),1)
    diff_bigrams = abs(len(set(bigrams_que1)) - len(set(bigrams_que2)))
    out_list.extend([common_bigrams_len, common_bigrams_ratio])

    # get trigram features #
    trigrams_que1 = [i for i in ngrams(unigrams_que1, 3)]
    trigrams_que2 = [i for i in ngrams(unigrams_que2, 3)]
    common_trigrams_len = len(set(trigrams_que1).intersection(set(trigrams_que2)))
    common_trigrams_ratio = float(common_trigrams_len) / max(len(set(trigrams_que1).union(set(trigrams_que2))),1)
    diff_trigrams = abs(len(set(trigrams_que1)) - len(set(trigrams_que2)))
    out_list.extend([common_trigrams_len, common_trigrams_ratio])
    
    #get fourgram features
    fourgrams_que1 = [i for i in ngrams(unigrams_que1, 4)]
    fourgrams_que2 = [i for i in ngrams(unigrams_que2, 4)]
    common_fourgrams_len = len(set(fourgrams_que1).intersection(set(fourgrams_que2)))
    common_fourgrams_ratio = float( common_fourgrams_len ) / max(len(set(fourgrams_que1).union(set(fourgrams_que2))),1)
    diff_fourgrams = abs(len(set(fourgrams_que1)) - len(set(fourgrams_que2)))
    out_list.extend([common_fourgrams_len,common_fourgrams_ratio])
    
    #compare jaccard
    q1_list = que1.lower().split()
    q2_list = que2.lower().split()
    q_inter = set(q1_list).intersection(set(q2_list))
    q_union = set(q1_list).union(set(q2_list))
    if len(q_union) == 0:
        q_union = [1]
    jaccard = float(len(q_inter))/len(q_union)
    out_list.extend([jaccard])
    
    #same start word 1
    if(q1_list[0] == q2_list[0]):
        out_list.extend([1])
    else:
        out_list.extend([0])
    
    #same start word 2
    if(len(q1_list) >= 2 and len(q2_list) >= 2):
        if(q1_list[0] == q2_list[0] and q1_list[1] == q2_list[1]):
            out_list.extend([1])
        else:
            out_list.extend([0])
    else:
        out_list.extend([0])
    
    #same start word 3
    if(len(q1_list) >=3 and len(q2_list) >=3):
        if(q1_list[0] == q2_list[0] and q1_list[1] == q2_list[1] and q1_list[2] == q2_list[2]):
            out_list.extend([1])
        else:
            out_list.extend([0])
    else:
        out_list.extend([0])

    #Stop ratio
    q1_stop = set(q1_list).intersection(stops)
    q2_stop = set(q2_list).intersection(stops)
    
    #q1_stop ratio
    if len(q1_stop) == 0:
        q1_stop_ratio = 0
    else:
        q1_stop_ratio = float(len(q1_stop))/len(q1_list)
    out_list.extend([q1_stop_ratio])
    
    #q2_stop ratio
    if len(q2_stop) == 0:
        q2_stop_ratio = 0
    else:
        q2_stop_ratio = float(len(q2_stop))/len(q2_list)
    out_list.extend([q2_stop_ratio])
    
    #diff_stops_ratio
    diff_stop_ratio = q1_stop_ratio - q2_stop_ratio
    out_list.extend([diff_stop_ratio])
    
    #Stops jaccard
    q_stops_inter = q1_stop.intersection(q2_stop)
    q_stops_union = q1_stop.union(q2_stop)
    if len(q_stops_union) == 0:
        q_stops_union = [1]
    stops_jaccard = float(len(q_stops_inter))/len(q_stops_union)
    out_list.extend([stops_jaccard])
    
    #Average word
    q1_len = len(str(row['question1']).replace(' ', ''))
    if q1_len == 0:
        q1_ave_world = 0.0
    else:
        q1_ave_world = float(len(q1_list))/q1_len
    q2_len = len(str(row['question2']).replace(' ',''))
    if q2_len == 0:
        q2_ave_world = 0.0
    else:
        q2_ave_world = float(len(q2_list))/q2_len
    diff_ave_world = q1_ave_world - q2_ave_world
    out_list.extend([q1_ave_world,q2_ave_world,diff_ave_world])
    
    #Shared two grams
    q1_2gram = set([i for i in zip(q1_list, q1_list[1:])])
    q2_2gram = set([i for i in zip(q2_list, q2_list[1:])])
    shared_2gram = q1_2gram.intersection(q2_2gram)
    if len(q1_2gram) + len(q2_2gram) == 0:
        shared_2gram_ratio = 0
    else:
        shared_2gram_ratio = float(len(shared_2gram))/(len(q1_2gram) + len(q2_2gram))
        
    out_list.extend([len(shared_2gram), shared_2gram_ratio])
    
    ####Magic feature
    q1_freq = len(q_dict[row['question1']])
    q2_freq = len(q_dict[row['question2']])
    q1_q2_intersect = len(set(q_dict[row['question1']]).intersection(set(q_dict[row['question2']])))
    out_list.extend([q1_freq, q2_freq, q1_q2_intersect])
    
    ####diff char len/ diff world len
    diff_char_len = abs(q1_len - q2_len)
    diff_word_len = abs(len(q1_list) - len(q2_list))
    diff_2grams_len = abs(len(q1_2gram) - len(q2_2gram))
    diff_stops = abs(len(q1_stop) - len(q2_stop))
    out_list.extend([diff_char_len, diff_word_len, diff_2grams_len, diff_unigrams, diff_bigrams, diff_trigrams, diff_fourgrams, diff_stops])
    
    ####total uni words/total uni words stop
    total_uni_words = len(set(q1_list).union(q2_list))
    total_uni_stops = len([ x for x in set(q1_list).union(q2_list) if x not in stops ])
    out_list.extend([total_uni_words, total_uni_stops])
    
    
    return out_list

def feature_selection(train_df): 

    # question length
    train_df['q1len'] = train_df['question1'].str.len()
    train_df['q2len'] = train_df['question2'].str.len()
    
    # number of words
    train_df['q1_words'] = train_df['question1'].apply(lambda x: len(str(x).split(" ")))
    train_df['q2_words'] = train_df['question2'].apply(lambda x: len(str(x).split(" ")))
    
    # word share
    train_df['word_share'] = train_df.apply(normalized_word_share, axis=1)
    
    print('done nouns')
    
    # tfidf word share
    train_qs = pd.Series(train_df['question1'].tolist() + train_df['question2'].tolist()).astype(str)
     
    words = (" ".join(train_qs)).lower().split()
    counts = Counter(words)
    weights = {word: get_weight(count) for word, count in counts.items()}
    
    #train_df['tfidf_word_share'] = train_df.apply(lambda x: tfidf_word_share(x, weights), axis=1, raw=True)
    weights_features = np.vstack( np.array(train_df.apply(lambda x: tfidf_word_share(x, weights), axis=1)) )
    weights_features = pd.DataFrame(weights_features)
    train_df['tfidf_word_share_stops'] = weights_features[[0]]
    
    print('tfidf word share done')
    
    gram_feature = np.vstack( np.array(train_df.apply(lambda row: grams_select(row), axis=1)) ) 
    grams = pd.DataFrame(gram_feature)
    train_df['unigrams_common_count'] = grams[[0]]
    train_df['unigrams_common_ratio'] = grams[[1]]
    train_df['bigrams_common_count'] = grams[[2]]
    train_df['bigrams_common_ratio'] = grams[[3]]
    train_df['trigrams_common_count'] = grams[[4]]
    train_df['trigrams_common_ratio'] = grams[[5]]
    train_df['fourgrams_common_count'] = grams[[6]]
    train_df['fourgrams_common_ratio'] = grams[[7]]
    train_df['jaccard'] = grams[[8]]
    train_df['same_start_word1'] = grams[[9]]
    train_df['same_start_word2'] = grams[[10]]
    train_df['same_start_word3'] = grams[[11]]
    train_df['q1stops_ratio'] = grams[[12]]
    train_df['q2stops_ratio'] = grams[[13]]
    train_df['diff_stops_ratio'] = abs(grams[[14]])
    train_df['stops_jaccard'] = grams[[15]]
    train_df['q1_ave_world'] = grams[[16]]
    train_df['q2_ave_world'] = grams[[17]]
    train_df['diff_ave_world'] = abs(grams[[18]])
    train_df['shared_two_grams'] = grams[[19]]
    train_df['shared_two_grams_ratio'] = grams[[20]]
    train_df['q1_freq'] = grams[[21]]
    train_df['q2_freq'] = grams[[22]]
    train_df['q1_q2_intersect'] = grams[[23]]
    train_df['diff_char_len'] = grams[[24]]
    train_df['diff_word_len'] = grams[[25]]
    train_df['diff_2grams_len'] = grams[[26]]
    train_df['diff_unigrams'] = grams[[27]]
    train_df['diff_bigrams'] = grams[[28]]
    train_df['diff_trigrams'] = grams[[29]]
    train_df['diff_fourgrams'] = grams[[30]]
    train_df['diff_stops'] = grams[[31]]
    train_df['total_uni_words'] = grams[[32]]
    train_df['total_uni_stops'] = grams[[33]]
       
    print 'done grams'
    
    #######################Create new feature here and then add feature name to feature_list
    
    train_df['cross_idf'] = train_df['word_share'] * train_df['tfidf_word_share_stops']
    train_df['sum_idf'] = train_df['word_share'] + train_df['tfidf_word_share_stops']
    train_df['diff_idf'] = abs(train_df['word_share'] - train_df['tfidf_word_share_stops'])
    
     ######################################################################################
    
    feature_list = ['q1len','q2len',
    'q1_words','q2_words','word_share',
    'tfidf_word_share_stops',
    'q1stops_ratio','q2stops_ratio','diff_stops_ratio',
    'unigrams_common_count','unigrams_common_ratio',
    'bigrams_common_count','bigrams_common_ratio',
    'trigrams_common_count','trigrams_common_ratio',
    'fourgrams_common_count','fourgrams_common_ratio',
    'jaccard',
    'same_start_word1','same_start_word2','same_start_word3',
    'stops_jaccard',
    'q1_ave_world','q2_ave_world','diff_ave_world',
    'cross_idf','sum_idf','diff_idf',
    'shared_two_grams','shared_two_grams_ratio',
    'q1_freq','q2_freq','q1_q2_intersect',
    'qid1_max_kcore','qid2_max_kcore',
    'diff_char_len','diff_word_len','diff_2grams_len','diff_stops',
    'diff_unigrams','diff_bigrams','diff_trigrams','diff_fourgrams',
    'total_uni_words','total_uni_stops'
    ]
   
    return train_df[feature_list]



# get train X and train Y
train_X = feature_selection(train_df)
train_Y = train_df[['is_duplicate']]

# get test X
test_X = feature_selection(test_df)
test_id = test_df['test_id']

#add feature page_rank
page_rank_train = pd.read_csv('pagerank_train.csv')
train_X = pd.concat([train_X, page_rank_train], axis=1).reset_index(drop=True)

page_rank_test = pd.read_csv('pagerank_test.csv')
test_X = pd.concat([test_X, page_rank_test], axis=1).reset_index(drop=True)

#deal with na value
train_X['q2len'].fillna(train_X['q2len'].mean(), inplace=True)
test_X['q1len'].fillna(test_X['q1len'].mean(), inplace=True)
test_X['q2len'].fillna(test_X['q2len'].mean(), inplace=True)

train_X.to_csv('featured_train.csv',index=False)
test_X.to_csv('featured_test.csv',index=False)

######resampling
whole_train = train_X
whole_train['is_duplicate'] = train_Y

pos_train = whole_train[whole_train['is_duplicate'] ==1]
neg_train = whole_train[whole_train['is_duplicate'] == 0]

train_X = pd.concat([neg_train, pos_train, neg_train, neg_train])
train_Y = np.array([0]*neg_train.shape[0] + [1]*pos_train.shape[0] + [0]*neg_train.shape[0] + [0]*neg_train.shape[0])

train_X = train_X.drop('is_duplicate', axis=1)
train_Y = pd.DataFrame({ 'is_duplicate':train_Y })
print "mean target rate:", train_Y.mean()


# split train set to train_train and train_test to test our model
is_test = np.random.uniform(0,1,len(train_X)) > 0.85
train_train_X = train_X[is_test == True]
train_test_X = train_X[is_test == False]

train_train_Y = train_Y[is_test == True]['is_duplicate']
train_test_Y = train_Y[is_test == False]['is_duplicate']

##############################

############################# test model xgboost
#def runXGB(train_X, train_Y, test_X, test_Y=None, feature_names=None, seed_val=0):
def runXGB(train_X, train_Y, test_X, test_Y=None, feature_names=None):
    params = {}
    params['objective'] = 'binary:logistic'
    params['eval_metric'] = 'logloss'
    params['eta'] = 0.02
    params['subsample'] = 0.6
    params['min_child_weight'] = 1
    params['colsample_bytree'] = 0.7
    params['max_depth'] = 4 ###
    params['silent'] = 1
    #params['seed'] = seed_val
    num_rounds = 3000
    plst = list(params.items())
    xgtrain = xgb.DMatrix(train_X, label = train_Y)
    
    if test_Y is not None:
        xgtest = xgb.DMatrix(test_X, label=test_Y)
        watchlist = [(xgtrain,'train'),(xgtest,'test')]
        model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds = 100, verbose_eval=10)
    else:
        xgtest = xgb.DMatrix(test_X)
        model = xgb.train(plst, xgtrain, num_rounds)
    
    pred_test_y = model.predict(xgtest)
    loss = 1
    if test_Y is not None:
        loss = log_loss(test_Y, pred_test_y)
        return pred_test_y, loss, model
    else:
        return pred_test_y, loss, model
      
pred_train_test_y, loss, model = runXGB(train_train_X, train_train_Y, train_test_X, train_test_Y)
print('log loss for Xgboost is ', loss)
plot_importance(model)
###############################################################

########################### final predict
# Use model  XGBoost
pred_test_y,loss,model = runXGB(train_X, train_Y, test_X)


# print final result to cvs
out_df = pd.DataFrame({'test_id':test_id,'is_duplicate':pred_test_y})
out_df.to_csv('res0606_2_4.csv',index=False)


end_time = time.time()
print time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(end_time))
print (end_time - start_time)/3600
