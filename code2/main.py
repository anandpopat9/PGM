
import numpy as np
import csv
import nltk
from hmmlearn.hmm import GaussianHMM
from nltk.corpus import treebank
from nltk.tag import hmm
import math

data=[]
with open('/Users/chiragyeole/Downloads/Sheet_1.csv') as f:
	reader = csv.reader(f)
	for row in reader:
		data.append(row[2])
#print data

dict={}
dict['CC']=1;
dict['CD']=2;
dict['DT']=3;
dict['EX']=4;
dict['FW']=5;
dict['IN']=6;
dict['JJ']=7;
dict['JJR']=8;
dict['JJS']=9;
dict['LS']=10;
dict['MD']=11;
dict['NN']=12;
dict['NNS']=13;
dict['NNP']=14;
dict['NNPS']=15;
dict['PDT']=16;
dict['POS']=17;
dict['PRP']=18;
dict['PRP$']=19;
dict['RB']=20;
dict['RBR']=21;
dict['RBS']=22;
dict['RP']=23;
dict['SYM']=24;
dict['TO']=25;
dict['UH']=26;
dict['VB']=27;
dict['VBD']=28;
dict['VBG']=29;
dict['VBN']=30;
dict['VBP']=31;
dict['VBZ']=32;
dict['WDT']=33;
dict['WP']=34;
dict['WP$']=35;
dict['WRB']=36;

X=[[1]]
len1=[]
for i in range(1,80):
	#print(data[i])
	words=data[i].split()
	#print(words)
	pos=nltk.pos_tag(words)
	obs=[]
	for i in xrange(len(pos)):
		n=dict[pos[i][1]]
		obs.append([n])
	#print list(obs)
	len1.append(len(obs))
	X=np.concatenate([X,obs])

#print X
#print len1
'''

X1=[[1],[2],[3],[4]]
X2=[[1],[2],[4],[3]]
X3=[[1],[2],[3],[4]]
X4=[[1],[2],[4]]
X = np.concatenate([X1, X2,X3,X4])
print(X)

lengths = [len(X1), len(X2),len(X3),len(X4)]
'''
print("######################################################")
print("Fitting to HMM and decoding ...")
model = GaussianHMM(n_components=36, covariance_type="diag", n_iter=1000).fit(X,len1)
#print(model)
print("######################################################")
print("############## Start Probabilities ###################")
print(model.startprob_)
Z=model.predict(X)
print(Z)

#
# Transition Matrix #
#
print("######################################################")
print("################ Transition_Matrix ###################")
print(model.transmat_)
print('\n')

#
# Hidden Layers - Means and Variance #
#

for x in xrange(model.n_components):
	print("######################################################")
	print("Hidden Layer "+str(x)+":")
	print("\n")
	print("Mean: "+str(model.means_[x]))
	print("Variance: "+str(model.covars_[x]))
	print("\n")

#########################################################################################
#
# Supervised learning using nltk
#


pos_tags=[]
for i in range(1,80):
	#print(data[i])
	words=data[i].split()
	#print(words)
	pos=nltk.pos_tag(words)
	pos_tags.append(pos)

len_of_data=len(pos_tags)
len_of_train_data=abs(0.95 * len_of_data)
len_of_train_data=math.floor(len_of_train_data)
len_of_train_data=int(len_of_train_data)


#train=pos_tags[:len_of_train_data]
train=treebank.tagged_sents()

#print train

#
# Train the data
#

myhmm = hmm.HiddenMarkovModelTrainer()
tag1 = myhmm.train_supervised(train)
#print len_of_train_data
#
# Test data set
#

len_of_test_data=abs(0.2 * len_of_data)
len_of_test_data=math.floor(len_of_test_data)
len_of_test_data=int(len_of_test_data)
start=len(pos_tags)-len_of_test_data

#print start
test=data[start:]

#
# Results
#

result=[]
for x in xrange(len(test)):
	words1=test[x].split()
	#print(words)
	pos1=tag1.tag(words1)
	result.append(pos1)
print("######################################################")
print ("########### Supervised learning #####################")
print result

##############################################################################
#
# Unsupervised learning
#

big_seq=[]
symbol=[]
for words in range(0,80):
	word=data[words].split(' ')
	#print word
	seq=[]
	for x in word:
		symbol.append(x)
		tup=(x,'')
	#print tup
		seq.append(tup)
	big_seq.append(seq)
#print( "######### Printing Big sequence #########")
#print big_seq

#seq = [map(lambda x:(x,''), ss.split(' ')) for ss in sentences]
#print seq

#symbols = list(set([ss[0] for sss in big_seq for ss in sss]))

#print symbols
#print symbol

#print seq

states = range(5)
#rint states
trainer = nltk.tag.hmm.HiddenMarkovModelTrainer(states=states,symbols=symbol)
m = trainer.train_unsupervised(big_seq)

#m.random_sample(random.Random(),10)
print("###################################################################")
print ("########### Unsupervised learning #####################")
print m.tag("I just try to be kind and helpful".split())
