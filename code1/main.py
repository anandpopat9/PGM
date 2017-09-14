"""
Created on Thu Feb 23 15:13:26 2017

@author: anandpopat
"""
#### DATA CLEANING ####
import csv
import numpy as np
import pandas as pd
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from pgmpy.sampling import BayesianModelSampling
from pgmpy.inference import BeliefPropagation
from pgmpy.sampling import GibbsSampling
import scipy as sc

#Get the modified columns file
columns=[]
array=[]
i=0
l=[]
with open('columns1.csv', 'rb') as csvfile:
    for line in csvfile.readlines():
        array = line.split(',')
        l.append(array[1])
        i=i+1
        
#get the response file

i=0
final=[]
l1=[]
index=[]
with open('responses.csv', 'rb') as csvfile:
    for line in csvfile.readlines():
        array = line.split(',')
        l1.append(array)
        i=i+1

#get the index of the columns to keep
i=1
j=0
try:    
    while(j<len(l1[0])):
        if(l[i].rstrip()==l1[0][j].replace('"','')):
            index.append(j)
            i=i+1
        j=j+1
except IndexError:
    q='error'
    

#get the columns to keep for all the rows

j=0
final_list=[]
inside=[]

with open('responses.csv', 'rb') as csvfile:
    for line in csvfile.readlines():
        line=line.split(',')
        inside=[]
        for index1 in index:
            if line[index1] is '1' or line[index1] is '2':
                line[index1]='0'
            if line[index1] is '3' or line[index1] is '4' or line[index1] is '5':
                line[index1]='1'
            inside.append(line[index1])
        final_list.append(inside)
'''        
with open("data1.csv", "wb") as f:
    writer = csv.writer(f)
    writer.writerows(final_list)
'''

#### calculating CPTs ############

#initialize the data
data=pd.DataFrame.from_csv('/Users/anandpopat/desktop/spring/aml/data1.csv')

#Gender
gen=data.groupby(['Gender']).size().div(len(data))

#print gen

#Education
edu=data.groupby(['Education']).size().div(len(data))

#print edu
#Left-Right handed
lrh=data.groupby(['Left - right handed']).size().div(len(data))

#print lrh
#Movies
mov=data.groupby(['Movies','Gender', 'Left - right handed','Education']).size().div(len(data))
mov1=data.groupby(['Gender', 'Left - right handed','Education']).size().div(len(data))

#print mov
#print ">>>>>>>>>>>>>"
#print mov1
#print ">>>>>>>>>>>>>"
#print mov[0].div(mov1)
#print ">>>>>>>>>>>>>"
#print mov[1].div(mov1)


#Mathematics
mat=data.groupby(['Mathematics','Education']).size().div(len(data))
mat1=data.groupby(['Education']).size().div(len(data))

#print mat
#print ">>>>>>>>>>>>>"
#print mat1
#print ">>>>>>>>>>>>>"
#print mat[0].div(mat1)
#print ">>>>>>>>>>>>>"
#print mat[1].div(mat1)

#Art
art=data.groupby(['Art exhibitions','Gender', 'Left - right handed','Education']).size().div(len(data))
art1=data.groupby(['Gender', 'Left - right handed','Education']).size().div(len(data))

#print art
#print ">>>>>>>>>>>>>"
#print art1
#print ">>>>>>>>>>>>>"
#print art[0].div(art1)
#print ">>>>>>>>>>>>>"
#print art[1].div(art1)


#Music
mus=data.groupby(['Music1','Gender', 'Left - right handed','Education','Art exhibitions']).size().div(len(data))
mus1=data.groupby(['Gender', 'Left - right handed','Education','Art exhibitions']).size().div(len(data))

#print mus
#print ">>>>>>>>>>>>>"
#print mus1
#print ">>>>>>>>>>>>>"
#print mus[0].div(mus1)
#print ">>>>>>>>>>>>>"
#print mus[1].div(mus1)

#Sci-fi
sci=data.groupby(['Sci-fi','Mathematics','Movies']).size().div(len(data))
sci1=data.groupby(['Mathematics','Movies']).size().div(len(data))
#
#print sci
#print ">>>>>>>>>>>>>"
#print sci1
#print ">>>>>>>>>>>>>"
#print sci[0].div(sci1)
#print ">>>>>>>>>>>>>"
#print sci[1].div(sci1)

#Horror
hor=data.groupby(['Horror','Gender','Movies']).size().div(len(data))
hor1=data.groupby(['Gender','Movies']).size().div(len(data))

#print hor
#print ">>>>>>>>>>>>>"
#print hor1
#print ">>>>>>>>>>>>>"
#print hor[0].div(hor1)
#print ">>>>>>>>>>>>>"
#print hor[1].div(hor1)

#Romantic
rom=data.groupby(['Romantic','Movies','Gender']).size().div(len(data))
rom1=data.groupby(['Movies','Gender']).size().div(len(data))

#print rom
#print ">>>>>>>>>>>>>"
#print rom1
#print ">>>>>>>>>>>>>"
#print rom[0].div(rom1)
#print ">>>>>>>>>>>>>"
#print rom[1].div(rom1)

#Pop
pop=data.groupby(['Pop','Music1']).size().div(len(data))
pop1=data.groupby(['Music1']).size().div(len(data))

#print pop
#print ">>>>>>>>>>>>>"
#print pop1
#print ">>>>>>>>>>>>>"
#print pop[0].div(pop1)
#print ">>>>>>>>>>>>>"

#metal
met=data.groupby(['Metal or Hardrock','Music1']).size().div(len(data))
met1=data.groupby(['Music1']).size().div(len(data))

#print met
#print ">>>>>>>>>>>>>"
#print met1
#print ">>>>>>>>>>>>>"
#print met[0].div(met1)
#print ">>>>>>>>>>>>>"
#print met[1].div(met1)


########################################################3

####### GOD #########

p_god_emp_rel = data.groupby(['God','Empathy','Religion']).size().div(len(data))
p_emp_rel = data.groupby(['Empathy','Religion']).size().div(len(data))
GOD =  p_god_emp_rel.unstack('God').div(p_emp_rel,axis=0,level=['Empathy','Religion'])
#print(GOD)

###### GIVING #######

p_giving_emp_rel = data.groupby(['Giving','Empathy','Religion']).size().div(len(data))
p_gemp_rel = data.groupby(['Empathy','Religion']).size().div(len(data))
GIVING =  p_giving_emp_rel.unstack('Giving').div(p_gemp_rel,axis=0,level=['Empathy','Religion'])

###### RELIGION #######

p_rel_gen_edu_emp_lon = data.groupby(['Religion','Gender','Education','Empathy','Loneliness']).size().div(len(data))
p_gen_edu_emp_lon = data.groupby(['Gender','Education','Empathy','Loneliness']).size().div(len(data))
RELIGION =  p_rel_gen_edu_emp_lon.unstack('Religion').div(p_gen_edu_emp_lon,axis=0,level=['Gender','Education','Empathy','Loneliness'])

###### FRIENDS ######

p_friends_emp_soc_int = data.groupby(['Fun with friends','Empathy','Socializing','Internet']).size().div(len(data))
p_emp_soc_int = data.groupby(['Empathy','Socializing','Internet']).size().div(len(data))
FRIENDS =  p_friends_emp_soc_int.unstack('Fun with friends').div(p_emp_soc_int,axis=0,level=['Empathy','Socializing','Internet'])
#print(FRIENDS)

####### PUBLIC SPEAKING ########

p_speak_frends_soc = data.groupby(['Public speaking','Fun with friends','Socializing']).size().div(len(data))
p_frends_soc = data.groupby(['Fun with friends','Socializing']).size().div(len(data))
SPEAKING =  p_speak_frends_soc.unstack('Public speaking').div(p_frends_soc,axis=0,level=['Fun with friends','Socializing'])
#print(SPEAKING)
######  EMPATHY ##########

EMPATHY = data.groupby('Empathy').size().div(len(data))
#print(EMPATHY)

p_happy_frends_int_emp_rel_speak = data.groupby(['Happiness in life','Fun with friends','Internet','Empathy','Religion','Public speaking']).size().div(len(data))
p_frends_int_emp_rel_speak = data.groupby(['Fun with friends','Internet','Empathy','Religion','Public speaking']).size().div(len(data))
HAPPY =  p_happy_frends_int_emp_rel_speak.unstack('Happiness in life').div(p_frends_int_emp_rel_speak,axis=0,level=['Fun with friends','Internet','Empathy','Religion','Public speaking'])
#print(HAPPY)

########### SMOKING ##############

p_smok_frends_metal_lon = data.groupby(['Smoking','Fun with friends','Metal or Hardrock','Loneliness']).size().div(len(data))
p_frends_metal_lon = data.groupby(['Fun with friends','Metal or Hardrock','Loneliness']).size().div(len(data))
SMOKING =  p_smok_frends_metal_lon.unstack('Smoking').div(p_frends_metal_lon,axis=0,level=['Fun with friends','Metal or Hardrock','Loneliness'])
#print(SMOKING)

##### SOCIALIZING #################

p_soc_pol_emp_rom = data.groupby(['Socializing','Politics','Empathy','Romantic']).size().div(len(data))
p_pol_emp_rom = data.groupby(['Politics','Empathy','Romantic']).size().div(len(data))
SOCIALIZING =  p_soc_pol_emp_rom.unstack('Socializing').div(p_pol_emp_rom,axis=0,level=['Politics','Empathy','Romantic'])
#print(SOCIALIZING)

##########LONELINESS ################

p_lon_soc_frends = data.groupby(['Loneliness','Socializing','Fun with friends']).size().div(len(data))
p_soc_frends = data.groupby(['Socializing','Fun with friends']).size().div(len(data))
LONELINESS =  p_lon_soc_frends.unstack('Loneliness').div(p_soc_frends,axis=0,level=['Socializing','Fun with friends'])
#print(LONELINESS)

##### INTERNET ######
p_internet = data.groupby('Internet').size().div(len(data))
INTERNET = data.groupby(['Internet','Sci-fi']).size().div(len(data)).div(p_internet, axis=0, level='Internet')
#print(INTERNET)

p_pol_gen_edu = data.groupby(['Politics','Gender','Education']).size().div(len(data))
p_gen_edu = data.groupby(['Gender','Education']).size().div(len(data))
POLITICS = p_pol_gen_edu.unstack('Politics').div(p_gen_edu,axis=0,level=['Gender','Education'])
#print(POLITICS)

######## ALCOHOL ########
p_alco_frends_met_lon = data.groupby(['Alcohol','Fun with friends','Metal or Hardrock','Loneliness']).size().div(len(data))
p_frends_met_lon = data.groupby(['Fun with friends','Metal or Hardrock','Loneliness']).size().div(len(data))
ALCOHOL = p_alco_frends_met_lon.unstack('Alcohol').div(p_frends_met_lon,axis=0,level=['Fun with friends','Metal or Hardrock','Loneliness'])
#print(ALCOHOL)

#### BUILDING BAYESIAN MODEL IN PGMPY AND FEEDING CPTs #############

bayesian_model = BayesianModel([('Gender', 'Religion'), 
                       ('Gender', 'Art'), 
                       ('Gender', 'Music'), 
                       ('Gender', 'Politics'),
                       ('Gender', 'Horror'),
                       ('Gender', 'Romantic'),
                       ('Gender', 'Movies'),
                       ('Education', 'Politics'),
                       ('Education', 'Music'),
                       ('Education', 'Art'),
                       ('Education', 'Movies'),
                       ('Education', 'Mathematics'),
                       ('Education', 'Religion'),
                       ('Left - right handed', 'Art'),
                       ('Left - right handed', 'Movies'),
                       ('Left - right handed', 'Music'),
                       ('Movies', 'Horror'),
                       ('Movies', 'Romantic'),
                       ('Movies', 'Sci-fi'),
                       ('Mathematics', 'Sci-fi'),
                       ('Music', 'Pop'),
                       ('Music', 'Metal'),
                       ('Art', 'Music'),
                       ('Sci-fi', 'Internet'),
                       ('Metal', 'Alcohol'),
                       ('Metal', 'Smoking'),
                       ('Romantic', 'Socializing'),
                       ('Politics', 'Socializing'),
                       ('Internet', 'Friends'),
                       ('Internet', 'Happy'),
                       ('Lonliness', 'Alcohol'),
                       ('Lonliness', 'Smoking'),
                       ('Lonliness', 'Religion'),
                       ('Empathy', 'Religion'),
                       ('Empathy', 'Happy'),
                       ('Empathy', 'Giving'),
                       ('Empathy', 'God'),
                       ('Empathy', 'Friends'),
                       ('Empathy', 'Socializing'),
                       ('Socializing', 'Public Speaking'),
                       ('Socializing', 'Friends'),
                       ('Socializing', 'Lonliness'),
                       ('Religion', 'Happy'),
                       ('Religion', 'God'),
                       ('Religion', 'Giving'),
                       ('Friends', 'Happy'),
                       ('Friends', 'Alcohol'),
                       ('Friends', 'Smoking'),
                       ('Friends', 'Lonliness'),
                       ('Friends', 'Public Speaking'),
                       ('Public Speaking', 'Happy')])

cpd_gen = TabularCPD(variable='Gender', variable_card=2, values=[[0.58, 0.42]])
cpd_edu = TabularCPD(variable='Education', variable_card=6, values=[[0.20, 0.00,0.00,0.08,0.07,0.65]])
cpd_lrh = TabularCPD(variable='Left - right handed', variable_card=2, values=[[0.1, 0.9]])

cpd_mov = TabularCPD(variable='Movies', variable_card=2, 
                   values=[[0.12,0.5,0.5, 0, 0.15, 0,0,0,0.5,0,0.06,0.00,0,0,0,0,0,0,0.03,0,0,0,0,0.02],
                           [0.88,0.5,0.5, 1, 0.85, 1,1,1,0.5,1,0.94,1.00,1,1,1,1,1,1,0.97,1,1,1,1,0.98]],
                  evidence=['Gender','Left - right handed', 'Education'],
                  evidence_card=[2, 2, 6])
#print cpd_mov

cpd_mat = TabularCPD(variable='Mathematics', variable_card=2, 
                   values=[[0.57,0.50,0.40,0.45,0.65,0.59],
                           [0.43,0.50,0.60,0.55,0.35,0.41]],
                  evidence=['Education'],
                  evidence_card=[6])

#print cpd_mat

cpd_mus = TabularCPD(variable='Music', variable_card=2, 
                   values=[[0,0,0.5,0.5,0.5,0.5,0,0,0,0,0,0,0,0.04,0,0,0.5,0.5,0,0,0,0,0.02,0.01,0,0,0,0.5,0,0,0,0.5,0,0.5,0,0,0.12,0.03,0,0,0.500000,0.5,0,0,0.09,0,0.04,0.01],
                           [1,1,0.5,0.5,0.5,0.5,1,1,1,1,1,1,1,0.96,1,1,0.5,0.5,1,1,1,1,0.98,0.99,1,1,1,0.5,1,1,1,0.5,1,0.5,1,1,0.88,0.97,1,1,0.500000,0.5,1,1,0.91,1,0.96,0.99]],
                  evidence=['Gender','Left - right handed', 'Education','Art'],
                  evidence_card=[2, 2, 6, 2])
#print cpd_mus

cpd_art = TabularCPD(variable='Art', variable_card=2, 
                   values=[[0.62,0.5,0.5,0.34,0.42,0.24,0.43,0.34,0.5,0.35,0.54,0.46,0.500000,1,0.34,0.34,1,0.78,0.54,0.66,0,0.45,0.6,0.63],
                           [0.38,0.5,0.5,0.66,0.58,0.76,0.57,0.66,0.5,0.65,0.46,0.54,0.500000,0,0.66,0.66,0,0.22,0.46,0.34,1,0.55,0.4,0.37]],
                  evidence=['Gender','Left - right handed', 'Education'],
                  evidence_card=[2, 2, 6])

#print cpd_art

cpd_sci = TabularCPD(variable='Sci-fi', variable_card=2, 
                   values=[[0.66,0.29,0.57,0.26],
                           [0.34,0.71,0.43,0.74]],
                  evidence=['Mathematics','Movies'],
                  evidence_card=[2, 2])
#print cpd_sci

cpd_pop = TabularCPD(variable='Pop', variable_card=2, 
                   values=[[0.3,0.97],
                           [0.7,0.03]],
                  evidence=['Music'],
                  evidence_card=[2])
#print cpd_pop

cpd_met = TabularCPD(variable='Metal', variable_card=2, 
                   values=[[0.500000,0.59],
                           [0.500000,0.41]],
                  evidence=['Music'],
                  evidence_card=[2])
#print cpd_met

cpd_hor = TabularCPD(variable='Horror', variable_card=2, 
                   values=[[0.87,0.51,0.75,0.36],
                           [0.13,0.49,0.25,0.64]],
                  evidence=['Gender','Movies'],
                  evidence_card=[2,2])
#print cpd_hor

cpd_rom = TabularCPD(variable='Romantic', variable_card=2, 
                   values=[[0.12,0.500000,0.11,0.37],
                           [0.88,0.500000,0.89,0.63]],
                  evidence=['Movies','Gender'],
                  evidence_card=[2,2])

#print cpd_rom

cpd_religion= TabularCPD(variable='Religion', variable_card=2,
                   values=[[0.6,0.55,0.65,0.64, 0.5, 0.5, 0.5,0.500000,1,1,0.29,0.42, 0.5, 0.5, 0.5, 0.5, 0.5 ,0.34,0.64,0.65,0.64,0.54,0.63,0.64,0.59,0.5,0.65,0.71,0.5,0,0.5,0.67,0,0.5, 0.5,0.34,0,0.84,0.250000,0.36, 0.5, 1,0.58,0.73,0.86,0.75,0.60,0.58],
                           [0.4,0.45,0.35,0.36, 0.5, 0.5, 0.5,0.500000,0,0,0.71,0.58, 0.5, 0.5, 0.5, 0.5, 0.5 ,0.66,0.36,0.35,0.36,0.46,0.37,0.36,0.41,0.5,0.35,0.29,0.5,1,0.5,0.33,1,0.5, 0.5,0.66,1,0.16,0.750000,0.64, 0.5, 0,0.42,0.27,0.14,0.25,0.40,0.42]],
                  evidence=['Gender','Education','Empathy','Lonliness'],
                  evidence_card=[2, 6, 2,  2])

#print cpd_religion

cpd_friends= TabularCPD(variable='Friends', variable_card=2,
                   values=[[0.5,0.02,0,0.01,0.12,0.05,0,0.01],
                           [0.5,0.98,1,0.99,0.88,0.95,1,0.99]],
                  evidence=['Empathy', 'Socializing','Internet'],
                  evidence_card=[2, 2, 2])

#print cpd_friends

cpd_speaking = TabularCPD(variable='Public Speaking', variable_card=2,
                   values=[[0.14,0.11,0.12,0.25],
                           [0.86,0.89,0.88,0.75]],
                  evidence=['Friends', 'Socializing'],
                  evidence_card=[2, 2])
#print cpd_speaking

cpd_empathy = TabularCPD(variable='Empathy', variable_card=2, values=[[0.12, 0.88]])

#print cpd_empathy

cpd_happy= TabularCPD(variable='Happy', variable_card=2,
                   values=[[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0,0.5,0.5,0.5,0,1,0.4,0,0.34,0,0,0,0.5,0,0,0,0.08,0,0.09,0.09,0.11,0.06,0.07,0.01,0.05],
                           [0.5,0.5,0.5,0.5,0.5,0.5,0.5,1,0.5,0.5,0.5,1,0,0.6,1,0.66,1,1,1,0.5,1,1,1,0.92,1,0.91,0.91,0.89,0.94,0.93,0.99,0.95]],
                  evidence=['Friends', 'Internet','Empathy','Religion','Public Speaking'],
                  evidence_card=[2, 2, 2, 2, 2])

#print cpd_happy
cpd_smoking= TabularCPD(variable='Smoking', variable_card=4,
                   values=[[0.200000,0.0,0,0.00,0.17,0.16,0.22,0.22],
                           [0.200000,0.2,0,0.00,0.17,0.19,0.16,0.15],
                           [0.200000,0.6,0,0.66,0.19,0.20,0.20,0.19],
                           [0.400000,0.2,1,0.34,0.47,0.45,0.42,0.44]],
                  evidence=['Friends', 'Metal','Lonliness'],
                  evidence_card=[2, 2, 2])

#print cpd_smoking

cpd_socializing= TabularCPD(variable='Socializing', variable_card=2,
                   values=[[0.3,0.21,0.41,0.24,0.47,0.18,0.21,0.23],
                           [0.7,0.79,0.59,0.76,0.53,0.82,0.79,0.77]],
                  evidence=['Politics', 'Empathy','Romantic'],
                  evidence_card=[2, 2, 2])

#print cpd_socializing

cpd_loneliness = TabularCPD(variable='Lonliness', variable_card=2,
                   values=[[0.14,0.31,0.55,0.39],
                           [0.86,0.69,0.45,0.61]],
                  evidence=['Socializing', 'Friends'],
                  evidence_card=[2, 2])

#print cpd_loneliness

cpd_internet = TabularCPD(variable='Internet', variable_card=2,
                   values=[[0.45, 0.35],
                           [0.55, 0.65]],
                   evidence=['Sci-fi'],
                   evidence_card=[2])

#print cpd_internet

cpd_politics = TabularCPD(variable='Politics', variable_card=2,
                   values=[[0.52, 0.84, 0.5, 0.45, 0.66, 0.57, 0.41, 0.75, 0.20, 0.52, 0.66, 0.38],
                           [0.48, 0.16, 0.5, 0.55, 0.34, 0.43, 0.59, 0.25, 0.80, 0.48, 0.34, 0.62]],
                  evidence=['Gender', 'Education'],
                  evidence_card=[2, 6])

#print cpd_politics

cpd_alcohol= TabularCPD(variable='Alcohol', variable_card=3,
                   values=[[0.0,0.1,0,0.16,0.22,0.17,0.29,0.25],
                           [0.2,0.2,0,0.16,0.13,0.10,0.13,0.13],
                           [0.8,0.7,1,0.68,0.65,0.73,0.58,0.62]],
                  evidence=['Friends', 'Metal','Lonliness'],
                  evidence_card=[2, 2, 2])

#print cpd_alcohol

cpd_god = TabularCPD(variable='God', variable_card=2,
                   values=[[0.523256, 0.139535, 0.415354, 0.125714],
                           [0.485116, 0.870465, 0.592734, 0.884286]],
                  evidence=['Empathy', 'Religion'],
                  evidence_card=[2, 2])

#print cpd_god

cpd_giving = TabularCPD(variable='Giving', variable_card=2,
                   values=[[0.534884,0.372093,0.342256,0.351429],
                           [0.475116,0.637907,0.657744,0.648571]],
                  evidence=['Empathy', 'Religion'],
                  evidence_card=[2, 2])

#print cpd_giving

bayesian_model.add_cpds(cpd_gen, cpd_edu, cpd_lrh, cpd_mov, cpd_mus,cpd_mat,cpd_sci,cpd_art,cpd_pop,cpd_met,cpd_hor,
               cpd_rom,cpd_religion,cpd_friends,cpd_speaking,cpd_empathy,cpd_happy,
               cpd_smoking,cpd_socializing,cpd_loneliness,cpd_internet,cpd_politics,
               cpd_alcohol,cpd_god,cpd_giving)

bayesian_model.check_model()

#### Bayesian Model Sampling

infer = BayesianModelSampling(bayesian_model)
bay = infer.forward_sample(100)

#### calculate mean of each distribution
mean=[]
for cat in bay:
    mean.append(np.mean(bay[cat]))
    
print '#### MEAN ####\n'
print mean
    

#### calculate entropy of each distribution
entropy=[]
for c in range(0,5):
    norm=sc.stats.norm(bay).pdf(c)
    entropy.append(sc.stats.entropy(norm))

print '#### ENTROPY ####\n'
print entropy
    
#### Relative entropy b/w two distributions
#### suppose for distribution 0 and 1

dist1=sc.stats.norm(bay).pdf(0)
dist2=sc.stats.norm(bay).pdf(1)

print '#### RELATIVE ENTROPY ####\n'
print sc.stats.entropy(dist1,dist2)


infer = VariableElimination(bayesian_model)
belief_propagation = BeliefPropagation(bayesian_model)


#Queries and Inferences

print "#### QUERIES AND  INFERENCES ####\n"
print "NOTE: Processing Queries may take many minutes\n"

print "Query1 -> Do people who like socializing have more friends"
print "Inference1 -> Sociable people have more number of friends"
print "\nThrough Variable Elimination\n"
print(infer.query(['Friends'],evidence={'Socializing':1}) ['Friends'])
print "Through Belief Propogation\n"
print belief_propagation.map_query(variables=['Friends'],evidence={'Socializing': 1})

print "Query2 -> Which gender likes mathematics more"
print "Inference2 -> Females like mathematics more than male"
print "\nThrough Variable Elimination\n"
print(infer.query(['Mathematics'],evidence={'Gender':1}) ['Mathematics'])
print "Through Belief Propogation\n"
print belief_propagation.map_query(variables=['Mathematics'],evidence={'Gender': 1})

print "Query3 -> Do people who like metal songs and smoking tend to drink more"
print "Inference3 -> People drink more if they like metal and smoking"
print "\nThrough Variable Elimination\n"
print(infer.query(['Alcohol'],evidence={'Metal':1,'Smoking':2}) ['Alcohol'])
print "Through Belief Propogation\n"
print belief_propagation.map_query(variables=['Alcohol'],evidence={'Metal': 1,'Smoking':2})

print "Query4 -> Do people who are empathetic and generous have more friends"
print "Inference4 -> People who are empathetic and generous have more number of friends"
print "\nThrough Variable Elimination\n"
print(infer.query(['Friends'],evidence={'Empathy':1,'Giving':1}) ['Friends'])
print "Through Belief Propogation\n"
print belief_propagation.map_query(variables=['Friends'],evidence={'Empathy': 1,'Giving':1})

print "Query5 -> What is the higher level of education of female who like music,movies,art and dont like mathematics"
print "Inference5 -> Most female who like music, movies, art and dont like mathematics have secondary level education"
print "\nThrough Variable Elimination\n"
print(infer.query(['Education'],evidence={'Music':1,'Movies':1,'Art':1,'Mathematics':0}) ['Friends'])
print "Through Belief Propogation\n"
print belief_propagation.map_query(variables=['Education'],evidence={'Music':1,'Movies':1,'Art':1,'Mathematics':0})

