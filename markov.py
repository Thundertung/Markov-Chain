import numpy as np
import pandas as pd
from pandas import Series
import matplotlib.pyplot as plt 
SPRC = pd.read_csv("C:/Users/computer/Desktop/DATA/Dataset/SPRC/SPRC_Price.csv")
SPRC.columns
SPRC.info()
# no missing value 

#np.array(type loey)
#np.zeroes()
#np.arange(start,stop,marginal)
#np.linspace(start,stop,marginal)


#This things is suck but "OKAY"
SPRC[SPRC['SPRC.Close']>SPRC['SPRC.Open']].count
#94 times that Close price > Open Price


SPRC[SPRC['SPRC.Open']>SPRC['SPRC.Close']].count()
# 132 times that Close < Open 


# Now We have P(U) P(D) and P(S)

P_Up = 94/300
P_Down = 132/300
P_Side = 74/300

# SPRC = SPRC.drop(['OBL'],['OSL'])

#plt.plot(SPRC['Timestamp'][11:100],SPRC['SPRC.Close'][11:100],label = 'Close')
#plt.plot(SPRC['Timestamp'][11:100],SPRC['SPRC.Open'][11:100],label = 'Open')
#plt.xlabel('Days')
#plt.ylabel('Baht')
#plt.title('SPRC Close/Open Price')
#plt.legend()
#plt.show()


SPRC.mean()


#Movement_mapping = {"Up": 0, "Miss": 1, "Mrs": 2, 
#                 "Master": 3, "Dr": 3, "Rev": 3, "Col": 3, "Major": 3, "Mlle": 3,"Countess": 3,
#                 "Ms": 3, "Lady": 3, "Jonkheer": 3, "Don": 3, "Dona" : 3, "Mme": 3,"Capt": 3,"Sir": 3 }
#for dataset in traintest_dta:
#    dataset['Title'] = dataset['Title'].map(title_mapping)

SPRC['Movement'] = SPRC['SPRC.Close']-SPRC['SPRC.Open']

#for dataset in SPRC:
#    dataset.loc[dataset['Movement'] > 0 , 'Movement'] = 1,
#    dataset.loc[dataset['Movement'] < 0, 'Movement'] = -1,
#    dataset.loc[dataset['Movement'] ==0, 'Movement'] = 0

#plt.plot(SPRC['Timestamp'][11:100],SPRC['SPRC.Close'][11:100],label = 'Close')
#plt.plot(SPRC['Timestamp'][11:100],SPRC['SPRC.Open'][11:100],label = 'Open')
#plt.plot(SPRC['Timestamp'][11:100],SPRC['Movement'][11:100],label = 'Change')
#plt.xlabel('Days')
#plt.ylabel('Baht')
#plt.title('SPRC Close/Open Price')
#plt.legend()
#plt.show()

# create a boundary
avg_range = [SPRC['Movement'].mean()+1.96*(SPRC['Movement'].std()),SPRC['Movement'].mean()-1.96*(SPRC['Movement'].std())]

SPRC['Upper'] = SPRC['Movement'].mean()+1.96*(SPRC['Movement'].std())
SPRC['Lower'] = SPRC['Movement'].mean()-1.96*(SPRC['Movement'].std())

plt.plot(SPRC['Timestamp'][11:100],SPRC['Upper'][11:100],label = 'Upper')
plt.plot(SPRC['Timestamp'][11:100],SPRC['Lower'][11:100],label = 'Lower')
plt.plot(SPRC['Timestamp'][11:100],SPRC['Movement'][11:100],label = 'Change')

plt.xlabel('Days')
plt.ylabel('Baht')
plt.title('SPRC Close/Open Price')
plt.legend()
#plt.show()

#Probability counting

Up = 0
Down = 0
Side = 0
Total = 300
High = 0
Low = 0
Prob = SPRC['Movement'].values
for i in Prob:
    if i > 0:
        Up = Up +1
    elif i < 0:
        Down = Down + 1
    else:
        Side = Side + 1
print(Up)
print(Down)
print(Side)
UU = 0
UD =0
US = 0
DD =0
DU= 0
DS=0
SS=0
SU=0
SD=0
##for i in Prob:
##    if i > 0 and Prob[Prob.index(i-1)]>0:
##        UU = UU +1
##    elif i > 0 and  <0:
##        UD = UD +1
##    elif i > 0 and ==0:
##        US = US +1
##    elif i < 0 and <0:
##        DD = DD +1
##    elif i < 0 and >0:
##        DU =DU+1
##    elif i < 0 and ==0:
##        DS = DS +1
##    elif i == 0 and ==0:
##        SS=SS+1
##    elif i == 0 and  >0:
##        SU = SU +1
##    else:
##        SD =SD +1 
print(UU)
print(UD)
print(US)
print(DD)
print(DU)
print(DS)
print(SS)
print(SU)
print(SD)

##for i,j in enumerate(Prob):
##	if j == 0:
##		i = 1
##		Count = Count +i
##		print(Count)


# I have to admit I don't know how to do this in pd.DataFrame and Numpyarray
# So I just Convert from DataFrame to Numpyarray to list
# And that is the solution guy!!

Prob = SPRC['Movement'].values
# DONT FUCKIMNG USE THIS BEFORE ROUND AND INTEGERIZATION Prob = Prob.tolist()

# get rid of floating problem
# first what we have is the decimal of 0.00000
# so basically make it a real value first

Prob = Prob*100
# As the prediction is not about the amount of the change we can do the monotonic transformation
# 10000 to make sure we got the all the positive change as when we round the little positive become 0
#Then rounding and make it integer

newProb =[round(x) for x in Prob]
newProb =[int(x) for x in newProb]


# AT THIS PART IT will be ugly 
for i,j in enumerate(newProb,0):
	if j < 0 and newProb[i-1]>0:
		UD = UD +1 
		print(i,j,UD)


for i,j in enumerate(newProb,0):
	if j > 0 and newProb[i-1]>0:
		UU = UU +1 
		print(i,j,UU)
		
for i,j in enumerate(newProb,0):
	if j > 0 and newProb[i-1]==0:
		SU = SU +1 
		print(i,j,SU)

for i,j in enumerate(newProb,0):
	if j > 0 and newProb[i-1]<0:
		DU = DU +1 
		print(i,j,DU)

for i,j in enumerate(newProb,0):
	if j == 0 and newProb[i-1]>0:
		US = US +1 
		print(i,j,US)

for i,j in enumerate(newProb,0):
	if j < 0 and newProb[i-1]<0:
		DD = DD +1 
		print(i,j,DD)

for i,j in enumerate(newProb,0):
	if j == 0 and newProb[i-1]<0:
		DS = DS +1 
		print(i,j,DS)

for i,j in enumerate(newProb,0):
	if j < 0 and newProb[i-1]==0:
		SD = SD +1 
		print(i,j,SD)

for i,j in enumerate(newProb,0):
	if j == 0 and newProb[i-1]==0:
		SS = SS+ 1 
		print(i,j,SS)

print(UU)
print(UD)
print(US)
print(DD)
print(DU)
print(DS)
print(SS)
print(SU)
print(SD)

UgU = UU/Up
UgD = UD/Down
UgS = US/Side
DgU = DU/Up
DgD = DD/Down
DgS = DS/Side
SgS = SS/Side
SgU = SU/Up
SgD = SD/Down


transition = np.array([(UgU,SgU,DgU),(UgS,SgS,DgS),(UgD,SgD,DgD)])
transition2 =np.dot(transition,transition)
transition3 =np.dot(transition2,transition)

def Markov_Play(Total_Plays):
	Money= np.array([100,100,100])
	Move = np.dot(Money,transition)
	for i in range(Total_Plays):
		Move = np.dot(Move,transition)
	print(Move)

