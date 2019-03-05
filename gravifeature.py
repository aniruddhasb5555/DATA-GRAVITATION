import numpy as np
import sys
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler

class particle :
    def __init__(self,cen,ma) :
        self.centriod=cen
        self.mass=ma
    
def normalization(trainsample) :
    trainsample=np.array(trainsample)
    trainsample=np.transpose(trainsample)
    newsam=[]
    for i in trainsample :
        mi=min(i)
        ma=max(i)
        i=(i-mi)*(1/(ma-mi))
        newsam.append(i)
    newsam=np.array(newsam)
    newsam=np.transpose(newsam)
    return newsam

def calupdate(cen,partiCla) :
    global epsilon
    a=cen-partiCla.centriod
    a=np.power(a,2)
    a=np.sum(a)
    a=a**0.5
    if a<=epsilon :
        print(a)
        pc=np.array(partiCla.centriod)*partiCla.mass
        pc=pc+cen
        pc=np.divide(pc,partiCla.mass+1)
        partiCla.centriod=pc
        partiCla.mass+=1
        return 1
    else : return 0
        

#trainsample trainlable convert to 1,2,....in preprocessing
def train(trainsample,trainlable) :
    global PCL
    global DDD
    numflows=len(trainsample)
    for i in range(numflows) :
        print(i,end="    ")
        cl=DDD[trainlable[i]]
        cc=0
        for j in PCL[cl] :
            temp=np.array(trainsample[i])
            cc=calupdate(temp,j)
            if cc==1 :
                break
        if cc==0 :
            temp=np.array(trainsample[i])
            PCL[cl].append(particle(temp,1))

#testing the model
def calforce(testsample,j) :
    pcen=np.array(j.centriod)
    rsq=np.subtract(pcen,testsample)
    rsq=np.power(rsq,2)
    rsq=np.sum(rsq)
    if rsq==0 : return sys.maxsize
    else : return j.mass/rsq
    

# corner case is if forces r equal but will never arise i feel bcz of float values

def calAccu(pridicted,testlable,testnumflows) :
    ans=pridicted==testlable
    ans=(np.sum(ans)/testnumflows)*100
    return ans

def pridictClass(testsample) :
    global n,PCL
    force=[]
    for i in range(n) :
        temp=0
        for j in PCL[i] :
            temp+=calforce(testsample,j)
        force.append(temp)
    ma=0
    index=0
    for i in range(n) :
        if ma<=force[i] :
            ma=force[i]
            index=i
    return index


def test(testsample,testlable) :
    global PCL
    global DD
    testnumflows=len(testsample)
    pridicted=[]
    for i in range(testnumflows) :
        print(i)
        pridicted.append(DD[pridictClass(testsample[i])])
    accu=calAccu(pridicted,testlable,testnumflows)
    return accu

print("Program started")
PCL=[]
DDD={}
DD={}
#DDD["WWW"]=0
#DDD["MAIL"]=1
#DDD["FTP-DATA"]=2
#DDD["FTP-CONTROL"]=3
#DDD["FTP-PASV"]=4
#DDD["DATABASE"]=5
#DDD["SERVICES"]=6
#DDD["P2P"]=7
#DDD["ATTACK"]=8
#DDD["MULTIMEDIA"]= 9

epsilon=0.1
print("reading data........")
data = np.genfromtxt("Finial_entry01_ITC_Dataset train1.csv", dtype=float, delimiter=',', skip_header=1) 
#trainsample=data[:,[0,1,59,82,89,94,95,158]]
#trainsample = PCA(n_components=3)
#fit = pca.fit(X)
trainlable = np.genfromtxt("Finial_entry01_ITC_Dataset.csv", dtype=str, delimiter=',', skip_header=1) 
trainlable=trainlable[:,[-1]]
print(trainlable)
trainlable = trainlable.ravel()
DDD={}
trainlable1=[]
n=0
for i in trainlable :
    if i not in DDD :
        DDD[i]=n
        trainlable1.append(n)
        n+=1
    else :
        trainlable1.append(DDD[i])
data=MinMaxScaler().fit_transform(data)
imputer = Imputer()
data = imputer.fit_transform(data)
#data=normalization(data)
pca = PCA(n_components=5)
fit = pca.fit(data)
trainsample=pca.transform(data)
print(trainsample[0],np.size(trainsample[0]))
for i in range(n) :
    PCL.append([])

for i,j in DDD.items() :
    DD[j]=i

print(DDD)
print("Read data.......")
#print("Normalizing data.....")
#trainsample=normalization(trainsample)
train(trainsample,trainlable)
print()
for i in PCL :
    print(len(i))
print("TRAINED SUCESSFULLY")
print("Testing the data")
print("Reading test sample......")
#data = np.genfromtxt("Finial_entry01_ITC_Dataset.csv", dtype=float, delimiter=',', skip_header=1) 
#testsample=data[:,[0,1,59,82,89,94,95,158]]
#testsample=normalization(testsample)
#data = np.genfromtxt("Finial_entry01_ITC_Dataset.csv", dtype=str, delimiter=',', skip_header=1) 
#testlable=data[:,[-1]]
#testlable = testlable.ravel()
ans=test(trainsample,trainlable)
print("Accu=",ans)
"""onec=0
for i in PCL :
    for j in i :
        if j.mass==1 :
            onec+=1
print(onec)"""
for i in PCL :
    print(len(i))