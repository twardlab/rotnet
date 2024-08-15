import numpy as np
import torch
import matplotlib.pyplot as plt
from glob import glob
from os.path import join
from scipy.stats import mode
import csv

from sklearn.metrics import accuracy_score, roc_auc_score
from moment_kernels import *

# NOTE: THIS NOTEBOOK IS SIMPLY A SCRIPT VERSION OF THE train_unet_model_abc_V00.ipynb NOTEBOOK

data_dir = '/nafs/dtward/allen/npz_files/'

class Dataset(torch.utils.data.Dataset):
    def __init__(self,data_dir=data_dir,lr=0,n_per_slice=100,size=64,
                 level='/nafs/dtward/allen/rois/categories.csv'
                ):
        self.data_dir = data_dir
        self.lr = lr
        files = glob(join(data_dir,f'*_lr_{lr}.npz'))
        files.sort()
        files = files
        self.files = files
        self.n_per_slice = n_per_slice
        self.size = size
        self.current_slice = -1
        
        # get a map from the level
        label_mapper = {}
        label_names = {}
        count = 0
        with open(level) as f:
            reader = csv.reader(f)
            for line in reader:                
                if count == 0:
                    headers = line
                    count += 1
                    continue
                labels = line[2]
                labels = labels.replace('\n',' ').replace('[','').replace(']','')
                labels = labels.split(' ')
                #print(labels)
                for l in labels:
                    if l:
                        label_mapper[int(l)] = int(line[0])
                label_names[int(line[0])] = line[1]
                count += 1
        self.label_mapper = label_mapper
        self.label_names = label_names
        
    def __len__(self):
        return self.n_per_slice * len(self.files)
    def __getitem__(self,i):
        slice_number = i//self.n_per_slice
        if slice_number != self.current_slice:
            # load a slice            
            data = np.load(self.files[slice_number])
            self.I = data['I']
            self.L = data['L']
            self.current_slice = slice_number
        # get a region
        rowmin = 0
        rowmax = self.I.shape[1]
        colmin = 0
        colmax = self.I.shape[2]
        
        # cutout
        r = np.random.randint(rowmin,rowmax-self.size)
        c = np.random.randint(colmin,colmax-self.size)
        I_ = self.I[:,r:r+self.size,c:c+self.size]
        L_ = self.L[:,r:r+self.size,c:c+self.size]
        self.L_ = L_
        self.I_ = I_
        L__ = L_.ravel()
        L__ = [self.label_mapper[l] for l in L__]
        L__ = np.reshape(L__,L_.shape)
        self.L__ = L__
        self.L_ = L_
        return I_,L__
        
dataset = Dataset(level='/nafs/dtward/allen/rois/divisions.csv')
dataset_test = Dataset(level='/nafs/dtward/allen/rois/divisions.csv',lr=1)

keys = list(dataset.label_names.keys())
keys.sort()
weights = 1.0 - np.array(['unassigned' in dataset.label_names[k] and dataset.label_names[k] != 'unassigned' for k in keys])

data_loader = torch.utils.data.DataLoader(dataset,batch_size=8,shuffle=False)
data_loader_test = torch.utils.data.DataLoader(dataset_test,batch_size=8,shuffle=False)

# build a quick net
class EqNet(torch.nn.Module):
    def __init__(self,):
        super().__init__()
        
        # downsampling
        k = 3
        p = 1
        c0 = 16
        cin = 501
        cout = len(dataset.label_names)

        self.c1 = ScalarToScalar(in_channels = cin, out_channels=c0, kernel_size=k, padding=p)
        self.p1 = Downsample()
        self.s1 = ScalarSigmoid()

        self.c2 = ScalarToScalar(in_channels = c0, out_channels=c0*2, kernel_size=k, padding=p)
        self.p2 = Downsample()
        self.s2 = ScalarSigmoid()

        self.c3 = ScalarToScalar(in_channels = c0*2, out_channels=c0*4, kernel_size=k, padding=p)
        self.p3 = Downsample()
        self.s3 = ScalarSigmoid()

        self.c4 = ScalarToScalar(in_channels = c0*4, out_channels=c0*8, kernel_size=k, padding=p)
        self.p4 = Downsample()
        self.s4 = ScalarSigmoid()

        self.c5 = ScalarToScalar(in_channels = c0*8, out_channels=c0*16, kernel_size=k, padding=p)
        self.p5 = Downsample()
        self.s5 = ScalarSigmoid()
        
        # upsampling
        self.p5_ = Upsample()
        self.c5_ = ScalarToScalar(in_channels = c0*16+c0*8, out_channels=c0*8, kernel_size=k, padding=p)
        self.s5_ = ScalarSigmoid()

        self.p4_ = Upsample()
        self.c4_ = ScalarToScalar(in_channels = c0*8+c0*4, out_channels=c0*4, kernel_size=k, padding=p)
        self.s4_ = ScalarSigmoid()
        
        self.p3_ = Upsample()
        self.c3_ = ScalarToScalar(in_channels = c0*4+c0*2, out_channels=c0*2, kernel_size=k, padding=p)
        self.s3_ = ScalarSigmoid()

        self.p2_ = Upsample()
        self.c2_ = ScalarToScalar(in_channels = c0*2+c0, out_channels=c0, kernel_size=k, padding=p)
        self.s2_ = ScalarSigmoid()
        
        self.p1_ = Upsample()
        self.c1_ = ScalarToScalar(in_channels = c0+cin, out_channels=cout, kernel_size=k, padding=p)
           
    def forward(self,x):
        x0 = [x] # 64x64
        x = self.s1(self.p1(self.c1(x)))
        x0.append(x) # 32x32
        x = self.s2(self.p2(self.c2(x)))
        x0.append(x) # 16x16
        x = self.s3(self.p3(self.c3(x)))
        x0.append(x) # 8x8
        x = self.s4(self.p4(self.c4(x)))
        x0.append(x) # 4x4
        x = self.s5(self.p5(self.c5(x)))
        # 2x2
        
        # now upsampling
        x = self.p5_(x) # upsample        
        # # concat
        x = torch.concatenate((x,x0.pop()),-3)        
        # # conv
        x = self.c5_(x)
        x = self.s5_(x)
        
        x = self.p4_(x)     
        x = torch.concatenate((x,x0.pop()),-3)                
        x = self.c4_(x)
        x = self.s4_(x)
        
        x = self.p3_(x)     
        x = torch.concatenate((x,x0.pop()),-3)                
        x = self.c3_(x)
        x = self.s3_(x)
        
        x = self.p2_(x)     
        x = torch.concatenate((x,x0.pop()),-3)                
        x = self.c2_(x)
        x = self.s2_(x)
        
        x = self.p1_(x)     
        x = torch.concatenate((x,x0.pop()),-3)                
        x = self.c1_(x)
        
        return x
    
eqnet = EqNet()
optimizer = torch.optim.Adam(eqnet.parameters())
device = 'cuda' if torch.cuda.is_available() else 'cpu'
loss = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights,dtype=torch.float32,device=device))

# get some metrics
def compute_dice(ltruesave,lpredictsave,labels=None):
    '''
    Returns dice score for each structure
    
    TODO
    probably should return weights for averaging across structures, later
    '''
    if labels is None:
        labels = np.unique(ltruesave)
    # for each example
    # look at volume, intersection, and unions
    dice = np.zeros(len(labels))
    dicesum = np.zeros(len(labels))
    count = 0
    for ltrue,lpredict in zip(ltruesave,lpredictsave):
        ntrue_ = []
        npredict_ = []
        nobth_ = []
        lcount = 0
        for l in labels:
            ntrue = np.sum(ltrue==l,(-1,-2))
            npredict = np.sum(lpredict==l,(-1,-2))
            nboth = np.sum((lpredict==l)*(ltrue==l),(-1,-2))
            dice[lcount] += 2*nboth / (ntrue + npredict + 1e-6)
            dicesum[lcount] += nboth
            lcount += 1
            
        count += 1
    dice /= count
    # to average
    weights = dicesum / np.sum(dicesum)
    dicemean = np.sum(dice*weights)
        
        
    
    return dice,dicemean

nepochs = 500
Esave = []
accuracytestsave = []
accuracytrainsave = []
dicetestsave = []
dicetrainsave = []

eqnet = eqnet.to(device)
count = 0
for e in range(nepochs):
    Esave_ = []
    ltruesave = []
    lpredictsave = []
    probsave = []
    for x,l in data_loader:
        x = x.to(device)
        ltruesave.append( l )
        l = l.to(device)
        optimizer.zero_grad()
        
        lhat = eqnet(x)
        
        E = loss(lhat,l[:,0])
        
        E.backward()
        optimizer.step()
        
        Esave_.append(E.item())
        probsave.append( torch.softmax(lhat,-3).clone().detach().cpu() )
        lpredictsave.append(torch.argmax(lhat,-3).clone().detach().cpu())
    
        count += 1
    Esave.append(np.mean(Esave_))
    accuracytrainsave.append( accuracy_score(torch.concatenate(ltruesave).ravel(),torch.concatenate(lpredictsave).ravel()) )
    dicetrain,dicetrainmean = compute_dice(torch.concatenate(ltruesave,0).numpy(),torch.concatenate(lpredictsave,0).numpy(),labels=np.arange(len(dataset.label_names))) 
    dicetrainsave.append(dicetrainmean )
    
    ltruesave = []
    lpredictsave = []
    probsave = []
    with torch.no_grad():
        eqnet.eval()
        for x,l in data_loader_test:
            ltruesave.append( l )
            x = x.to(device)
            lhat = eqnet(x)
            lhat = lhat.clone().detach().cpu()
            probsave.append( torch.softmax(lhat,-3).clone().detach().cpu() )
            lpredictsave.append(torch.argmax(lhat,-3).clone().detach().cpu())
    
            
        # josef is computing
        # haussdorf dice iou and loss
        # NOTE I DIDN'T DO ANYTHING WITH THE TEST SET OR REPORT ANY METRICS OF PERFORMANCE
        eqnet.train()
    
    accuracytestsave.append( accuracy_score(torch.concatenate(ltruesave).ravel(),torch.concatenate(lpredictsave).ravel()) )
    dicetest,dicetestmean = compute_dice(torch.concatenate(ltruesave,0).numpy(),torch.concatenate(lpredictsave,0).numpy(),labels=np.arange(len(dataset.label_names))) 
    dicetestsave.append(dicetestmean )

    # if accuracy improves, save the model
    if e == 0 or accuracytestsave[-1] > np.max(accuracytestsave[:-1]):
        torch.save(eqnet.state_dict(),'best_eqUnetCNN_v01.pth')

torch.save(eqnet.state_dict(),'final_eqUnetCNN_v01.pth')