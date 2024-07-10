
import torch
import numpy as np


class ScalarToScalar(torch.nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size, padding=0, bias=True, padding_mode='zeros'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size    
        self.padding = padding
        self.padding_mode = padding_mode
        if padding_mode == 'zeros': self.padding_mode = 'constant'
        # use kernel size to get x
        r = (kernel_size - 1)//2
        x = torch.arange(-r,r+1)
        X = torch.stack(torch.meshgrid(x,x,indexing='ij'),-1)          
        R = torch.sqrt(torch.sum(X**2,-1))
        Xhat = X/R[...,None]
        Xhat[R==0] = 0        
        rs,inds = torch.unique(R,return_inverse=True)        
        # register buffers, this will allow them to move to devices                
        self.register_buffer('Xhat',Xhat)
        self.register_buffer('inds',inds)
        self.weights = torch.nn.parameter.Parameter(torch.randn(out_channels,in_channels,len(rs))/np.sqrt(3.0*in_channels)) # TODO: use the right normalizatoin
        self.bias = torch.nn.parameter.Parameter(torch.randn(out_channels)/np.sqrt(3.0))        
    def forward(self,x):
        # note this works with size 1 images, as long as padding is 0
        
        # convert the weights into a kernel
        # we reshape from out x in x len(rs)
        # to
        # out x in x kernel_size x kernel_size        
        c = self.weights[...,self.inds]
        self.c = c
                        
        tmp = torch.nn.functional.pad(x,(self.padding,self.padding,self.padding,self.padding),mode=self.padding_mode)                
        return torch.nn.functional.conv2d(tmp,c,self.bias)
        
        
class ScalarToVector(torch.nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size, padding=0, padding_mode='zeros'):
        # with vectors, out channel will be the number of vectors, not the number of components
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size    
        self.padding = padding
        self.padding_mode = padding_mode
        if padding_mode == 'zeros': self.padding_mode = 'constant'
        # use kernel size to get x
        r = (kernel_size - 1)//2
        x = torch.arange(-r,r+1)
        X = torch.stack(torch.meshgrid(x,x,indexing='ij'),-1)          
        R = torch.sqrt(torch.sum(X**2,-1))
        Xhat = X/R[...,None]
        Xhat[R==0] = 0        
        # reshape it to the way I will want to use it
        # it should match out channels on the left
        Xhat = Xhat.permute(-1,0,1)[:,None]
        Xhat = Xhat.repeat((out_channels,1,1,1))
        rs,inds = torch.unique(R,return_inverse=True)        
        # register buffers, this will allow them to move to devices                
        self.register_buffer('Xhat',Xhat)
        
        inds = inds - 1 # we will not use r=0.  the filter will get assigned a different number, but hten multiplied by 0
        inds[inds==-1] = 0        
        self.register_buffer('inds',inds) # don't need a parameter for r=0, but this makes
        
        self.weights = torch.nn.parameter.Parameter(torch.randn(out_channels,in_channels,len(rs)-1)/np.sqrt(3*in_channels)) # TODO: use the right normalizatoin
        
    def forward(self,x):
        # size 1 needs to be a special case, the result is just 0
        # no padding allowed
        # note we assume square
        if x.shape[-1] == 1:
            return torch.zeros(x.shape[0],self.out_channels*2,1,1,dtype=x.dtype,device=x.device)
            
        
        # convert the weights into a kernel
        # we reshape from out x in x len(rs)
        # to
        # out x in x kernel_size x kernel_size          
        
        c = torch.repeat_interleave(self.weights,2,0)[...,self.inds]*self.Xhat
        self.c = c
        
        
        # for somme reason the output is not zero mean, has to do with padding
        # here's a better way
        tmp = torch.nn.functional.pad(x,(self.padding,self.padding,self.padding,self.padding),mode=self.padding_mode)
        return torch.nn.functional.conv2d(tmp,c)
        
def rotate_vector_and_image(x):
    with torch.no_grad():
        tmp = x.rot90(1,(-1,-2))
        tmp2 = tmp.clone()        
        for i in range(tmp.shape[1]//2):
            tmp2[:,i*2] = tmp[:,i*2+1]
            tmp2[:,i*2+1] = -tmp[:,i*2]

    return tmp2

def rotate_vector(x):
    with torch.no_grad():  
        tmp = x.clone()
        tmp2 = x.clone()        
        for i in range(tmp.shape[1]//2):
            tmp2[:,i*2] = tmp[:,i*2+1]
            tmp2[:,i*2+1] = -tmp[:,i*2]

    return tmp2




class VectorToScalar(torch.nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size, padding=0, bias=True, padding_mode='zeros'):
        # with vectors, in channel will be the number of vectors, not the number of components
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size    
        self.padding = padding
        self.padding_mode = padding_mode
        if padding_mode == 'zeros': self.padding_mode = 'constant'
        # use kernel size to get x
        r = (kernel_size - 1)//2
        x = torch.arange(-r,r+1)
        X = torch.stack(torch.meshgrid(x,x,indexing='ij'),-1)          
        R = torch.sqrt(torch.sum(X**2,-1))
        Xhat = X/R[...,None]
        Xhat[R==0] = 0        
        # reshape it to the way I will want to use it
        # it should match out channels on the left
        Xhat = Xhat.permute(-1,0,1)[None]
        Xhat = Xhat.repeat((1,in_channels,1,1))
        rs,inds = torch.unique(R,return_inverse=True)        
        inds = inds - 1
        inds[inds<0] = 0
        # register buffers, this will allow them to move to devices                
        self.register_buffer('Xhat',Xhat)
        self.register_buffer('inds',inds)
        self.weights = torch.nn.parameter.Parameter(torch.randn(out_channels,in_channels,len(rs)-1)/np.sqrt(3*in_channels*2)) # TODO: use the right normalizatoin
        if bias:
            self.bias = torch.nn.parameter.Parameter(torch.randn(out_channels)/np.sqrt(3.0))        
        else:
            self.bias = None
    def forward(self,x):
        # size 1 is a special case, just return 0 + bias
        if x.shape[-1] == 1:
            return torch.zeros(x.shape[0],self.out_channels,1,1,dtype=x.dtype,device=x.device) + self.bias[...,None,None]
        # convert the weights into a kernel
        # we reshape from out x in x len(rs)
        # to
        # out x in x kernel_size x kernel_size             
        c = torch.repeat_interleave(self.weights[...,self.inds],2,1)*self.Xhat                
        self.c = c
        tmp = torch.nn.functional.pad(x,(self.padding,self.padding,self.padding,self.padding),mode=self.padding_mode)                
        return torch.nn.functional.conv2d(tmp,c,self.bias) 
        
        
        
class VectorToVector(torch.nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size, padding=0, padding_mode='zeros'):
        # with vectors, in channel will be the number of vectors, not the number of components
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size    
        self.padding = padding
        self.padding_mode = padding_mode
        if padding_mode == 'zeros': self.padding_mode = 'constant'
        # use kernel size to get x
        r = (kernel_size - 1)//2
        x = torch.arange(-r,r+1)
        X = torch.stack(torch.meshgrid(x,x,indexing='ij'),-1)          
        R = torch.sqrt(torch.sum(X**2,-1))
        Xhat = X/R[...,None]
        Xhat[R==0] = 0        
        # reshape it to the way I will want to use it
        # it should match out channels on the left
        # we need XhatXhat, and identity
        Xhat = Xhat.permute(-1,0,1)
        XhatXhat = Xhat[None,:]*Xhat[:,None]        
        XhatXhat = XhatXhat.repeat((out_channels,in_channels,1,1))
        rs,inds = torch.unique(R,return_inverse=True)        
        indsxx = inds.clone()-1
        indsxx[indsxx==-1] = 0# wlil get multiplied by zero
        # register buffers, this will allow them to move to devices
        indsidentity = inds
        
        identity = torch.eye(2).repeat((out_channels,in_channels))[...,None,None]
        self.register_buffer('XhatXhat',XhatXhat)
        self.register_buffer('identity',identity)
        self.register_buffer('indsxx',indsxx)
        self.register_buffer('indsidentity',indsidentity)
        
        self.weightsxx = torch.nn.parameter.Parameter(torch.randn(out_channels,in_channels,len(rs)-1)/np.sqrt(3*in_channels*2))
        self.weightsidentity = torch.nn.parameter.Parameter(torch.randn(out_channels,in_channels,len(rs))/np.sqrt(3*in_channels*2))
             
    def forward(self,x):
        # convert the weights into a kernel
        # we reshape from out x in x len(rs)
        # to
        # out x in x kernel_size x kernel_size             
        if x.shape[-1] == 1: # special case if the image is size 1
            cidentity = torch.repeat_interleave(torch.repeat_interleave(self.weightsidentity[...,self.indsidentity],2,0),2,1)*self.identity
            return torch.nn.functional.conv2d(x,cidentity)
        else:
            cxx = torch.repeat_interleave(torch.repeat_interleave(self.weightsxx,2,0),2,1)[...,self.indsxx]*self.XhatXhat
            cidentity = torch.repeat_interleave(torch.repeat_interleave(self.weightsidentity,2,0),2,1)[...,self.indsidentity]*self.identity
            c = cxx + cidentity
            self.c = c
            self.cxx = cxx
            self.cidentity = cidentity
            tmp = torch.nn.functional.pad(x,(self.padding,self.padding,self.padding,self.padding),mode=self.padding_mode)                
            return torch.nn.functional.conv2d(tmp,c) # no bias when output is vector
        
class ScalarVectorToScalarVector(torch.nn.Module):
    def __init__(self, in_scalars, in_vectors, out_scalars, out_vectors, kernel_size, padding=0, bias=True, padding_mode='zeros'):
        super().__init__()
        
        self.in_scalars = in_scalars
        self.in_vectors = in_vectors
        self.out_scalars = out_scalars
        self.out_vectors = out_vectors
        self.padding_mode = padding_mode
        if padding_mode == 'zeros': self.padding_mode = 'constant'
        
        if in_scalars > 0 and out_scalars > 0:
            self.ss = ScalarToScalar(in_scalars, out_scalars, kernel_size, padding, bias, padding_mode)
        if in_scalars > 0 and out_vectors > 0:
            self.sv = ScalarToVector(in_scalars, out_vectors, kernel_size, padding, padding_mode)
        if in_vectors > 0 and out_scalars > 0:
            self.vs = VectorToScalar(in_scalars, out_scalars, kernel_size, padding, bias, padding_mode)
        if in_vectors > 0 and out_vectors > 0:
            self.vv = VectorToVector(in_scalars, out_vectors, kernel_size, padding, padding_mode)
    def forward(self,x):
        
        outs = torch.zeros((x.shape[0],self.out_scalars,x.shape[2],x.shape[3]),device=x.device,dtype=x.dtype)
        outv = torch.zeros((x.shape[0],self.out_vectors*2,x.shape[2],x.shape[3]),device=x.device,dtype=x.dtype)
        #print(outs.shape,outv.shape)
        if self.in_scalars > 0 and self.out_scalars > 0:
            outs = outs + self.ss( x[:,:self.in_scalars])
        if self.in_scalars > 0 and self.out_vectors > 0:
            outv = outv + self.sv( x[:,:self.in_scalars])
        if self.in_vectors > 0 and self.out_scalars > 0:
            outs = outs + self.vs(x[:,self.in_scalars:])
        if self.in_vectors > 0 and self.out_vectors > 0:
            outv = outv + self.vv(x[:,self.in_scalars:])        
        #print(outs.shape,outv.shape)
        
        return torch.concatenate(  ( outs, outv )  , dim=-3)
    
    
class Downsample(torch.nn.Module):
    def __init__(self):
        super().__init__()
        pass
    def forward(self,x):
        # downsample on the last two dimensions by a factor of 2
        # if it is even, we average
        # if it is odd we skip
        if not x.shape[-1]%2: # if even
            x = (x[...,0::2] + x[...,1::2])*0.5
        else:
            x = x[...,0::2]
        
        if not x.shape[-2]%2: # if even
            x = (x[...,0::2,:] + x[...,1::2,:])*0.5
        else:
            x = x[...,0::2,:]
        
        return x
            
    
class Upsample(torch.nn.Module):
    def __init__(self):
        super().__init__()        
    def forward(self,x,roweven=True,coleven=True):
        
        if coleven:
            x = torch.repeat_interleave(x,2,dim=-1)
        else:
            # if odd we insert zeros
            x = (torch.repeat_interleave(x,2,dim=-1) * (1-torch.arange(2*x.shape[-1])%2))[...,:-1]
        
        if roweven:
            x = torch.repeat_interleave(x,2,dim=-2)
        else:
            x = (torch.repeat_interleave(x,2,dim=-2) * (1-torch.arange(2*x.shape[-2])%2)[...,None])[...,:-1,:]
        return x

    
    
# as before, the sigmoid is causing a problem that leads to nans
# perhaps because the sqrt has an infinite slope at x=0?
class VectorSigmoid(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self,x):
        #return torch.relu(x)
        #return torch.abs(x)
        # the vector has some multiple of 2 chanels
        
        x2 = x**2
        l2 = x2[:,0::2] + x2[:,1::2] + 1e-6
        #l2r = torch.repeat_interleave(l2,2,dim=1)
        #return x * l2r / (l2r + 1.0)
        #return x / torch.sqrt((l2r + 1.0))
        #return x*torch.relu(l2r-1)/l2r
        l = torch.sqrt(l2)
        # now I have the length of each vector
        lr = torch.repeat_interleave(l,2,dim=1)
        # now it is repeated
        return x*torch.relu((lr-1.0))/lr
        
        #return torch.relu(x)
        
class ScalarSigmoid(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self,x):
        #l = torch.sqrt(x**2 + 1e-5)
        #return x*torch.relu((l-1.0))/l
        return torch.relu(x)
        
class ScalarVectorSigmoid(torch.nn.Module):
    def __init__(self,n_scalars):
        super().__init__()
        self.n_scalars = n_scalars
        self.s = ScalarSigmoid()
        self.v = VectorSigmoid()
    def forward(self,x):
        return torch.concatenate((self.s(x[:,:self.n_scalars]), self.v(x[:,self.n_scalars:])),-3)
        
        
class ScalarBatchnorm(torch.nn.Module):
    def __init__(self,n):
        super().__init__()
        self.b = torch.nn.BatchNorm2d(n)
    def forward(self,x):
        return self.b(x)
        
        
class VectorBatchnorm(torch.nn.Module):
    def __init__(self,n):
        super().__init__()
        self.b = torch.nn.BatchNorm2d(n)
    def forward(self,x):                
        magnitude2 = x[:,0::2]**2 + x[:,1::2]**2 + 1e-6
        logmagnitude2 = torch.log(magnitude2)
        scaledlogmagnitude2 = self.b(logmagnitude2)
        
        return x * torch.repeat_interleave((  (scaledlogmagnitude2 - logmagnitude2)*0.5 ).exp(),2,dim=1)

    
class ScalarVectorBatchnorm(torch.nn.Module):
    def __init__(self,nscalar,nvector):
        super().__init__()
        self.nscalar = nscalar
        self.nvector = nvector
        self.bs = ScalarBatchnorm(nscalar)
        self.bv = VectorBatchnorm(nvector)
    def forward(self,x):
        return torch.concatenate( (self.bs(x[:,:self.nscalar]),self.bv(x[:,self.nscalar:])) , 1)