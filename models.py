import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from IPython.display import clear_output, display

'''
This module provides the rotational equivariant network with vector fields.

We interpret feature maps in blocks of size 3.  The first component is a scalar map, and the second two are vector components.

Inputs and outputs are supposed to be scalars.

Convolutions include linear to and from scalars and vectors (4 kinds) that all respect rotation equivariance.


'''

class ExtractScalar(torch.nn.Module):
    def __init__(self,):
        super().__init__()
    def forward(self,x):
        return x[...,::3,:,:]

class ToVector(torch.nn.Module):
    '''
    This module takes a set of N scalar images, 
    and converts them to a set of N scalars, and N 2D 0 vectors
    for a total of 3N dimensions
        
    
    Todo
    ----
    Make this work with or without batch dims
    '''
    def __init__(self):
        super().__init__()

    def forward(self,x):        
        # x is size
        # B x N x R x C
        z = torch.zeros_like(x)
        xz = torch.stack((x,z,z),-3)
        if x.ndim == 3:
            x = xz.reshape(3*x.shape[-3],x.shape[-2],x.shape[-1])
        elif x.ndim == 4:
            x = xz.reshape(x.shape[-4],3*x.shape[-3],x.shape[-2],x.shape[-1])
        else:
            raise Exception(f'x should have dim 3 or 4 but has dim {x.ndim}')
        return x

class Conv2DRot3x3(torch.nn.Module):
    '''
    This module provides our most important functionality.

    We consider matrix valued kernels with special constraints.

    Note that with a stride of 2, we lose rotation  equivariance unless the higher resolution size is odd.

    Instead we will use pooling when even.
    
    TODO
    ----
    Update bias correctly.  No bias for vector fields.  Kerently I never use bias because it is always followed by batchnorm.

    TODO
    ----
    Save k so it doesn't need to be recomputed during eval


    '''
    def __init__(self,in_channels,out_channels,stride=1,padding=1,bias=True):
        super().__init__()
        if (in_channels//3 != in_channels/3) or (out_channels//3 != out_channels/3):
            raise Exception('channels must be divisible by 3')
            
        # in and out channels must be divisible by 3
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_blocks = in_channels//3
        self.out_blocks = out_channels//3
        self.stride = stride
        self.padding = padding
        self.use_bias = bias
        
        k = 1/np.sqrt(9*in_channels)
        self.s = torch.nn.parameter.Parameter(torch.randn(3,self.out_blocks,self.in_blocks)*k) # scalar
        self.r0 = torch.nn.parameter.Parameter(torch.randn(2,self.out_blocks,self.in_blocks)*k) # row
        self.r1 = torch.nn.parameter.Parameter(torch.randn(2,self.out_blocks,self.in_blocks)*k) # row
        self.c0 = torch.nn.parameter.Parameter(torch.randn(2,self.out_blocks,self.in_blocks)*k) # col
        self.c1 = torch.nn.parameter.Parameter(torch.randn(2,self.out_blocks,self.in_blocks)*k) # col
        self.a = torch.nn.parameter.Parameter(torch.randn(3,self.out_blocks,self.in_blocks)*k) # matrix identity
        self.b00 = torch.nn.parameter.Parameter(torch.randn(2,self.out_blocks,self.in_blocks)*k) # matrix outer
        self.b01 = torch.nn.parameter.Parameter(torch.randn(2,self.out_blocks,self.in_blocks)*k) # matrix outer
        self.b10 = torch.nn.parameter.Parameter(torch.randn(2,self.out_blocks,self.in_blocks)*k) # matrix outer
        self.b11 = torch.nn.parameter.Parameter(torch.randn(2,self.out_blocks,self.in_blocks)*k) # matrix outer
        
        
        # TODO, update bias so it is only there for scalars (done)
        if self.use_bias:
            self.bias = torch.nn.parameter.Parameter(torch.randn(self.out_channels))
        else:
            self.register_buffer('bias',torch.zeros(self.out_channels))            
        
        
        # save a bunch of stuff so I don't have to compute it every time        
        self.register_buffer('x',torch.arange(-1.0,2.0))
        self.register_buffer('X',torch.stack(torch.meshgrid(self.x,self.x,indexing='ij'),0))
        self.register_buffer('R2',torch.sum(self.X**2,0))
                
        
        
        R90 = torch.tensor([[0.0,-1.0],[1.0,0.0]])        
        self.register_buffer('R90X',(R90@self.X.permute(1,2,0)[...,None])[...,0].permute(-1,0,1) )
        
        self.register_buffer('eye',torch.eye(2)[:,:,None,None,None,None])
        self.register_buffer('XX',self.X[:,None,None,None]*self.X[None,:,None,None])
        self.register_buffer('XR90X',self.X[:,None,None,None]*self.R90X[None,:,None,None])
        self.register_buffer('R90XX',self.R90X[:,None,None,None]*self.X[None,:,None,None])
        self.register_buffer('R90XR90X',self.R90X[:,None,None,None]*self.R90X[None,:,None,None])
        self.register_buffer('R20',(self.R2==0))
        self.register_buffer('R21',(self.R2==1))
        self.register_buffer('R22',(self.R2==2))

        self.register_buffer('k',None)
        
    def forward(self,x):
        if self.training or self.k is None:
            # assemble the kernel from the parameters
            # make it
            # 3 x 3 x out_blocks x in_blocks x 3 x 3
            # I will then reshape it to
            # 3 * out_blocks x 3 * in_blocks x 3 x 3
            # okay here's a start        
            
            # scalar part
            s = (self.s[0,...,None,None]*self.R20 + self.s[1,...,None,None]*self.R21 + self.s[2,...,None,None]*self.R22)[None,None]        
            # s is size 1 x 1 x out_blocks x in_blocks x 3 x 3
            
            # column part
            c = ( (self.c0[0,...,None,None]*self.R21 + self.c0[1,...,None,None]*self.R22)*self.X[:,None,None,None] 
                + (self.c1[0,...,None,None]*self.R21 + self.c1[1,...,None,None]*self.R22)*self.R90X[:,None,None,None])
            # size is 2 x 1 x out_blocks x in_blocks x 3 x 3            
            # row part
            r = ( (self.r0[0,...,None,None]*self.R21 + self.r0[1,...,None,None]*self.R22)*self.X[None,:,None,None]
                + (self.r1[0,...,None,None]*self.R21+ self.r1[1,...,None,None]*self.R22)*self.R90X[None,:,None,None])
            
            # matrix part proportional to identity
            a = (self.a[0,...,None,None]*self.R20 + self.a[1,...,None,None]*self.R21+ self.a[2,...,None,None]*self.R22)*self.eye
            # matrix part proportional to xx^T
            b00 = (self.b00[0,...,None,None]*self.R21 + self.b00[1,...,None,None]*self.R22)*self.XX
            b01 = (self.b01[0,...,None,None]*self.R21 + self.b01[1,...,None,None]*self.R22)*self.XR90X
            b10 = (self.b10[0,...,None,None]*self.R21 + self.b10[1,...,None,None]*self.R22)*self.R90XX
            b11 = (self.b11[0,...,None,None]*self.R21 + self.b11[1,...,None,None]*self.R22)*self.R90XR90X
            # put them together into a matrix
            m = a+b00+b01+b10+b11
            
            # stick all the components to gether into a big matrix
            kblock = torch.concatenate( (torch.concatenate((s,r),1), torch.concatenate((c,m),1)), 0)        
            kblockp = kblock.permute(2,0,3,1,4,5)  # so which one is right? I think it is this one, I double checked everything        
            k = kblockp.reshape(self.out_channels,self.in_channels,3,3)

            # todo, save k and don't recompute it during evaluation mode
            self.k = k
        else: # if in eval mode we can speed up
            k = self.k
        

        out = torch.nn.functional.conv2d(x,k,self.bias,self.stride,self.padding)
        return out
    
class Conv2DRot(torch.nn.Module):
    def __init__(self, in_channels,out_channels,kernel_size,stride=1,padding=None,bias=True):
        super().__init__()
        # check kernel size
        if not kernel_size%2:
            raise Exception(f'Only odd kernel sizes supported, but you input {kernel_size}')
        
        # check channels
        # TODO if channels is not divisible by 3, add extra scalar channels
        if (in_channels//3 != in_channels/3) or (out_channels//3 != out_channels/3):
            raise Exception('channels must be divisible by 3')
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.in_blocks = in_channels//3
        self.out_blocks = out_channels//3
        self.stride = stride
        if padding is None:
            padding = (kernel_size-1)//2 
        self.padding = padding
        self.use_bias = bias
        
        

        # set up a bunch of other quantities we'll use for calculations
        # first the "domain" of the kernel, x
        self.r = (kernel_size-1)/2
        self.register_buffer('x',torch.arange(-self.r*1.0,self.r+1))
        self.register_buffer('X',torch.stack(torch.meshgrid(self.x,self.x,indexing='ij'),0))
        self.register_buffer('R2',torch.sum(self.X**2,0))
        
        # now in 2D we will work with 90 degree rotations to form a basis for 2x2 matrices
        R90 = torch.tensor([[0.0,-1.0],[1.0,0.0]])        
        self.register_buffer('R90X',(R90@self.X.permute(1,2,0)[...,None])[...,0].permute(-1,0,1) )
        self.register_buffer('eye',torch.eye(2)[:,:,None,None,None,None])
        self.register_buffer('XX',self.X[:,None,None,None]*self.X[None,:,None,None])
        self.register_buffer('XR90X',self.X[:,None,None,None]*self.R90X[None,:,None,None])
        self.register_buffer('R90XX',self.R90X[:,None,None,None]*self.X[None,:,None,None])
        self.register_buffer('R90XR90X',self.R90X[:,None,None,None]*self.R90X[None,:,None,None])

        # now our kernels will be functions of distnace |c|, so we will use indicator functions
        # TODO, use some kind of a list so I can support arbitrary sizes
        self.register_buffer('R20', (self.R2==0))
        if kernel_size > 1:
            # for 3x3 kernels or bigger we will have |x|^2 =1 or 2
            self.register_buffer('R21',(self.R2==1))
            self.register_buffer('R22',(self.R2==2))
        if kernel_size > 3:
            # for 5x5 kernels or bigger we will have more
            # 0^2 + 0^2 = 0 (done above)
            # 0^2 + 1^2 = 1 (done above)
            # 0^2 + 2^2 = 4
            # 1^2 + 1^2 = 2 (done above)
            # 1^2 + 2^2 = 5
            # 2^2 + 2^2 = 8
            self.register_buffer('R24',(self.R2==4))
            self.register_buffer('R25',(self.R2==5))
            self.register_buffer('R28',(self.R2==8))

        # initialize a couple things to zero
        self.register_buffer('czero',torch.zeros((2,1,self.out_blocks,self.in_blocks,kernel_size,kernel_size)))
        self.register_buffer('rzero',torch.zeros((1,2,self.out_blocks,self.in_blocks,kernel_size,kernel_size)))
        # we will save the kernel for use in eval mode, 
        # # to be updated when we call forward in training mode
        self.register_buffer('k',None)


        # initializer as gaussian with a certain variance
        # note that 1x1 kernels will only have s and a, others will just be 0
        if self.kernel_size == 1:
            self.nr = 1
        elif self.kernel_size == 3:
            self.nr = 3
        elif self.kernel_size == 5:
            self.nr = 6
        
        k = 1/np.sqrt(9*in_channels)
        self.s = torch.nn.parameter.Parameter(torch.randn(self.nr,self.out_blocks,self.in_blocks)*k) # scalar
        if kernel_size > 1:
            self.r0 = torch.nn.parameter.Parameter(torch.randn(self.nr-1,self.out_blocks,self.in_blocks)*k) # row
            self.r1 = torch.nn.parameter.Parameter(torch.randn(self.nr-1,self.out_blocks,self.in_blocks)*k) # row
            self.c0 = torch.nn.parameter.Parameter(torch.randn(self.nr-1,self.out_blocks,self.in_blocks)*k) # col
            self.c1 = torch.nn.parameter.Parameter(torch.randn(self.nr-1,self.out_blocks,self.in_blocks)*k) # col
        self.a = torch.nn.parameter.Parameter(torch.randn(self.nr,self.out_blocks,self.in_blocks)*k) # matrix identity
        if kernel_size > 1:
            self.b00 = torch.nn.parameter.Parameter(torch.randn(self.nr-1,self.out_blocks,self.in_blocks)*k) # matrix outer
            self.b01 = torch.nn.parameter.Parameter(torch.randn(self.nr-1,self.out_blocks,self.in_blocks)*k) # matrix outer
            self.b10 = torch.nn.parameter.Parameter(torch.randn(self.nr-1,self.out_blocks,self.in_blocks)*k) # matrix outer
            self.b11 = torch.nn.parameter.Parameter(torch.randn(self.nr-1,self.out_blocks,self.in_blocks)*k) # matrix outer
        
        # set up bias
        # TODO, only use bias for scalar channels
        if self.use_bias:
            self.tv = ToVector()
            self.bias = torch.nn.parameter.Parameter(torch.randn(self.out_blocks))
        else:
            self.register_buffer('bias', torch.zeros(self.out_channels))
    
    def forward(self,x):
        if self.training or self.k is None:
            # we have to build the convolution kernel from the parameters

            # scalar part            
            s = self.s[0,...,None,None]*self.R20
            if self.kernel_size > 1:
                s = s + self.s[1,...,None,None]*self.R21 + self.s[2,...,None,None]*self.R22
            if self.kernel_size > 3:
                s = s + self.s[3,...,None,None]*self.R24 + self.s[4,...,None,None]*self.R25 + self.s[5,...,None,None]*self.R28
            s = s[None,None] # 1x1 matrix on first two dimensions

            # column part
            # we need to initialize the c's as zeros of the correct size
            # TODO make this work with or without batch dimension
            c0 = self.czero            
            c1 = self.czero
            if self.kernel_size > 1:
                c0 = c0 + (self.c0[0,...,None,None]*self.R21 + self.c0[1,...,None,None]*self.R22)
                c1 = c1 + (self.c1[0,...,None,None]*self.R21 + self.c1[1,...,None,None]*self.R22)
            if self.kernel_size > 3:
                c0 = c0 + (self.c0[2,...,None,None]*self.R24 + self.c0[3,...,None,None]*self.R25 + self.c0[4,...,None,None]*self.R28)
                c1 = c1 + (self.c1[2,...,None,None]*self.R24 + self.c1[3,...,None,None]*self.R25 + self.c1[4,...,None,None]*self.R28)
            c0 = c0 * self.X[:,None,None,None]
            c1 = c1 * self.R90X[:,None,None,None]
            c = c0 + c1

            # row part
            r0 = self.rzero
            r1 = self.rzero            
            if self.kernel_size > 1:
                r0 = r0 + (self.r0[0,...,None,None]*self.R21 + self.r0[1,...,None,None]*self.R22)
                r1 = r1 + (self.r1[0,...,None,None]*self.R21 + self.r1[1,...,None,None]*self.R22)
            if self.kernel_size > 3:
                r0 = r0 + (self.r0[2,...,None,None]*self.R24 + self.r0[3,...,None,None]*self.R25 + self.r0[4,...,None,None]*self.R28)
                r1 = r1 + (self.r1[2,...,None,None]*self.R24 + self.r1[3,...,None,None]*self.R25 + self.r1[4,...,None,None]*self.R28)           
            r0 = r0 * self.X[None,:,None,None]            
            r1 = r1 * self.R90X[None,:,None,None]
            r = r0 + r1

            # matrix part proportional to identity
            a = self.a[0,...,None,None]*self.R20 
            if self.kernel_size > 1:
                a = a + self.a[1,...,None,None]*self.R21 + self.a[2,...,None,None]*self.R22
            if self.kernel_size > 3:
                a = a + self.a[3,...,None,None]*self.R24 + self.a[4,...,None,None]*self.R25 + self.a[5,...,None,None]*self.R25
            a = a*self.eye
            

            # matrix part proportional to xx^T
            b00 = 0.0
            b01 = 0.0
            b10 = 0.0
            b11 = 0.0
            if self.kernel_size > 1:
                b00 = self.b00[0,...,None,None]*self.R21 + self.b00[1,...,None,None]*self.R22
                b01 = self.b01[0,...,None,None]*self.R21 + self.b01[1,...,None,None]*self.R22
                b10 = self.b10[0,...,None,None]*self.R21 + self.b10[1,...,None,None]*self.R22
                b11 = self.b11[0,...,None,None]*self.R21 + self.b11[1,...,None,None]*self.R22
            if self.kernel_size > 3:
                b00 = self.b00[2,...,None,None]*self.R24 + self.b00[3,...,None,None]*self.R25 + self.b00[4,...,None,None]*self.R28
                b01 = self.b01[2,...,None,None]*self.R24 + self.b01[3,...,None,None]*self.R25 + self.b01[4,...,None,None]*self.R28
                b10 = self.b10[2,...,None,None]*self.R24 + self.b10[3,...,None,None]*self.R25 + self.b10[4,...,None,None]*self.R28
                b11 = self.b11[2,...,None,None]*self.R24 + self.b11[3,...,None,None]*self.R25 + self.b11[4,...,None,None]*self.R28
            b00 = b00 * self.XX
            b01 = b01 * self.XR90X
            b10 = b10 * self.R90XX
            b11 = b11 * self.R90XR90X
            # add them all into a matrix
            m = a+b00+b01+b10+b11
            

            # build the convolution block
            kblock = torch.cat( (torch.cat((s,r),1), torch.cat((c,m),1)), 0)                  
            kblockp = kblock.permute(2,0,3,1,4,5)  # so which one is right? I think it is this one, I double checked everything        
            k = kblockp.reshape(self.out_channels,self.in_channels,self.kernel_size,self.kernel_size)
            self.k = k
            
        else: # if in eval mode, speed up by reusing k
            k = self.k
        if self.use_bias:
            bias = self.tv(self.bias[...,None,None])[...,0,0]
        else:
            bias = self.bias
        out = torch.nn.functional.conv2d(x,k,bias,self.stride,self.padding)
        return out




class BatchNormRot(torch.nn.Module):
    '''
    Make sure that we respect the structure in blocks of 3.
    Use the same scale factor for the last two of 3
    
    Note: this mean shift breaks the invariance.  We need to account for this.
    
    one more thing to do.  they say something about biased versus unbiased variance.  I want to make sure it is consistent.
    
    '''
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        if self.num_features%3 != 0:
            raise Exception('must be multiple of 3')
        
        self.register_buffer('running_mean', torch.zeros(num_features) )
        self.register_buffer('running_var', torch.ones(num_features) )
        
        
        # block diagonal
        groups = torch.eye(num_features)
        for i in range(0,num_features,3):
            groups[i+1:i+3,i+1:i+3] = 0.5 # I think 0.5, not 0.25
        self.register_buffer('groups', groups) 
        
        # for a we need two numbers for each block
        ia = []        
        for i in range(0,num_features//3*2,2):
            ia.append(i)
            ia.append(i+1)
            ia.append(i+1)
        self.register_buffer('ia',torch.tensor(ia))
        
        # for bias
        tmp = torch.zeros(num_features, num_features//3)
        count = 0
        for i in range(0,num_features,3):
            tmp[i,count] = 1
            count += 1
        self.register_buffer('bmat',tmp)
        
        # get vector components
        vector_ind = torch.arange(num_features)%3 > 0
        self.register_buffer('vecind',vector_ind)
        scalar_ind = torch.arange(num_features)%3 == 0
        self.register_buffer('scalarind',scalar_ind)
        
        
        # a is for scalar and vector (2 out of 3 in a block)
        # b is for scalar only (1 out of 3 in a block)
        self.a = torch.nn.parameter.Parameter(torch.ones(num_features//3*2))
        self.b = torch.nn.parameter.Parameter(torch.zeros(num_features//3))
                             
    def forward(self,x):
        # if training get the mean and update running mean
        if self.training:
            # get the mean, respecting block structure
            mean = self.groups@torch.mean(x,dim=(-1,-2,-4) )
            # for mean remove any shift on the vector part
            mean = mean*self.scalarind
            # get the variance, respecting block structure
            var = self.groups@torch.mean(x**2,dim=(-1,-2,-4)) - mean**2

            with torch.no_grad():
                # update running mean and var            
                # current value is *0.1, old value is *0.9
                self.running_mean = self.running_mean*(1-self.momentum) + mean*(self.momentum)
                self.running_var = self.running_var*(1-self.momentum) + var*(self.momentum)
        
        else: # if in eval mode, use the running mean
            mean = self.running_mean
            var = self.running_var
        
        # normalize                
        # note self.a is expanded into blocks with indexing
        # self.b is expanded with matrix multiplication
        x = (x - mean[...,None,None]) / torch.sqrt(var[...,None,None] + self.eps )*self.a[self.ia][...,None,None] + (self.bmat@self.b)[...,None,None]
            
        
        return x
                             





class SigmoidRot(torch.nn.Module):
    def __init__(self,ch=None,ep=1):
        super().__init__()
        self.ch = ch
        if ch is not None:
            blocks = torch.block_diag( *[torch.eye(1),torch.ones(2,2,)/2]*(ch//3) )
            self.register_buffer('blocks',blocks)
        else:
            self.register_buffer('blocks',None)
        self.ep = ep
        

    def forward(self,x):
        mag = x**2
        # what if I average it across blocks
        #sh = x.shape[-3]
        #blocks = ( (torch.arange(sh)[None,:] < torch.arange(3,sh+3,3)[:,None])*(torch.arange(sh)[None,:] >= torch.arange(0,sh+0,3)[:,None]) ).to(dtype=x.dtype,device=x.device)
        #blocks = ( (torch.arange(sh)[None,:] < torch.arange(3,sh+3,3)[:,None])*(torch.arange(sh)[None,:] >= torch.arange(0,sh+0,3)[:,None]) ).to(dtype=x.dtype,device=x.device)
        #print(blocks)
        #blocks = (blocks.T@blocks)/3
        #print(blocks)
        
        #print(blocks)
        #print(mag.transpose(-3,-1)[...,None].shape)
        if self.blocks is None or self.blocks.shape[0] != x.shape[-3]:
            blocks = torch.block_diag( *[torch.eye(1,device=x.device,dtype=x.dtype),torch.ones(2,2,device=x.device,dtype=x.dtype)/2]*(x.shape[-3]//3) )
            self.register_buffer('blocks',blocks)
        mag = (self.blocks@(mag.transpose(-3,-1)[...,None]))[...,0]
        mag = mag.transpose(-3,-1)
        
        #x = x * (mag / (1+mag))
        #x = x * (mag / (4+mag))
        #x = x * (mag**2 / (1+mag**2))
        x = x * (mag / (self.ep+mag)**2)
        # in my simulation, about 60% have magnitude > 1    
        #mag = torch.sqrt(mag)
        #mag1 = (mag - 1.1).clip(min=0)# this should be about 50%
        # above was giving problems
        #mag1 = (mag - 0.5).clip(0)
        #x = x *(mag1/(mag + 1e-5))
    
        return x
    

class ZeroZero(torch.nn.Module):
    ''' This class just extracts the 0,0 component on the last two dimensions'''
    def __init__(self):
        super().__init__()
    def forward(self,x):
        return x[...,0,0]
    

class Down2Rot(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        if x.shape[-1]%2 == 0:
            # if even average
            x = 0.5*(x[...,0::2] + x[...,1::2])
        else:
            # if odd, subsample
            x = x[...,0::2]
        if x.shape[-2]%2 == 0:
            # if even average
            x = 0.5*(x[...,0::2,:] + x[...,1::2,:])
        else:
            # if odd subsample
            x = x[...,0::2,:]
        return x
    
class GlobalAvg(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        return torch.mean(x,(-1,-2))
    


class RotNet18(torch.nn.Module):
    '''
    18 layer resnet with 4 scales of two repeats each.

    This is to match the medmnist evaluation
    '''
    def __init__(self,n0=63,n1=10,kernel_size=3):
        super().__init__()
        self.n0 = n0
        self.n1 = n1

        if kernel_size==3:
            padding = 1
        elif kernel_size == 5:
            padding = 2
        

        self.toblock = ToVector()
        

        # first layer
        self.c0 = Conv2DRot(9,n0,kernel_size,1,padding,bias=False) # 32x32
        self.b0 = BatchNormRot(n0)
        self.s0 = SigmoidRot(n0)


        # now a set of 2, at full res
        self.c1 = Conv2DRot(n0,n0,kernel_size,1,padding,bias=False) # 32x32
        self.b1 = BatchNormRot(n0)
        self.s1 = SigmoidRot(n0)
        self.c1a = Conv2DRot(n0,n0,kernel_size,1,padding,bias=False) # 32x32
        self.b1a = BatchNormRot(n0)
        self.s1a = SigmoidRot(n0)

        self.c1_ = Conv2DRot(n0,n0,kernel_size,1,padding,bias=False) # 32x32
        self.b1_ = BatchNormRot(n0)
        self.s1_ = SigmoidRot(n0)
        self.c1a_ = Conv2DRot(n0,n0,kernel_size,1,padding,bias=False) # 32x32
        self.b1a_ = BatchNormRot(n0)
        self.s1a_ = SigmoidRot(n0)

        # now a set of 2, at half res
        self.c2 = Conv2DRot(n0,n0*2,kernel_size,1,padding,bias=False)
        self.b2 = BatchNormRot(n0*2)
        self.s2 = SigmoidRot(n0*2)        
        self.c2a = Conv2DRot(n0*2,n0*2,kernel_size,1,padding,bias=False)
        self.b2a = BatchNormRot(n0*2)
        self.s2a = SigmoidRot(n0*2)
        self.d2 = Down2Rot()

        self.c2_ = Conv2DRot(n0*2,n0*2,kernel_size,1,padding,bias=False)
        self.b2_ = BatchNormRot(n0*2)
        self.s2_ = SigmoidRot(n0*2)        
        self.c2a_ = Conv2DRot(n0*2,n0*2,kernel_size,1,padding,bias=False)
        self.b2a_ = BatchNormRot(n0*2)
        self.s2a_ = SigmoidRot(n0*2)

        # now a set of 2, at quarter res
        self.c3 = Conv2DRot(n0*2,n0*4,kernel_size,1,padding,bias=False)
        self.b3 = BatchNormRot(n0*4)
        self.s3 = SigmoidRot(n0*4)        
        self.c3a = Conv2DRot(n0*4,n0*4,kernel_size,1,padding,bias=False)
        self.b3a = BatchNormRot(n0*4)
        self.s3a = SigmoidRot(n0*4)
        self.d3 = Down2Rot()

        self.c3_ = Conv2DRot(n0*4,n0*4,kernel_size,1,padding,bias=False)
        self.b3_ = BatchNormRot(n0*4)
        self.s3_ = SigmoidRot(n0*4)        
        self.c3a_ = Conv2DRot(n0*4,n0*4,kernel_size,1,padding,bias=False)
        self.b3a_ = BatchNormRot(n0*4)
        self.s3a_ = SigmoidRot(n0*4)


        # now a set of 2, at eighth res
        self.c4 = Conv2DRot(n0*4,n0*8,kernel_size,1,padding,bias=False)
        self.b4 = BatchNormRot(n0*8)
        self.s4 = SigmoidRot(n0*8)        
        self.c4a = Conv2DRot(n0*8,n0*8,kernel_size,1,padding,bias=False)
        self.b4a = BatchNormRot(n0*8)
        self.s4a = SigmoidRot(n0*8)
        self.d4 = Down2Rot()

        self.c4_ = Conv2DRot(n0*8,n0*8,kernel_size,1,padding,bias=False)
        self.b4_ = BatchNormRot(n0*8)
        self.s4_ = SigmoidRot(n0*8)        
        self.c4a_ = Conv2DRot(n0*8,n0*8,kernel_size,1,padding,bias=False)
        self.b4a_ = BatchNormRot(n0*8)
        self.s4a_ = SigmoidRot(n0*8)

        # finally, extract scalar and apply linear map
        self.extract_scalar = ExtractScalar()
        self.global_avg = GlobalAvg()
        self.linear = torch.nn.Linear(n0//3*8,n1)



       

    def forward(self,x):
        # if grayscale, map to RGB
        if x.shape[-3] == 1:
            x = torch.cat((x,x,x),-3)
        
        # convert to scalars in block form
        x = self.toblock(x)
        
        # first block
        x = self.s0(self.b0(self.c0(x)))
        

        # second block, two repeats at full resolution
        x0 = x.clone()
        x = self.s1(self.b1(self.c1(x)))        
        x = self.b1a(self.c1a(x))        
        x = x + x0
        x = self.s1a(x)        
        
        
        x0 = x.clone()
        x = self.s1_(self.b1_(self.c1_(x)))        
        x = self.b1a_(self.c1a_(x))
        x = x + x0
        x = self.s1a_(x)        


        # third block, two repeats at half resolution       
        x0 = x.clone()
        x = self.s2(self.b2(self.c2(x)))
        x = self.b2a(self.c2a(x))
        x = self.d2(x + x0.repeat(1,2,1,1))
        x = self.s2(x)

        x0 = x.clone()
        x = self.s2_(self.b2_(self.c2_(x)))
        x = self.b2a_(self.c2a_(x))
        x = x + x0
        x = self.s2a_(x)


        # fourth block, two repeats at quarter resolution       
        x0 = x.clone()
        x = self.s3(self.b3(self.c3(x)))
        x = self.b3a(self.c3a(x))
        x = self.d3(x + x0.repeat(1,2,1,1))
        x = self.s3(x)

        x0 = x.clone()
        x = self.s3_(self.b3_(self.c3_(x)))
        x = self.b3a_(self.c3a_(x))
        x = x + x0
        x = self.s3a_(x)


        # fifth block, two repeats at eighth resolution  
        x0 = x.clone()
        x = self.s4(self.b4(self.c4(x)))
        x = self.b4a(self.c4a(x))
        x = self.d4(x + x0.repeat(1,2,1,1))
        x = self.s4(x)

        x0 = x.clone()
        x = self.s4_(self.b4_(self.c4_(x)))
        x = self.b4a_(self.c4a_(x))
        x = x + x0
        x = self.s4a_(x)

        # finally extract scalar and apply one linear map
        x = self.extract_scalar(x)
        x = self.global_avg(x)
        x = self.linear(x)

        return x

        







class ResNet18(torch.nn.Module):
    '''
    18 layer res net as in medmnist evaluation.
    
    
    '''
    def __init__(self,n0=64,n1=10):
        super().__init__()
        # note in the resnet paper, downsampling is done as the first block at a scale, not the last
        
        # follow the resnet idea
        # a first conv layer
        
        #self.sigmoid = sigmoid_rot
        self.sigmoid = torch.relu
        
        # note medmnist uses 64,128,256,512
        # also some expansion
        
        
        
        # changes to make
        # first more channels, 63
        # second 1x1 convolutions instea of just repeating
        self.n0 = n0
        self.n1 = n1
        #self.c0 = torch.nn.Conv2d(3,n0,3,1,1) # 32x32
        #self.bn0 = torch.nn.BatchNorm2d(n0)
        
        # I'd like to initialize it as 3 scalar fields
        # therefore it will start as 9 dimensions, with zeros
        self.c0 = torch.nn.Conv2d(3,n0,3,1,1,bias=False) # 32x32
        self.bn0 = torch.nn.BatchNorm2d(n0)
        
        # now a set of 2        
        self.c1 = torch.nn.Conv2d(n0,n0,3,1,1,bias=False) # 32x32
        self.bn1 = torch.nn.BatchNorm2d(n0)
        self.c1a = torch.nn.Conv2d(n0,n0,3,1,1,bias=False) # 32x32
        self.bn1a = torch.nn.BatchNorm2d(n0)
        
        # and again
        self.c1_ = torch.nn.Conv2d(n0,n0,3,1,1,bias=False) # 32x32
        self.bn1_ = torch.nn.BatchNorm2d(n0)
        self.c1a_ = torch.nn.Conv2d(n0,n0,3,1,1,bias=False) # 32x32
        self.bn1a_ = torch.nn.BatchNorm2d(n0)
       
    
        
        
        
        self.c2 = torch.nn.Conv2d(n0,n0*2,3,2,1,bias=False) # 16x16
        self.bn2 = torch.nn.BatchNorm2d(n0*2)
        self.c2a = torch.nn.Conv2d(n0*2,n0*2,3,1,1,bias=False) # 16x16
        self.bn2a = torch.nn.BatchNorm2d(n0*2)
        # here we change dimension
        # so this needs a shortcut projection
        # note a 1x1 convolution is always rotationally invariant?
        #self.s2 = torch.nn.Conv2d(n0,n0*2,1,2,1,bias=False)
        #self.bns2 = BatchNormRot(n0*2)
        
        
        self.c2_ = torch.nn.Conv2d(n0*2,n0*2,3,1,1,bias=False) # 16x16
        self.bn2_ = torch.nn.BatchNorm2d(n0*2)
        self.c2a_ = torch.nn.Conv2d(n0*2,n0*2,3,1,1,bias=False) # 16x16
        self.bn2a_ = torch.nn.BatchNorm2d(n0*2)
        
     
    
        
        
        self.c3 = torch.nn.Conv2d(n0*2,n0*4,3,2,1,bias=False) # 8x8
        self.bn3 = torch.nn.BatchNorm2d(n0*4)
        self.c3a = torch.nn.Conv2d(n0*4,n0*4,3,1,1,bias=False) # 8x8
        self.bn3a = torch.nn.BatchNorm2d(n0*4)
                
            
        self.c3_ = torch.nn.Conv2d(n0*4,n0*4,3,1,1,bias=False) # 8x8
        self.bn3_ = torch.nn.BatchNorm2d(n0*4)
        self.c3a_ = torch.nn.Conv2d(n0*4,n0*4,3,1,1,bias=False) # 8x8
        self.bn3a_ = torch.nn.BatchNorm2d(n0*4)
        
        
        
        self.c4 = torch.nn.Conv2d(n0*4,n0*8,3,2,1,bias=False) # 4x4
        self.bn4 = torch.nn.BatchNorm2d(n0*8)
        self.c4a = torch.nn.Conv2d(n0*8,n0*8,3,1,1,bias=False) # 4x4
        self.bn4a = torch.nn.BatchNorm2d(n0*8)
                
            
        self.c4_ = torch.nn.Conv2d(n0*8,n0*8,3,1,1,bias=False) # 4x4
        self.bn4_ = torch.nn.BatchNorm2d(n0*8)
        self.c4a_ = torch.nn.Conv2d(n0*8,n0*8,3,1,1,bias=False) # 4x4
        self.bn4a_ = torch.nn.BatchNorm2d(n0*8)
        
     
    
        

        # in the very last layer, I'll just use the scalar parts
        #self.linear = torch.nn.Linear(n0*4,10)
        self.linear = torch.nn.Linear(n0*8,n1)
        
        
    def forward(self,x):
        
        # set up x as a scalar field
        #z = torch.zeros_like(x[...,0,:,:])
        #x = torch.stack( (x[...,0,:,:], z, z, 
        #                 x[...,1,:,:], z,z,
        #                 x[...,2,:,:], z, z), -3)
        
        if x.shape[-3] == 1:
            x = torch.concatenate((x,x,x),-3)
        x = self.sigmoid(self.bn0(self.c0(x)))
        
        
        x0 = x.clone()
        x = self.sigmoid(self.bn1(self.c1(x)))        
        x = self.bn1a(self.c1a(x))        
        x = x + x0
        x = self.sigmoid(x)        
        
        x0 = x.clone()
        x = self.sigmoid(self.bn1_(self.c1_(x)))        
        x = self.bn1a_(self.c1a_(x))
        x = x + x0
        x = self.sigmoid(x)        
        
   

        
        
        
        
        x0 = x.clone()
        x = self.sigmoid(self.bn2(self.c2(x)))
        x = self.bn2a(self.c2a(x))
        x = x + x0[...,::2,::2].repeat(1,2,1,1)
        #x0 = self.bns2(self.s2(x0))
        x = self.sigmoid(x)
        
        x0 = x.clone()
        x = self.sigmoid(self.bn2_(self.c2_(x)))
        x = self.bn2a_(self.c2a_(x))
        x = x + x0
        x = self.sigmoid(x)
        
     
    
        
        
        
        x0 = x.clone()
        x = self.sigmoid(self.bn3(self.c3(x)))
        x = self.bn3a(self.c3a(x))
        x = x + x0[...,::2,::2].repeat(1,2,1,1)
        x = self.sigmoid(x)
        
        x0 = x.clone()
        x = self.sigmoid(self.bn3_(self.c3_(x)))
        x = self.bn3a_(self.c3a_(x))
        x = x + x0
        x = self.sigmoid(x)        
        
        
        
        x0 = x.clone()
        x = self.sigmoid(self.bn4(self.c4(x)))
        x = self.bn4a(self.c4a(x))
        x = x + x0[...,::2,::2].repeat(1,2,1,1)
        x = self.sigmoid(x)
        
        x0 = x.clone()
        x = self.sigmoid(self.bn4_(self.c4_(x)))
        x = self.bn4a_(self.c4a_(x))
        x = x + x0
        x = self.sigmoid(x)        
        
 


        
        #x = x[...,0::3,:,:]
        x = torch.mean(x,(-1,-2))
        x = self.linear(x)
        
        
        return x
    












class RotNet20(torch.nn.Module):
    '''
    Match the resnet architecture for the cifar dataset described in original resnet paper

    Three scales with three repeats each

    16 32 and 64 channels

    '''
    def __init__(self,n0=15,n1=10,kernel_size=3):
        super().__init__()
        self.n0 = n0
        self.n1 = n1

        if kernel_size==3:
            padding = 1
        elif kernel_size == 5:
            padding = 2

        self.tv = ToVector()        

        # first layer
        self.c0 = Conv2DRot(9,n0,kernel_size,1,padding,bias=False) # 32x32
        self.b0 = BatchNormRot(n0)
        self.s0 = SigmoidRot(n0)


        # now a set of 3, at full res
        self.c1 = Conv2DRot(n0,n0,kernel_size,1,padding,bias=False) # 32x32
        self.b1 = BatchNormRot(n0)
        self.s1 = SigmoidRot(n0)
        self.c1a = Conv2DRot(n0,n0,kernel_size,1,padding,bias=False) # 32x32
        self.b1a = BatchNormRot(n0)
        self.s1a = SigmoidRot(n0)

        self.c1_ = Conv2DRot(n0,n0,kernel_size,1,padding,bias=False) # 32x32
        self.b1_ = BatchNormRot(n0)
        self.s1_ = SigmoidRot(n0)
        self.c1a_ = Conv2DRot(n0,n0,kernel_size,1,padding,bias=False) # 32x32
        self.b1a_ = BatchNormRot(n0)
        self.s1a_ = SigmoidRot(n0)

        self.c1__ = Conv2DRot(n0,n0,kernel_size,1,padding,bias=False) # 32x32
        self.b1__ = BatchNormRot(n0)
        self.s1__ = SigmoidRot(n0)
        self.c1a__ = Conv2DRot(n0,n0,kernel_size,1,padding,bias=False) # 32x32
        self.b1a__ = BatchNormRot(n0)
        self.s1a__ = SigmoidRot(n0)

        # now a set of 3, at half res
        self.c2 = Conv2DRot(n0,n0*2,kernel_size,1,padding,bias=False)
        self.b2 = BatchNormRot(n0*2)
        self.s2 = SigmoidRot(n0*2)        
        self.c2a = Conv2DRot(n0*2,n0*2,kernel_size,1,padding,bias=False)
        self.b2a = BatchNormRot(n0*2)
        self.s2a = SigmoidRot(n0*2)
        self.d2 = Down2Rot()

        self.c2_ = Conv2DRot(n0*2,n0*2,kernel_size,1,padding,bias=False)
        self.b2_ = BatchNormRot(n0*2)
        self.s2_ = SigmoidRot(n0*2)        
        self.c2a_ = Conv2DRot(n0*2,n0*2,kernel_size,1,padding,bias=False)
        self.b2a_ = BatchNormRot(n0*2)
        self.s2a_ = SigmoidRot(n0*2)

        self.c2__ = Conv2DRot(n0*2,n0*2,kernel_size,1,padding,bias=False)
        self.b2__ = BatchNormRot(n0*2)
        self.s2__ = SigmoidRot(n0*2)        
        self.c2a__ = Conv2DRot(n0*2,n0*2,kernel_size,1,padding,bias=False)
        self.b2a__ = BatchNormRot(n0*2)
        self.s2a__ = SigmoidRot(n0*2)

        # now a set of 3, at quarter res
        self.c3 = Conv2DRot(n0*2,n0*4,kernel_size,1,padding,bias=False)
        self.b3 = BatchNormRot(n0*4)
        self.s3 = SigmoidRot(n0*4)        
        self.c3a = Conv2DRot(n0*4,n0*4,kernel_size,1,padding,bias=False)
        self.b3a = BatchNormRot(n0*4)
        self.s3a = SigmoidRot(n0*4)
        self.d3 = Down2Rot()

        self.c3_ = Conv2DRot(n0*4,n0*4,kernel_size,1,padding,bias=False)
        self.b3_ = BatchNormRot(n0*4)
        self.s3_ = SigmoidRot(n0*4)        
        self.c3a_ = Conv2DRot(n0*4,n0*4,kernel_size,1,padding,bias=False)
        self.b3a_ = BatchNormRot(n0*4)
        self.s3a_ = SigmoidRot(n0*4)

        self.c3__ = Conv2DRot(n0*4,n0*4,kernel_size,1,padding,bias=False)
        self.b3__ = BatchNormRot(n0*4)
        self.s3__ = SigmoidRot(n0*4)        
        self.c3a__ = Conv2DRot(n0*4,n0*4,kernel_size,1,padding,bias=False)
        self.b3a__ = BatchNormRot(n0*4)
        self.s3a__ = SigmoidRot(n0*4)



        # finally, extract scalar and apply linear map
        self.extract_scalar = ExtractScalar()
        self.global_avg = GlobalAvg()
        self.linear = torch.nn.Linear(n0//3*4,n1)



       

    def forward(self,x):
       
        # if grayscale, map to RGB
        if x.shape[-3] == 1:
            x = torch.cat((x,x,x),-3)
        
        
        # convert to scalars in block form
        x = self.tv(x)        
        

        # first block
        x = self.s0(self.b0(self.c0(x)))
        
        
        # second block, three repeats at full resolution
        x0 = x.clone()
        x = self.s1(self.b1(self.c1(x)))        
        x = self.b1a(self.c1a(x)) 
        x = x + x0
        x = self.s1a(x)   
        # something is broken, images are the wrong size, even for 3x3
        
        x0 = x.clone()
        x = self.s1_(self.b1_(self.c1_(x)))        
        x = self.b1a_(self.c1a_(x))
        x = x + x0
        x = self.s1a_(x)

        x0 = x.clone()
        x = self.s1__(self.b1__(self.c1__(x)))        
        x = self.b1a__(self.c1a__(x))
        x = x + x0
        x = self.s1a__(x)


        # third block, three repeats at half resolution       
        x0 = x.clone()
        x = self.s2(self.b2(self.c2(x)))
        x = self.b2a(self.c2a(x))
        x = self.d2(x + x0.repeat(1,2,1,1))
        x = self.s2(x)

        x0 = x.clone()
        x = self.s2_(self.b2_(self.c2_(x)))
        x = self.b2a_(self.c2a_(x))
        x = x + x0
        x = self.s2a_(x)

        x0 = x.clone()
        x = self.s2__(self.b2__(self.c2__(x)))
        x = self.b2a__(self.c2a__(x))
        x = x + x0
        x = self.s2a__(x)


        # fourth block, three repeats at quarter resolution       
        x0 = x.clone()
        x = self.s3(self.b3(self.c3(x)))
        x = self.b3a(self.c3a(x))
        x = self.d3(x + x0.repeat(1,2,1,1))
        x = self.s3(x)

        x0 = x.clone()
        x = self.s3_(self.b3_(self.c3_(x)))
        x = self.b3a_(self.c3a_(x))
        x = x + x0
        x = self.s3a_(x)

        x0 = x.clone()
        x = self.s3__(self.b3__(self.c3__(x)))
        x = self.b3a__(self.c3a__(x))
        x = x + x0
        x = self.s3a__(x)

        # finally extract scalar and apply one linear map
        x = self.extract_scalar(x)
        x = self.global_avg(x)
        x = self.linear(x)

        return x
    


class ResNet20(torch.nn.Module):
    '''
    20 layer res net as in cifar10 evaluation
    
    
    '''
    def __init__(self,n0=64,n1=10):
        super().__init__()
        # note in the resnet paper, downsampling is done as the first block at a scale, not the last
        
        # follow the resnet idea
        # a first conv layer
        
        #self.sigmoid = sigmoid_rot
        self.sigmoid = torch.relu
        
        # note medmnist uses 64,128,256,512
        # also some expansion
        
        
        
        # changes to make
        # first more channels, 63
        # second 1x1 convolutions instea of just repeating
        self.n0 = n0
        self.n1 = n1
        
        
        # first layer, map to n0
        self.c0 = torch.nn.Conv2d(3,n0,3,1,1,bias=False) # 32x32
        self.bn0 = torch.nn.BatchNorm2d(n0)
        
        # now a set of 3 at full res
        self.c1 = torch.nn.Conv2d(n0,n0,3,1,1,bias=False) # 32x32
        self.bn1 = torch.nn.BatchNorm2d(n0)
        self.c1a = torch.nn.Conv2d(n0,n0,3,1,1,bias=False) # 32x32
        self.bn1a = torch.nn.BatchNorm2d(n0)
                
        self.c1_ = torch.nn.Conv2d(n0,n0,3,1,1,bias=False) # 32x32
        self.bn1_ = torch.nn.BatchNorm2d(n0)
        self.c1a_ = torch.nn.Conv2d(n0,n0,3,1,1,bias=False) # 32x32
        self.bn1a_ = torch.nn.BatchNorm2d(n0)
       
        self.c1__ = torch.nn.Conv2d(n0,n0,3,1,1,bias=False) # 32x32
        self.bn1__ = torch.nn.BatchNorm2d(n0)
        self.c1a__ = torch.nn.Conv2d(n0,n0,3,1,1,bias=False) # 32x32
        self.bn1a__ = torch.nn.BatchNorm2d(n0)
       
    
        
        
        # now a set of 3 at half res
        self.c2 = torch.nn.Conv2d(n0,n0*2,3,2,1,bias=False) # 16x16
        self.bn2 = torch.nn.BatchNorm2d(n0*2)
        self.c2a = torch.nn.Conv2d(n0*2,n0*2,3,1,1,bias=False) # 16x16
        self.bn2a = torch.nn.BatchNorm2d(n0*2)                
        
        self.c2_ = torch.nn.Conv2d(n0*2,n0*2,3,1,1,bias=False) # 16x16
        self.bn2_ = torch.nn.BatchNorm2d(n0*2)
        self.c2a_ = torch.nn.Conv2d(n0*2,n0*2,3,1,1,bias=False) # 16x16
        self.bn2a_ = torch.nn.BatchNorm2d(n0*2)

        self.c2__ = torch.nn.Conv2d(n0*2,n0*2,3,1,1,bias=False) # 16x16
        self.bn2__ = torch.nn.BatchNorm2d(n0*2)
        self.c2a__ = torch.nn.Conv2d(n0*2,n0*2,3,1,1,bias=False) # 16x16
        self.bn2a__ = torch.nn.BatchNorm2d(n0*2)
        
     
    
        
        # now a set of 3 at quarter res
        self.c3 = torch.nn.Conv2d(n0*2,n0*4,3,2,1,bias=False) # 8x8
        self.bn3 = torch.nn.BatchNorm2d(n0*4)
        self.c3a = torch.nn.Conv2d(n0*4,n0*4,3,1,1,bias=False) # 8x8
        self.bn3a = torch.nn.BatchNorm2d(n0*4)
                            
        self.c3_ = torch.nn.Conv2d(n0*4,n0*4,3,1,1,bias=False) # 8x8
        self.bn3_ = torch.nn.BatchNorm2d(n0*4)
        self.c3a_ = torch.nn.Conv2d(n0*4,n0*4,3,1,1,bias=False) # 8x8
        self.bn3a_ = torch.nn.BatchNorm2d(n0*4)
        
        self.c3__ = torch.nn.Conv2d(n0*4,n0*4,3,1,1,bias=False) # 8x8
        self.bn3__ = torch.nn.BatchNorm2d(n0*4)
        self.c3a__ = torch.nn.Conv2d(n0*4,n0*4,3,1,1,bias=False) # 8x8
        self.bn3a__ = torch.nn.BatchNorm2d(n0*4)
        
                
        

        # in the very last layer, I'll just use the scalar parts        
        self.linear = torch.nn.Linear(n0*4,n1)
        
        
    def forward(self,x):
                
        # change from 1d to grayscale
        if x.shape[-3] == 1:
            x = torch.concatenate((x,x,x),-3)
        # apply first layer
        x = self.sigmoid(self.bn0(self.c0(x)))
        
        # now the first block of 3
        x0 = x.clone()
        x = self.sigmoid(self.bn1(self.c1(x)))        
        x = self.bn1a(self.c1a(x))        
        x = x + x0
        x = self.sigmoid(x)        
        
        x0 = x.clone()
        x = self.sigmoid(self.bn1_(self.c1_(x)))        
        x = self.bn1a_(self.c1a_(x))
        x = x + x0
        x = self.sigmoid(x)        
        
        x0 = x.clone()
        x = self.sigmoid(self.bn1__(self.c1__(x)))        
        x = self.bn1a__(self.c1a__(x))
        x = x + x0
        x = self.sigmoid(x)        
        
   

        
        # now the second set of 3                
        x0 = x.clone()
        x = self.sigmoid(self.bn2(self.c2(x)))
        x = self.bn2a(self.c2a(x))
        x = x + x0[...,::2,::2].repeat(1,2,1,1)
        # note there is downsample here, regular resnet uses a 1x1 convolution, here we skip it
        x = self.sigmoid(x)
        
        x0 = x.clone()
        x = self.sigmoid(self.bn2_(self.c2_(x)))
        x = self.bn2a_(self.c2a_(x))
        x = x + x0
        x = self.sigmoid(x)
        
        x0 = x.clone()
        x = self.sigmoid(self.bn2__(self.c2__(x)))
        x = self.bn2a__(self.c2a__(x))
        x = x + x0
        x = self.sigmoid(x)
        
     
    
        
        
        # now the third set of 3
        x0 = x.clone()
        x = self.sigmoid(self.bn3(self.c3(x)))
        x = self.bn3a(self.c3a(x))
        x = x + x0[...,::2,::2].repeat(1,2,1,1)
        x = self.sigmoid(x)
        
        x0 = x.clone()
        x = self.sigmoid(self.bn3_(self.c3_(x)))
        x = self.bn3a_(self.c3a_(x))
        x = x + x0
        x = self.sigmoid(x)    

        x0 = x.clone()
        x = self.sigmoid(self.bn3__(self.c3__(x)))
        x = self.bn3a__(self.c3a__(x))
        x = x + x0
        x = self.sigmoid(x)        
        
        
        # global average and linear
        x = torch.mean(x,(-1,-2))
        x = self.linear(x)
                
        return x
    


def train_and_eval(net,my_loader, my_loader_val, my_loader_test, device='cpu',nepochs=100,lr=1e-3):
    loss = torch.nn.CrossEntropyLoss()

    net = net.to(device)        

    optimizer = torch.optim.Adam(net.parameters(),lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[nepochs*0.5, nepochs*0.75])
    Esave = []
    accuracy_test = []
    accuracy_train = []
    accuracy_val = []
    auc_test = []
    auc_train = []
    auc_val = []
    hard_auc_test = []
    hard_auc_train = []
    hard_auc_val = []
    fig,ax = plt.subplots(2,2)
    ax = ax.ravel()
    for e in range(nepochs):
        count = 0
        correct = 0
        Esave_ = []
        truth = []
        prediction = []
        hard_prediction = []
        for x,l in my_loader:
            x = x.to(device)
            l = l.to(device).squeeze()
            optimizer.zero_grad()
            lhat = net(x)

            E = loss(lhat,l)        
            E.backward()
            optimizer.step()

            Esave_.append(E.item())

            label = torch.argmax(lhat.data,-1)
            count += len(label)            
            correct += torch.sum(label==l).item()
            truth.extend([li.item() for li in l])
            prediction.extend([li.cpu() for li in torch.softmax(lhat.data,-1)])
            hard_prediction.extend([torch.argmax(li.data,-1).item() for li in lhat.data])

        lr_scheduler.step()

        Esave.append(np.mean(Esave_))
        accuracy_train.append(correct/count)

        # auc
        unique_labels = np.unique(truth)
        scores = []
        for u in unique_labels:
            scores.append( roc_auc_score([t==u for t in truth],[t[u] for t in prediction]) )
        auc = np.mean(scores)
        auc_train.append(auc)
        scores = []
        for u in unique_labels:
            scores.append( roc_auc_score([t==u for t in truth],[t==u for t in hard_prediction]) )
        auc = np.mean(scores)
        hard_auc_train.append(auc)


        count = 0    
        correct = 0 
        net.train(False)
        truth = []
        prediction = []
        hard_prediction = []
        with torch.no_grad():
            for x,l in my_loader_test:
                x = x.to(device)
                l = l.to(device).squeeze()
                lhat = net(x)


                label = torch.argmax(lhat,-1)


                count += len(label)            
                correct += torch.sum(label==l).item()

                truth.extend([li.item() for li in l])
                prediction.extend([li.cpu() for li in torch.softmax(lhat.data,-1)])
                hard_prediction.extend([torch.argmax(li.data,-1).item() for li in lhat.data])
        net.train(True)    
        accuracy_test.append(correct/count)

        scores = []
        for u in unique_labels:
            scores.append( roc_auc_score([t==u for t in truth],[t[u] for t in prediction]) )
        auc = np.mean(scores)
        auc_test.append(auc)
        scores = []
        for u in unique_labels:
            scores.append( roc_auc_score([t==u for t in truth],[t==u for t in hard_prediction]) )
        auc = np.mean(scores)
        hard_auc_test.append(auc)


        count = 0    
        correct = 0 
        net.train(False)
        truth = []
        prediction = []
        hard_prediction = []
        with torch.no_grad():
            for x,l in my_loader_val:
                x = x.to(device)
                l = l.to(device).squeeze()
                lhat = net(x)

                label = torch.argmax(lhat,-1)           

                count += len(label)            
                correct += torch.sum(label==l).item()

                truth.extend([li.item() for li in l])
                prediction.extend([li.cpu() for li in torch.softmax(lhat.data,-1)])
                hard_prediction.extend([torch.argmax(li.data,-1).item() for li in lhat.data])
        net.train(True)

        accuracy_val.append(correct/count)

        scores = []
        for u in unique_labels:
            scores.append( roc_auc_score([t==u for t in truth],[t[u] for t in prediction]) )
        auc = np.mean(scores)
        auc_val.append(auc)
        scores = []
        for u in unique_labels:
            scores.append( roc_auc_score([t==u for t in truth],[t==u for t in hard_prediction]) )
        auc = np.mean(scores)
        hard_auc_val.append(auc)


        ax[0].cla()
        ax[0].plot(Esave)
        ax[1].cla()
        ax[1].plot(accuracy_train,label='train')
        ax[1].plot(accuracy_val,label='val')
        ax[1].plot(accuracy_test,label='test')
        ax[1].legend()
        ax[1].set_title('accuracy')

        ax[2].cla()
        ax[2].plot(auc_train,label='train')
        ax[2].plot(auc_val,label='val')
        ax[2].plot(auc_test,label='test')
        ax[2].legend()
        ax[2].set_title('AUC')

        ax[3].cla()
        ax[3].plot(hard_auc_train,label='train')
        ax[3].plot(hard_auc_val,label='val')
        ax[3].plot(hard_auc_test,label='test')
        ax[3].legend()
        ax[3].set_title('Hard AUC')
        clear_output(wait=True)
        display(fig)        
        fig.canvas.draw()
        
    return {        
        'fig':fig, 
        'net':net, 
        'accuracy_test':accuracy_test,  
        'accuracy_val':accuracy_val,
        'accuracy_train':accuracy_train,
        'auc_test':auc_test,  
        'auc_val':auc_val,
        'auc_train':auc_train,
        'hard_auc_test':hard_auc_test,  
        'hard_auc_val':hard_auc_val,
        'hard_auc_train':hard_auc_train,
    }


def count_parameters(net):
    count = 0
    for p in net.parameters():
        count += p.numel()
    return count