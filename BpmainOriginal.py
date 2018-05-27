import numpy as np
import tqdm
import random
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

class BaseNetwork(object):
    def __init__(self):
        pass
    def forward(self,*x):
        pass

    def parameters(self):
        pass

    def backward(self,grad):
        pass
    def __call__(self,*x):
        return self.forward(*x)

class Sequence(BaseNetwork):
    def __init__(self,*layer):
        super(Sequence,self).__init__()
        self.layers=[]
        self.parameter=[]
        for item in layer:
            self.layers.append(item)

        for layer in self.layers:
            if isinstance(layer,Linear):
                self.parameter.append(layer.parameters())

    def add_layer(self,layer):
        self.layers.append(layer)

    def forward(self,*x):
        x=x[0]
        for layer in self.layers:
            x=layer(x)
        return x

    def backward(self,grad):
        for layer in reversed(self.layers):
            grad=layer.backward(grad)

    def parameters(self):
        return self.parameter



class Variable(object):
    def __init__(self,weight,wgrad,bias,bgrad):
        self.weight=weight
        self.wgrad=wgrad
        self.v_weight=np.zeros(self.weight.shape)
        self.bias=bias
        self.bgrad=bgrad
class Linear(BaseNetwork):
    def __init__(self,inplanes,outplanes):
        super(Linear,self).__init__()
        self.weight=np.random.randn(inplanes,outplanes)*0.5
        self.bias=np.random.randn(outplanes)*0.5
        self.input=None
        self.output=None
        self.wgrad=np.zeros(self.weight.shape)
        self.bgrad=np.zeros(self.bias.shape)
        self.variable=Variable(self.weight,self.wgrad,self.bias,self.bgrad)
    def parameters(self):
        return self.variable
    def forward(self,*x):
        x=x[0]
        self.input=x
        self.output=np.dot(self.input,self.weight)+self.bias
        return self.output
    def backward(self,grad):
        self.bgrad=grad
        self.wgrad += np.dot(self.input.T, grad)
        # self.bgrad+=grad=
        grad = np.dot(grad, self.weight.T)
        return grad
class Relu(BaseNetwork):
    def __init__(self):
        super(Relu,self).__init__()
        self.input=None
        self.output=None
    def forward(self,*x):
        x=x[0]
        self.input=x
        x[self.input<=0]*=0
        self.output=x
        return self.output
    def backward(self,grad):
        grad[self.input>0]*=1
        grad[self.input<=0]*=0
        return grad

class Sigmoid(BaseNetwork):
    def __init__(self):
        super(Sigmoid,self).__init__()
        self.input=None
        self.output=None
    def forward(self,*x):
        x=x[0]
        self.input=x
        self.output=1/(1+np.exp(-self.input))
        return self.output
    def backward(self,grad):
        grad*=self.output*(1-self.output)
        return grad

class MSE(object):
    def __init__(self):
        self.label=None
        self.pred=None
        self.grad=None
        self.loss=None
    def __call__(self, pred,label):
        return self.forward(pred,label)
    def forward(self,pred,label):
        self.pred,self.label=pred,label
        self.loss=np.sum(0.5*np.square(self.pred-self.label))
        return self.loss
    def backward(self,grad=None):
        self.grad=(self.pred-self.label)
        ret_grad=np.sum(self.grad,axis=0)
        return np.expand_dims(ret_grad,axis=0)


class SGD(object):
    def __init__(self,parameters,lr=0.01,momentum=0.9):
        self.parameters=parameters
        self.lr=lr
        self.momentum=momentum

    def zero_grad(self):
        for parameters in self.parameters:
            parameters.wgrad*=0
            parameters.bgrad*=0

    def step(self):
        for parameters in self.parameters:
            v=parameters.v_weight*self.momentum-self.lr*parameters.wgrad
            parameters.weight+=v
            parameters.bias-=self.lr*parameters.bgrad

class Mynet(BaseNetwork):
    def __init__(self):
        super(Mynet,self).__init__()
        self.layers=Sequence(
            Linear(2, 100),
            Relu(),
            Linear(100, 1)
        )
        self.criterion=MSE()

    def parameters(self):
        return self.layers.parameters()


    def forward(self,*x):
        x=x[0]
        return self.layers.forward(x)


    def backward(self,grad=None):
        grad=self.criterion.backward(grad)
        self.layers.backward(grad)



mynet=Mynet()
criterion=mynet.criterion
optimizer=SGD(mynet.parameters(),lr=0.00001,momentum=0.9)

#a=np.linspace(-1,1,1000)
#a=np.expand_dims(a,1)
x=np.linspace(-20,20,41)
y=np.linspace(-20,20,41)
X,Y=np.meshgrid(x,y)
t=np.dstack((X,Y))

t=t.reshape(-1,2)


label=t[:,0]+t[:,1]
h=label.reshape(-1,1)



for i in tqdm.tqdm(range(1000)):
    running_loss=0.0
    for row in range(t.shape[0]):
        optimizer.zero_grad()
        input=t[row:row+1]
        label=h[row:row+1]
        pred=mynet(input)
        loss=criterion(pred,label)
        running_loss+=loss
        mynet.backward()
        optimizer.step()
    if i%100==0:
        valpred=[]
        print(" loss: ", running_loss/row)
        for row in range(t.shape[0]):
            optimizer.zero_grad()
            input = t[row:row + 1]
            pred = mynet(input)
            valpred.append(pred)
        valpred=np.array(valpred)
        valpred=valpred.reshape(41,41)
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.plot_surface(X, Y, valpred, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'))
        plt.show()



'''
import matplotlib.pyplot as plt

pltx=[]
plty=[]
for row in range(1000):
    input = a[row:row + 1]
    pred = mynet(input*np.pi)
    pltx.append(input.flatten())
    plty.append(pred.flatten())

pltx=np.array(pltx)
plty=np.array(plty)
plt.plot(pltx,plty,'-')
plt.show()

x=np.array([1,2,3,4])
y=np.array([5,6,7,8])
X,Y=np.meshgrid(x,y)
t=np.dstack((X,Y))

t=t.reshape(-1,2)
label=t[:,0]+t[:,1]
label=label.reshape(-1,1)
'''