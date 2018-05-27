import numpy as np
from tqdm import tqdm
from io import BytesIO
import base64
from PIL import Image
from numpy import sin,cos,power,pi


def img2base64(img):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str

class Z_socre(object):
    def z_score(self,x):
        mean = np.mean(x, axis=0)
        std = np.std(x, axis=0)
        return (x-mean)/std,mean,std
    def Dz_score(self,x,mean,std):
        return x*std+mean
    def z_score_mean_std(self,x,mean,std):
        return (x-mean)/std
'''
class MinMax(object):
    def min_max_norm(self,x):
        minn=x.min(0)
        maxx=x.max(0)
        return (x-minn)/(maxx-minn),minn,maxx

    def Dminmax(self,x,min,max):
        return x*(max-min)+min

    def minmax_withminmax(self,x,min,max):
        return (x-min)/(max-min)
'''
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
    def __init__(self,inplanes,outplanes,preweight=None):
        super(Linear,self).__init__()
        if preweight is None:
            self.weight = np.random.randn(inplanes, outplanes) * 0.5
            self.bias = np.random.randn(outplanes) * 0.5
        else:
            self.weight, self.bias = preweight
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
            #parameters.v_weight=parameters.v_weight*self.momentum-self.lr*parameters.wgrad
            parameters.weight-=self.lr*parameters.wgrad
            parameters.bias-=self.lr*parameters.bgrad

class Mynet(BaseNetwork):
    def __init__(self,inplanes,outplanes):
        super(Mynet,self).__init__()
        self.layers=Sequence(
            Linear(inplanes, 100),
            Relu(),
            Linear(100, outplanes)
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

class ParseExpression(object):
    def __call__(self, expression,mode='2d'):
        if mode=='2d':
            if ("sin" in expression) or ("cos" in expression):
                x = np.linspace(-1, 1, 100)

                trainx = np.expand_dims(x, 1)
                trainy = eval(expression)

                normx, mean_train, std_train = Z_socre().z_score(trainx)
                normy, mean_label, std_label = Z_socre().z_score(trainy)

                x = np.sort(np.random.uniform(-1, 1,100))
                val_x = np.expand_dims(x, 1)
                val_y = eval(expression)
                val_x = Z_socre().z_score_mean_std(val_x, mean_train, std_train)
                val_y = Z_socre().z_score_mean_std(val_y, mean_label, std_label)
                return (normx, normy, mean_train, std_train, mean_label, std_label), (val_x, val_y)
            else:
                x = np.linspace(-100, 100, 100)

                trainx=np.expand_dims(x,1)
                trainy=eval(expression)

                normx, mean_train, std_train = Z_socre().z_score(trainx)
                normy, mean_label, std_label = Z_socre().z_score(trainy)

                x=np.sort(np.random.uniform(-100,100,100))
                val_x=np.expand_dims(x,1)
                val_y=eval(expression)
                val_x=Z_socre().z_score_mean_std(val_x,mean_train,std_train)
                val_y=Z_socre().z_score_mean_std(val_y,mean_label,std_label)
                return (normx,normy,mean_train,std_train,mean_label,std_label),(val_x,val_y)

        elif mode=='3d':
            x = np.linspace(-2, 2, 11)
            y = np.linspace(-2, 2, 11)
            ret_X_grid=x
            ret_Y_grid=y
            X,Y=np.meshgrid(x,y)
            trainxy=np.dstack((X,Y))
            trainxy=trainxy.reshape(-1,2)
            x=trainxy[:,0]
            y=trainxy[:,1]
            trainz=eval(expression)
            trainz.reshape(-1,1)
            tmpx=x
            tmpy=y
            val_x=np.sort(np.random.uniform(-2,2,11))
            val_y=np.sort(np.random.uniform(-2,2,11))
            ret_valX_grid=val_x
            ret_valY_grid=val_y
            valX,valY=np.meshgrid(val_x,val_y)
            valxy=np.dstack((valX,valY))
            valxy=valxy.reshape(-1,2)
            x=valxy[:,0]
            y=valxy[:,1]
            valz=eval(expression)
            valz.reshape(-1,1)
            tmpvalx=x
            tmpvaly=y
            return (ret_X_grid,ret_Y_grid,trainxy,trainz,tmpx,tmpy),(ret_valX_grid,ret_valY_grid,valxy,valz,tmpvalx,tmpvaly)
        elif mode=='mnist':
            train = np.load('/home/guyuchao/PycharmProject/AI/MnistNpy/traindata.npy')
            trainlabel = np.load('/home/guyuchao/PycharmProject/AI/MnistNpy/trainlabellogit.npy')
            val = np.load('/home/guyuchao/PycharmProject/AI/MnistNpy/valdata.npy')
            vallabel = np.load('/home/guyuchao/PycharmProject/AI/MnistNpy/vallabellogit.npy')
            return (train,trainlabel,val,vallabel)

class Train2D(object):
    def __init__(self,para_dict):
        self.net=Mynet(inplanes=1,outplanes=1)
        self.criterion=self.net.criterion
        self.optimizer=SGD(self.net.parameters(),lr=para_dict["lr"],momentum=para_dict["momentum"])
        self.train_loss=[]
        self.val_loss=[]
        self.train_pred=[]
        self.val_pred=[]
        self.train_data,self.val_data=ParseExpression()(para_dict['func'],mode='2d')
        self.epoches=para_dict['epoches']

    def train_epoch(self):
        epoches=self.epoches
        loss_epoch=epoches//100

        train_x,train_y,mean_train,std_train,mean_label,std_label=self.train_data
        val_x,val_y=self.val_data

        batches_train=train_x.shape[0]
        batches_val=val_x.shape[0]

        for epoch in tqdm(range(epoches)):
            running_loss=0.0
            ############################train########################
            for batch in range(batches_train):
                self.optimizer.zero_grad()
                input = train_x[batch:batch + 1]
                label = train_y[batch:batch + 1]
                pred = self.net(input)
                loss = self.criterion(pred, label)
                running_loss += loss
                self.net.backward()
                self.optimizer.step()

            ###########################val############################
            val_loss=0.0
            for batch in range(batches_val):
                input = val_x[batch:batch + 1]
                label = val_y[batch:batch + 1]
                pred = self.net(input)
                loss = self.criterion(pred, label)
                val_loss += loss
            ###########################summary########################
            if epoch%loss_epoch==0:
                self.train_loss.append(running_loss/batches_train)
                self.val_loss.append(val_loss/batches_val)

            if epoch%10==0:
                ######################train graph###################
                tmp_pred=[]
                for batch in range(batches_train):
                    input = train_x[batch:batch + 1]
                    pred = self.net(input)
                    input=Z_socre().Dz_score(input,mean_train,std_train)
                    pred=Z_socre().Dz_score(pred,mean_label,std_label)
                    tmp_pred.append((input.flatten()[0],pred.flatten()[0]))
                self.train_pred=tmp_pred
                ######################val graph########################
                tmp_pred = []
                for batch in range(batches_val):
                    input = val_x[batch:batch + 1]
                    pred = self.net(input)
                    input = Z_socre().Dz_score(input, mean_train, std_train)
                    pred = Z_socre().Dz_score(pred, mean_label, std_label)
                    tmp_pred.append((input.flatten()[0], pred.flatten()[0]))
                self.val_pred = tmp_pred

    def get_loss(self):
        idx=np.ndarray.tolist(np.linspace(0,len(self.train_loss)-1,len(self.train_loss)))
        return idx,self.train_loss,self.val_loss

    def prepare_2ddata(self):
        train_x, train_y, mean_train, std_train, mean_label, std_label=self.train_data
        trainx,trainy=Z_socre().Dz_score(train_x,mean_train,std_train),Z_socre().Dz_score(train_y,mean_label,std_label)
        trainx=np.ndarray.tolist(trainx.flatten())
        trainy=np.ndarray.tolist(trainy.flatten())
        valx, valy = self.val_data
        valx,valy=Z_socre().Dz_score(valx,mean_train,std_train),Z_socre().Dz_score(valy,mean_label,std_label)
        valx = np.ndarray.tolist(valx.flatten())
        valy = np.ndarray.tolist(valy.flatten())
        return (trainx,trainy),(valx,valy)

    def get_pred_curve(self):
        input_train = [item[0] for item in self.train_pred]
        pred_train = [item[1] for item in self.train_pred]
        input_val = [item[0] for item in self.val_pred]
        pred_val = [item[1] for item in self.val_pred]
        return (input_train,pred_train),(input_val,pred_val)

class Train3D(object):
    def __init__(self,para_dict):
        self.net = Mynet(inplanes=2,outplanes=1)
        self.criterion = self.net.criterion
        self.optimizer = SGD(self.net.parameters(), lr=para_dict['lr'], momentum=para_dict['momentum'])
        self.loss = []
        self.valloss=[]
        self.pred = []
        self.val_pred=[]
        self.data=ParseExpression()(para_dict['func'],mode='3d')
        traindata,valdata=self.data
        self.x_axis,self.y_axis,self.trainxy,self.trainz,self.trainx,self.trainy=traindata
        self.valx_axis, self.valy_axis, self.valxy, self.valz, self.valx, self.valy = valdata
        self.epoches=para_dict['epoches']

    def prepare_3ddata(self):
        return np.ndarray.tolist(self.trainx.flatten()),np.ndarray.tolist(self.trainy.flatten()),np.ndarray.tolist(self.trainz.flatten()),np.ndarray.tolist(self.valx.flatten()),np.ndarray.tolist(self.valy.flatten()),np.ndarray.tolist(self.valz.flatten())


    def train_epoch(self):
        epoches=self.epoches
        loss_epoch=epoches//100
        train_x, train_y = self.trainxy,self.trainz
        val_x,val_y=self.valxy,self.valz
        batches = train_x.shape[0]
        batches_val=val_x.shape[0]
        for epoch in tqdm(range(epoches)):
            running_loss = 0.0
            for batch in range(batches):
                self.optimizer.zero_grad()
                input = train_x[batch:batch + 1]
                label = train_y[batch:batch + 1]
                pred = self.net(input)
                loss = self.criterion(pred, label)
                running_loss += loss
                self.net.backward()
                self.optimizer.step()
            #print(running_loss/batches)
            val_loss = 0.0
            for batch in range(batches_val):
                input = train_x[batch:batch + 1]
                label = train_y[batch:batch + 1]
                pred = self.net(input)
                loss = self.criterion(pred, label)
                val_loss += loss

            if epoch%loss_epoch==0:
                self.loss.append(running_loss / batches)
                self.valloss.append(val_loss / batches_val)

            if epoch % 10 == 0:
                tmppred = []
                for row in range(batches):
                    input = train_x[row:row + 1]
                    pred = self.net(input)
                    tmppred.append(pred)
                tmppred = np.array(tmppred)
                tmppred = tmppred.reshape(len(self.x_axis),len(self.y_axis) )
                self.pred=np.ndarray.tolist(tmppred)
                tmppred = []
                for batch in range(batches_val):
                    input = val_x[batch:batch + 1]
                    pred = self.net(input)
                    tmppred.append(pred)
                tmppred = np.array(tmppred)
                tmppred = tmppred.reshape(len(self.valx_axis), len(self.valy_axis))
                self.val_pred = np.ndarray.tolist(tmppred)

    def get_loss(self):
        idx = np.ndarray.tolist(np.linspace(0, len(self.loss) - 1, len(self.loss)))
        return idx, self.loss,self.valloss

    def get_pred_surface(self):
        return np.ndarray.tolist(self.x_axis),np.ndarray.tolist(self.y_axis),self.pred,np.ndarray.tolist(self.valx_axis),np.ndarray.tolist(self.valy_axis),self.val_pred

class Mnistnet(BaseNetwork):
    def __init__(self,inplanes=28*28,outplanes=10):
        super(Mnistnet,self).__init__()
        w1_init=np.random.random((28*28, 28*28*2)) * 0.005
        b1_init=np.random.randn(28*28*2)*0.005
        w2_init = np.random.random((28 * 28*2, 10)) * 0.005
        b2_init = np.random.randn(10) * 0.005
        self.layers=Sequence(
            Linear(inplanes, 28*28*2,(w1_init,b1_init)),
            Relu(),
            Linear(28*28*2, outplanes,(w2_init,b2_init))
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

class Trainmnist(object):
    def __init__(self,para_dict):
        self.net = Mnistnet()
        self.criterion = self.net.criterion
        self.optimizer = SGD(self.net.parameters(), lr=para_dict['lr'], momentum=para_dict['momentum'])
        self.loss = []
        self.val_loss=[]
        self.pred = []
        self.data=ParseExpression()(None,mode='mnist')
        self.train,self.trainlabel,self.val,self.vallabel=self.data
        self.epoches=para_dict['epoches']
        self.valbatches=para_dict['valbatches']

    def train_epoch(self):
        epoches=self.epoches
        train_x, train_y = self.train,self.trainlabel
        val_x,val_y=self.val,self.vallabel
        train_batches = train_x.shape[0]
        val_batches=val_x.shape[0]
        loss_batch=train_batches//1000
        for epoch in tqdm(range(epoches)):
            running_loss = 0.0
            for batch in tqdm(range(train_batches)):
                self.optimizer.zero_grad()
                input = train_x[batch:batch + 1]/255
                label = train_y[batch:batch + 1]
                pred = self.net(input)
                loss = self.criterion(pred, label)
                running_loss += loss
                self.net.backward()
                self.optimizer.step()
                if batch%loss_batch==1:
                    self.loss.append(running_loss / batch)
                    '''
                    val_loss=0.0
                    for valid in range(val_batches):
                        input = train_x[batch:batch + 1] / 255
                        label = train_y[batch:batch + 1]
                        pred = self.net(input)
                        loss = self.criterion(pred, label)
                        val_loss+=loss
                    self.val_loss.append(val_loss/val_batches)
                    '''
                if batch%self.valbatches==0:
                    tmp_pred=[]
                    for val_idx in range(10):
                        rnd_val_idx=np.random.randint(0,val_batches)
                        valinput = val_x[rnd_val_idx:rnd_val_idx + 1] / 255
                        valpred = self.net(valinput)
                        img=Image.fromarray(val_x[rnd_val_idx:rnd_val_idx + 1].reshape(28,28).astype(np.uint8))
                        img_str=img2base64(img)
                        tmp_pred.append((img_str,valpred.flatten().argmax()))
                    self.pred=tmp_pred
            self.loss=[]

    def get_loss(self):
        idx = np.ndarray.tolist(np.linspace(0, len(self.loss) - 1, len(self.loss)))
        return idx, self.loss,self.val_loss

    def get_pred_mnist(self):
        img_url = {}
        pred={}
        for idx,item in enumerate(self.pred):
            img_url['img' + str(idx)] = item[0]
            pred['label'+str(idx)]=str(item[1])
        return img_url,pred

if __name__=='__main__':
    '''
    para_dict={
        'lr':0.01,
        'momentum':0.9,
        'epoches':3000,
        'func':'power(x,2)'
    }
    trainmnist=Train2D(para_dict)
    trainmnist.train_epoch()
    '''
    a="sisinoianf"
    print("sin" in a)