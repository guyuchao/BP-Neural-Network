from flask import Flask,jsonify,render_template,request
from BPmain import Train2D,Train3D,Trainmnist
import numpy as np
from BPmain import ParseExpression
import json
app = Flask(__name__)


train2d_session=None
train3d_session=None
trainmnist_session=None
parse=ParseExpression()

@app.route('/')
def index():
    return render_template("train2d.html")

@app.route('/train2d')
def train2d():
    return render_template("train2d.html")

@app.route('/train3d')
def train3d():
    return render_template("train3d.html")

@app.route('/trainmnist')
def trainmnist():
    return render_template("mnist.html")
#####################################################2D section#########
@app.route('/get2dLoss',methods=['GET'])
def get2d_loss():
    if train2d_session is not None:
        x,ty,vy=train2d_session.get_loss()
        traindata = dict(
            x=x,
            y=ty,
            type='plot',
            mode='lines',
        )
        valdata = dict(
            x=x,
            y=vy,
            type='plot',
            mode='lines',
        )

    else:
        traindata = dict(
            x=[],
            y=[],
            type='plot',
            mode='lines'
        )
        valdata = dict(
            x=[],
            y=[],
            type='plot',
            mode='lines'
        )
        print("nodata")

    return jsonify(Datatrain=traindata,Dataval=valdata)

@app.route('/begin2dTrain',methods=['POST'])
def begin2d_train():
    train2d_session.train_epoch()#, (a, b))
    return jsonify({"code":200})

@app.route('/prepare2dData',methods=['POST'])
def prepare2d_data():
    jsondata = request.get_data()
    paradict = json.loads(jsondata)
    global train2d_session
    train2d_session = Train2D(paradict)
    traindata,valdata=train2d_session.prepare_2ddata()
    train = dict(
        x=traindata[0],
        y=traindata[1],
        type='scatter',
        mode='markers',
    )
    val = dict(
        x=valdata[0],
        y=valdata[1],
        type='scatter',
        mode='markers'
    )
    return jsonify(Datatrain=train,Dataval=val)

@app.route('/get2dTraincurve',methods=['GET'])
def get2d_train_curve():
    if train2d_session is not None:
        traincurve,valcurve=train2d_session.get_pred_curve()
        datatrain = dict(
            x=traincurve[0],
            y=traincurve[1],
            type='plot',
            bnmode='lines'
        )
        dataval = dict(
            x=valcurve[0],
            y=valcurve[1],
            type='plot',
            mode='lines'
        )
    else:
        datatrain = dict(
            x=[],
            y=[],
            type='plot',
            mode='lines'
        )
        dataval = dict(
            x=[],
            y=[],
            type='plot',
            mode='lines'
        )
    return jsonify(Datatrain=datatrain,Dataval=dataval)
########################################################3d section###############################################3
@app.route('/begin3dTrain',methods=['POST'])
def begin3d_train():
    train3d_session.train_epoch()
    return jsonify({"code":200})

@app.route('/prepare3dData',methods=['POST'])
def prepare3d_data():
    jsondata = request.get_data()
    paradict = json.loads(jsondata)
    global train3d_session
    train3d_session = Train3D(paradict)
    x,y,z,valx,valy,valz=train3d_session.prepare_3ddata()
    traindata = dict(
        x=x,
        y=y,
        z=z,
        type='scatter3d',
        mode='markers',
        marker=dict(
            size=3
        )
    )
    valdata = dict(
        x=valx,
        y=valy,
        z=valz,
        type='scatter3d',
        mode='markers',
        marker=dict(
            size=3
        )
    )
    return jsonify(TrainData=traindata,ValData=valdata)


@app.route('/get3dTrainsurface',methods=['GET'])
def get3d_train_surface():
    if train3d_session is not None:
        x,y,z,valx,valy,valz=train3d_session.get_pred_surface()
    else:
        x=y=z=valx=valy=valz=[]
    traindata=dict(
        type='surface',
        x=x,
        y=y,
        z=z
    )
    valdata = dict(
        type='surface',
        x=valx,
        y=valy,
        z=valz
    )
    return jsonify(Traindata=traindata,Valdata=valdata)

@app.route('/get3dLoss',methods=['GET'])
def get3d_loss():
    if train3d_session is not None:
        idx,tl,vl=train3d_session.get_loss()
        traindata = dict(
            x=idx,
            y=tl,
            type='plot',
        )
        valdata = dict(
            x=idx,
            y=vl,
            type='plot',
        )
    else:
        traindata = dict(
            x=[],
            y=[],
            type='plot'
        )
        valdata = dict(
            x=[],
            y=[],
            type='plot'
        )
    return jsonify(TrainData=traindata,ValData=valdata)
#################################mnist##############################
@app.route('/preparemnist',methods=['POST'])
def preparemnist():
    jsondata = request.get_data()
    paradict = json.loads(jsondata)
    global trainmnist_session
    trainmnist_session = Trainmnist(paradict)
    return jsonify(code=200)

@app.route('/beginmnistTrain',methods=['POST'])
def beginmnist_train():
    trainmnist_session.train_epoch()#, (a, b))
    return jsonify({"code":200})

@app.route('/getmnistLoss',methods=['GET'])
def getmnist_loss():
    if trainmnist_session is not None:
        x, ty, vy = trainmnist_session.get_loss()
        traindata = dict(
            x=x,
            y=ty,
            type='plot',
            mode='lines',
            marker=dict(
                color='black',
            )
        )
        valdata = dict(
            x=x,
            y=vy,
            type='plot',
            mode='lines',
            marker = dict(
                color='blue',
            )
        )

    else:
        traindata = dict(
            x=[],
            y=[],
            type='plot'
        )
        valdata = dict(
            x=[],
            y=[],
            type='plot'
        )
    return jsonify(Traindata=traindata,Valdata=valdata)

@app.route('/getmnistpred',methods=['GET'])
def getmnist_pred():
    if trainmnist_session is not None:
        predimg,predlable=trainmnist_session.get_pred_mnist()
        return jsonify(Code="200",Predimg=predimg,Predlabel=predlable)
    else:
        return jsonify(Code="404")
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5500, threaded=True)
    '''
    data=parse("power(x,2)",mode='2d')
    trainx,trainy=data
    import matplotlib.pyplot as plt
    plt.plot(trainx.flatten(),trainy.flatten())
    plt.show()
    '''