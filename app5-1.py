import flask
from flask import Flask, jsonify, request
from flask_restful import Resource, Api
import os
from torchvision import transforms
from torchvision import models
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.optim import lr_scheduler
from torch import optim
import io
from PIL import Image
import json

app = Flask(__name__)
api = Api(app)

model = models.resnet34(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 4)

device = torch.device('cpu')
model.to(device)

def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.ToTensor(),
                                        transforms.Resize((224,224))
                                        ])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)

def get_prediction_plastic(image_bytes):
    image_classs_index = json.load(open('/home/jjunhee98/kb_server/kb-hackerthon/class_index.json'))

    # model loading
    checkpoint_path = '/home/jjunhee98/kb_server/kb-hackerthon/trained_model/checkpoint1'
    file_name = 'checkpoint1test_lr_001_SGD_class4_freeze18-2.pt'

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.02, momentum=0.9, weight_decay=1e-5)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    if os.path.isdir(checkpoint_path) and os.path.isfile(checkpoint_path + '/' + file_name):
        checkpoint = torch.load(checkpoint_path + '/' + file_name,map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        exp_lr_scheduler.load_state_dict(checkpoint['schedular'])
        epoch_cnt = checkpoint['epoch_cnt']

    model.eval()

    tensor = transform_image(image_bytes=image_bytes)
    with torch.no_grad():
        inputs = Variable(tensor)
        outputs = model(inputs)
        _, y_hat = outputs.max(1)
        predicted_idx = str(y_hat.item())
        return image_classs_index[predicted_idx]

def get_prediction_paper(image_bytes):
    image_classs_index = json.load(open('/home/jjunhee98/kb_server/kb-hackerthon/class_index2.json'))

    # model loading
    checkpoint_path = '/home/jjunhee98/kb_server/kb-hackerthon/trained_model/checkpoint1'
    file_name = 'model2_papertest_lr_001_SGD_class4_freeze34-4.pt'

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.02, momentum=0.9, weight_decay=1e-5)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    if os.path.isdir(checkpoint_path) and os.path.isfile(checkpoint_path + '/' + file_name):
        checkpoint = torch.load(checkpoint_path + '/' + file_name,map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        exp_lr_scheduler.load_state_dict(checkpoint['schedular'])
        epoch_cnt = checkpoint['epoch_cnt']

    model.eval()

    tensor = transform_image(image_bytes=image_bytes)
    with torch.no_grad():
        inputs = Variable(tensor)
        outputs = model(inputs)
        _, y_hat = outputs.max(1)
        predicted_idx = str(y_hat.item())
        return image_classs_index[predicted_idx]

def get_prediction_paper_inside(image_bytes):
    image_classs_index = json.load(open('/home/jjunhee98/kb_server/kb-hackerthon/class_index3.json'))

    # model loading
    checkpoint_path = '/home/jjunhee98/kb_server/kb-hackerthon/trained_model/checkpoint1'
    file_name = 'model3_insidetest_lr_001_SGD_class4_freeze34-4.pt'

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.02, momentum=0.9, weight_decay=1e-5)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    if os.path.isdir(checkpoint_path) and os.path.isfile(checkpoint_path + '/' + file_name):
        checkpoint = torch.load(checkpoint_path + '/' + file_name,map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        exp_lr_scheduler.load_state_dict(checkpoint['schedular'])
        epoch_cnt = checkpoint['epoch_cnt']

    model.eval()

    tensor = transform_image(image_bytes=image_bytes)
    with torch.no_grad():
        inputs = Variable(tensor)
        outputs = model(inputs)
        _, y_hat = outputs.max(1)
        predicted_idx = str(y_hat.item())
        return image_classs_index[predicted_idx]



class Predict_Model_Plastic(Resource):
    def post(self):
        if request.method == 'POST':
            #file = request.files['file']
            file = request.get_data()
            if not file : return jsonify({"communication_success" : False})
            #img_bytes = file.read()
            #print(type(file))
            # plastic part(using model1)
            class_name = get_prediction_plastic(image_bytes=file)
            if class_name[0] == "plastic_cup":
                print(file)
                return jsonify({"communication_success":True,'class_name': class_name[0]})

            elif class_name[0] == "try_again":
                print(file)
                return jsonify({"communication_success":True,'class_name':class_name[0]})

            elif class_name[0] == "try_again_without":
                print(file)
                return jsonify({"communication_success": True,'class_name': class_name[0]})

            elif class_name[0] == "waste":
                print(file)
                return jsonify({"communication_success": True,'class_name': class_name[0]})

class Predict_Model_Paper(Resource):
    def post(self):
        if request.method == 'POST':
            #file = request.files['file']
            file = request.get_data()
            if not file : return jsonify({"communication_success" : False})
            #img_bytes = file.read()
            # paper part(using model2)
            class_name = get_prediction_paper(image_bytes=file)
            if class_name[0] == "paper_cup":
                return jsonify({"communication_success":True,'class_name': class_name[0]})

            elif class_name[0] == "try_again":
                return jsonify({"communication_success":True,'class_name':class_name[0]})

            elif class_name[0] == "try_again_without":
                return jsonify({"communication_success": True,'class_name': class_name[0]})

            elif class_name[0] == "waste":
                return jsonify({"communication_success": True,'class_name': class_name[0]})


class Predict_Model_Paper_Inside(Resource):
    def post(self):
        if request.method == 'POST':
            file = request.get_data()
            if not file : return jsonify({"communication_success" : False})
            class_name = get_prediction_paper_inside(image_bytes=file)
            if class_name[0] == "paper_inside":
                return jsonify({"communication_success":True,'class_name': class_name[0]})

            elif class_name[0] == "try_again":
                return jsonify({"communication_success":True,'class_name':class_name[0]})

            elif class_name[0] == "try_again_without":
                return jsonify({"communication_success": True,'class_name': class_name[0]})

            elif class_name[0] == "waste":
                return jsonify({"communication_success": True,'class_name': class_name[0]})

api.add_resource(Predict_Model_Plastic,'/plastic')
api.add_resource(Predict_Model_Paper,'/paper')
api.add_resource(Predict_Model_Paper_Inside,'/paperinside')

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8000,debug=False)
