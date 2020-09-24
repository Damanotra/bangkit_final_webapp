import numpy as np
from flask import Flask, jsonify, request, render_template
import requests
import os
from PIL import Image
import googleapiclient.discovery
from google.cloud import vision
import cv2
import io

def localize_objects(img):
    """Localize objects in the local image.

    Args:
    path: The path to the local file.
    """
    
    client = vision.ImageAnnotatorClient()

    # with open(path, 'rb') as image_file:
    #     content = image_file.read()
    image = vision.types.Image(content=img)
    try:
        objects = client.object_localization(
            image=image).localized_object_annotations
    except Exception as e:
        print(e)

    print('Number of objects found: {}'.format(len(objects)))
    for object_ in objects:
        print('\n{} (confidence: {})'.format(object_.name, object_.score))
        print('Normalized bounding polygon vertices: ')
        for vertex in object_.bounding_poly.normalized_vertices:
            print(' - ({}, {})'.format(vertex.x, vertex.y))

def run_inference(img):
    '''receive image as flask file, return np array // bounding box(left,top,w,h), name, and confidence for each object detected'''
    client = vision.ImageAnnotatorClient()
    content = img.read()
    image = vision.types.Image(content=content)
    # image_vision = vision.types.Image(content=img.read())
    # npimg = read_image(img)
    # npimg = cv2.imread(img)
    objects = client.object_localization(
        image=image).localized_object_annotations
    

    return objects
    detections = []
    for object_ in objects:
        detected = {}
        confidence = object_.score
        class_name = object_.name
        w = (object_.bounding_poly.normalized_vertices[0].x - object_.bounding_poly.normalized_vertices[1].x)*width
        h = (object_.bounding_poly.normalized_vertices[0].y - object_.bounding_poly.normalized_vertices[3].y)*height
        left = width*object_.bounding_poly.normalized_vertices[0].x + w/2
        top = height*object_.bounding_poly.normalized_vertices[0].y + h/2
        box = np.array([left, top, w, h])
        detected = {'box':box,'name':class_name,'confidence':confidence}
        detections.append(detected)
    return [(i['box'],i['confidence'],i['name']) for i in detections]




def read_image(file):
    ''' Convert image to numpy array'''
    # img = Image.open(request.files['file'])
    img = Image.open(file)
    img = img.convert('RGB')
    print("converted rgb")
    x = np.array(img)
    print(x.shape)
    return x

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'key.json'

app = Flask(__name__)
# 'bangkit-makeup-e913a86923af.json'

@app.route('/', methods=['GET', 'POST'])
def index():
    #Main Page
    return render_template('index.html')
    
@app.route('/predict', methods=['POST'])
def home():
    # img = read_image(request.files['file'])
    # localize_objects(request.files['file'].read())
    if 'file' not in request.files:
        print("no file part")
    img_ori = request.files['file']
    
    #inference
    objects = run_inference(img_ori)

    #preparing image
    in_memory_file = io.BytesIO()
    img_ori.stream.seek(0)
    img_ori.save(in_memory_file)
    data = np.fromstring(in_memory_file.getvalue(), dtype=np.uint8)
    npimg = cv2.imdecode(data, cv2.IMREAD_COLOR)
    print(npimg.shape)
    height, width, channels = npimg.shape

    

    #grabbing boxes
    detections = []
    for object_ in objects:
        detected = {}
        confidence = object_.score
        class_name = object_.name
        w = (object_.bounding_poly.normalized_vertices[0].x - object_.bounding_poly.normalized_vertices[1].x)*width
        h = (object_.bounding_poly.normalized_vertices[0].y - object_.bounding_poly.normalized_vertices[3].y)*height
        left = width*object_.bounding_poly.normalized_vertices[0].x + w/2
        top = height*object_.bounding_poly.normalized_vertices[0].y + h/2
        box = np.array([left, top, w, h])
        detected = {'box':box,'name':class_name,'confidence':confidence}
        detections.append(detected)
    detected_objects =  [(i['box'],i['confidence'],i['name']) for i in detections]
    
    #drawing

    shift = 40
    img_draw = np.zeros((height + shift, width, channels), dtype=np.uint8)  # this is where drawing happen
    img_draw[shift:] = npimg.copy() #timpa dari gambar

    for box, score, name in detected_objects:
        left, top = box[0], box[1]
        w, h = box[2], box[3]
        right, bottom = left + w, top + h
        text = '{} {:.4f}'.format(name, score)
        start = (left, top + shift)
        end = (right, bottom + shift)
        color = (0, 255, 0)
        img_draw = cv2.rectangle(img_draw, start, end, color, 2)
        img_draw = cv2.putText(img_draw, text, (left, top + shift), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2)

    return 'request sent' 

if __name__ == '__main__':
    app.run(debug = True)
