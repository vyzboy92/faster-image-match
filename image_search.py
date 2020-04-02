from elasticsearch.helpers import bulk
from flask import Flask, request, jsonify
from keras.preprocessing import image
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from elasticsearch import Elasticsearch
from elasticsearch.exceptions import ConnectionError, NotFoundError

import os
import numpy as np
import cv2
import tensorflow as tf

app = Flask(__name__)

client = Elasticsearch('localhost:9200')

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

graph = tf.get_default_graph()
model = InceptionV3(weights='imagenet', include_top=True)

SEARCH_SIZE = 1


def nn_features(img):
    imgs = cv2.resize(img, (299, 299))
    img_data = image.img_to_array(imgs)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    with graph.as_default():
        inception_v3_feature = model.predict(img_data)
    inception_v3_feature_np = np.array(inception_v3_feature).flatten()
    return inception_v3_feature_np


@app.route('/add_image', methods=['POST'])
def add_image():
    global graph
    images = request.files['image'].read()
    tag = request.form['tag']
    index_name = request.form['name_of_index']
    images = np.fromstring(images, np.uint8)
    images = cv2.imdecode(images, cv2.IMREAD_COLOR)
    vec = nn_features(images)
    doc = {
            '_op_type': 'index',
            '_index': index_name,
            'title': tag,
            'vector': vec.tolist()
    }
    try:
        bulk(client, [doc])
        return {'success': True}
    except Exception as e:
        return {'success': False}


@app.route('/match_image', methods=['POST'])
def match_image():
    global graph
    image = request.files['image'].read()
    index_name = request.form["index_name"]
    image = np.fromstring(image, np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    vec = nn_features(image)

    script_query = {
        "script_score": {
            "query": {"match_all": {}},
            "script": {
                "source": "cosineSimilarity(params.query_vector, doc['vector']) + 1.0",
                "params": {"query_vector": vec.tolist()}
            }
        }
    }

    try:
        response = client.search(
            index=index_name,  # name of the index
            body={
                "size": SEARCH_SIZE,
                "query": script_query,
                "_source": {"includes": ["title"]}
            }
        )
        tag = [hit['_source']['title'] for hit in response['hits']['hits']
               if hit['_score'] == response['hits']['max_score']][0]

        return {'success': True, 'tag': tag}

    except ConnectionError:
        print("[WARNING] docker isn't up and running!")
        return {'success': False, 'tag': None}
    except NotFoundError:
        print("[WARNING] no such index!")
        return {'success': False, 'tag': None}


app.run(host='0.0.0.0', port=4323, debug=True, use_reloader=False)

