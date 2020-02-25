#!/usr/bin/env python

import flask
from flask import request, jsonify, Response, send_file
from os import path, environ
from raven.contrib.flask import Sentry
import cv2
import numpy as np
import requests
import json
from datetime import datetime
from auth import token_required
from lib.detection_model import load_net, detect
import ptvsd
from io import BytesIO
from PIL import Image

THRESH = 0.08  # The threshold for a box to be considered a positive detection
SESSION_TTL_SECONDS = 60*2

app = flask.Flask(__name__)

# SECURITY WARNING: don't run with debug turned on in production!
app.config['DEBUG'] = (environ.get('DEBUG') == 'True')
debug = app.config['DEBUG']

if __name__ != '__main__':
    # setup remote debugger
    if debug:
        print('Attaching PTVSD debugging port')
        ptvsd.enable_attach(address = ('0.0.0.0', 3002), redirect_output=True)

# Sentry
sentry = None
if environ.get('SENTRY_DSN'):
    sentry = Sentry(app, dsn=environ.get('SENTRY_DSN'))

# get model, and settings from env
model_xml = environ.get("MODEL_XML", None)
if model_xml is None:
    model_xml = path.join(path.join(path.dirname(path.realpath(__file__)), 'model'), 'model.xml')

model_labels  = environ.get("MODEL_LABELS", None)
if model_labels is None:
    model_labels = path.join(path.join(path.dirname(path.realpath(__file__)), 'model'), 'model.labels')

device  = environ.get("OPENVINO_DEVICE", "CPU")
cpu_extension  = environ.get("OPENVINO_CPU_EXTENSION", None)

model = load_net(model_xml, model_labels, device, cpu_extension, debug)

@app.route('/p/', methods=['GET'])
#@token_required
def get_p():
    if 'img' in request.args:
        try:
            start = datetime.now()
            show = 'show' in request.args
            url = request.args['img']
            if url.startswith('http'):
                resp = requests.get(url, stream=True, timeout=(0.5, 5))
                resp.raise_for_status()
                img_array = np.array(bytearray(resp.content), dtype=np.uint8)
            else:
                img_path = path.join(path.dirname(path.realpath(__file__)), url)
                if debug: print('path %s' % img_path)
                with open(img_path, 'rb') as infile:
                    buf = infile.read()
                img_array = np.fromstring(buf, dtype='uint8')

            img = cv2.imdecode(img_array, -1)
            imageload = datetime.now() - start

            start = datetime.now()
            result = detect(model, img, thresh=THRESH, show=show, debug=debug)
            timetaken = datetime.now() - start

            if show:
                img = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                img2 = BytesIO()
                img.save(img2, format="jpeg")
                img2.seek(0)
                return send_file(img2, mimetype='image/jpeg')
            else:
                result = {'image-load': str(imageload), 'time-taken': str(timetaken),'detections': result}
                if debug and len(result) > 0: print(json.dumps(result))
                return jsonify(result)

        except Exception as ex:
            if sentry:
                sentry.captureException()
            return jsonify({'detections':[], 'error': '%s' % ex})
    else:
        print("Invalid request params: {}".format(request.args))

    return jsonify({'detections': []})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=3333, threaded=False)
