import glob
import requests
from os import path

# Host IP Address
HOST = "http://192.168.1.6"

# image server base url
IMAGE_URL = HOST + ":8080/%s"

# ml_api servers
ML_API_OPENVINO = HOST +":3333/p/"
ML_API_YOLOv2 = HOST + ":3332/p/"

# validation directories
BASE_DIR = "./images/val_jpgs"
POSITIVES = "positives"
NEGATIVES = "negatives"
FALSE_POSITIVES = "false_alerts"

# set ml_api server to run the test on
ML_API = ML_API_OPENVINO

TEST_DIRS = [
    (POSITIVES, True),
    (NEGATIVES, False),
    (FALSE_POSITIVES, False)
]

results = {
    "TP": 0,
    "FP": 0,
    "TN": 0,
    "FN": 0
}

def analyse(server, dir, ispositive):
    global results
    print("ANALYSE: %s %s" % (dir, ispositive))

    files = glob.glob("%s/%s/*.jpg" % (BASE_DIR, dir))
    num = len(files)
    count = 0
    for f in files:
        count += 1
        file = path.basename(f)

        url = IMAGE_URL % ("%s/%s" % (dir, file))

        payload = {'img': url}
        resp = requests.get(server, params=payload)
        if resp.status_code == 200:
            det = resp.json()
            det = det['detections']
            num_det = len(det)
            result = 'TN'
            if ispositive:
                if num_det > 0:
                    result = 'TP'
                else:
                    result = 'FN'
            elif num_det > 0:
                result = "FP"
            print("%d/%d: %s %s" % (count, num, file, result))

            results[result] += 1
        else:
            print("%d/%d: %s ERROR %d" % (count, num, file, resp.status_code))


for dir, ispositive in TEST_DIRS:
    analyse(ML_API, dir, ispositive)
    print(results)

