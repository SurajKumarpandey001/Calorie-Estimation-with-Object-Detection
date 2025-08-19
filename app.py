from flask import Flask, render_template, Response, request, jsonify
import cv2 as cv
import numpy as np

app = Flask(__name__)

cap_index = 0
cap = cv.VideoCapture(cap_index)
confThreshold = 0.5
whT = 320
nmsThreshold = 0.3
calorie_dict_per_100g = {
    "Apple": 52,
    "Banana": 89,
    "Carrot": 41,
    "Onion": 40,
    "Orange": 47,
    "Tomato": 18,
    "Kiwi": 44,
}
area_to_weight_ratio = {
    "Apple": 0.012,
    "Banana": 0.014,
    "Carrot": 0.010,
    "Onion": 0.013,
    "Orange": 0.011,
    "Tomato": 0.010,
    "Kiwi": 0.015,
}

modelConfiguration = "custom-yolov4-detector.cfg"
modelWeights = "custom-yolov4-detector_last(3500--98.61).weights"
net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

with open('darknet_Yolov4_obj_names.names', 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

def estimate_weight(w, h, label):
    area = w * h
    ratio = area_to_weight_ratio.get(label, 0.012)
    weight = area * ratio
    return max(50, min(weight, 500))

def findObjects(outputs, img, confThreshold):
    hT, wT, cT = img.shape
    bbox = []
    classIds = []
    confs = []
    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                w, h = int(det[2] * wT), int(det[3] * hT)
                x, y = int((det[0] * wT) - w / 2), int((det[1] * hT) - h / 2)
                bbox.append([x, y, w, h])
                classIds.append(classId)
                confs.append(float(confidence))

    indices = cv.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)

    if len(indices) > 0:
        for i in indices.flatten():
            label = str(classNames[classIds[i]])
            confidence = confs[i]
            estimated_weight = estimate_weight(w, h, label)
            calories_per_100g = calorie_dict_per_100g.get(label, 0)
            estimated_calories = (calories_per_100g / 100) * estimated_weight
            box = bbox[i]
            x, y, w, h = box[0], box[1], box[2], box[3]
            cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
            text = f"{label}: {estimated_calories:.1f} kcal ({estimated_weight:.1f}g) {int(confs[i] * 100)}%"
            cv.putText(img, text,(x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

def generate_frames():
    global cap, confThreshold
    while True:
        success, img = cap.read()
        if not success:
            break

        blob = cv.dnn.blobFromImage(img, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
        net.setInput(blob)
        layersNames = net.getLayerNames()
        outputNames = [layersNames[i - 1] for i in net.getUnconnectedOutLayers()]
        outputs = net.forward(outputNames)

        findObjects(outputs, img, confThreshold)

        ret, buffer = cv.imencode('.jpg', img)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/set_confidence', methods=['POST'])
def set_confidence():
    global confThreshold
    try:
        confThreshold = float(request.form['confidence'])
        print(f"Updated confidence threshold: {confThreshold}")
        return jsonify(success=True)
    except ValueError:
        return jsonify(success=False)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
