# 🍎 Calorie Estimation with Object Detection using YOLOv4 and Flask

This project uses a custom-trained YOLOv4 object detection model to detect various fruits and vegetables in real-time from a webcam feed, estimate their weight based on bounding box area, and calculate their approximate calorie content.

<p align="center">
  <img src="https://img.shields.io/badge/Framework-Flask-blue" />
  <img src="https://img.shields.io/badge/Model-YOLOv4-brightgreen" />
  <img src="https://img.shields.io/badge/Language-Python-yellow" />
</p>

---

## 🚀 Features

- Real-time object detection via webcam
- Detects food items like Apple, Banana, Carrot, Onion, etc.
- Calculates estimated **weight** using bounding box size
- Computes **calorie count** based on standard values per 100g
- Adjustable confidence threshold via POST request
- Web interface using **Flask**

---

## 📦 Requirements

Install the dependencies using:
pip install -r requirements.txt
Your requirements.txt should include:
flask
opencv-python
numpy


## 🧠 Model Info

This project uses a custom-trained YOLOv4 model. You need the following files to run it:

| File | Description | Download |
|------|-------------|----------|
| `custom-yolov4-detector.cfg` | YOLOv4 configuration file | [⬇️ Download](https://your-download-link.com/custom-yolov4-detector.cfg) |
| `custom-yolov4-detector_last(3500--98.61).weights` | Trained weights file | [⬇️ Download](https://your-download-link.com/custom-yolov4-detector_last.weights) |
| `darknet_Yolov4_obj_names.names` | Class label names | [⬇️ Download](https://your-download-link.com/darknet_Yolov4_obj_names.names) |

> ⚠️ **Important:** Ensure these files are placed in the **root directory** of the project before running the app.


📂 File Structure
├── app.py                        # Main Flask application
├── templates/
│   └── index.html               # Web interface
├── custom-yolov4-detector.cfg   # YOLOv4 config
├── custom-yolov4-detector_*.weights # Trained model weights
├── darknet_Yolov4_obj_names.names  # Class labels
├── static/                      # (Optional) Static files (CSS, JS, etc.)
├── requirements.txt             # Project dependencies
└── README.md


🖥️ Usage
Start the server:
python app.py
Visit in your browser:http://127.0.0.1:5000/
Live detection starts automatically.

🎯 Confidence Adjustment API
Send a POST request to /set_confidence to adjust the YOLO detection confidence threshold:
curl -X POST -F "confidence=0.6" http://127.0.0.1:5000/set_confidence

🧮 Calorie Estimation Logic
Area of the bounding box → estimated weight using item-specific ratio
Calories = (calories_per_100g / 100) × estimated weight
The calorie values per 100g and area-to-weight ratios are predefined for each class in the code.

🍌 Supported Items
Apple 🍎
Banana 🍌
Carrot 🥕
Onion 🧅
Orange 🍊
Tomato 🍅
Kiwi 🥝

📸 Sample Output
Detected objects will have:
Bounding box
Label
Estimated weight
Estimated calories
Confidence percentage

Example:
Apple: 79.6 kcal (153.0g) 92%


📌 Notes
Make sure your camera is connected and accessible.
This project requires a trained YOLOv4 model. If you want to train your own, use Darknet or any preferred training framework and update the weights/config files accordingly.

🧑‍💻 Author
Ch Atul Kumar Prusty
📫 [chatulprusty@gmail.com]
🔗 [https://github.com/ChAtulKumarPrusty/]
🔗 [https://www.linkedin.com/in/chatulkumarprusty/]
🔗 [https://my-portfolio-xi-ochre-74.vercel.app/]



---

Let me know if you'd like a matching `index.html`, a banner image for the README, or to auto-generate a `requirements.txt` from your project.

