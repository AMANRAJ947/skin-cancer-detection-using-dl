from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# ✅ Load your model (IMPORTANT: correct name)
model = load_model("skin_cancer_model.h5")

UPLOAD_FOLDER = "static/uploads/"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# 🔍 Prediction function
def predict(img_path):
    img = image.load_img(img_path, target_size=(224,224))  # ✅ FIXED
    img_array = image.img_to_array(img)

    img_array = np.expand_dims(img_array, axis=0)
    
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    img_array = preprocess_input(img_array)   # ✅ IMPORTANT

    result = model.predict(img_array)
    prob = result[0][0]

    if prob > 0.5:
        return f"Cancer ({prob*100:.2f}%)"
    else:
        return f"Non-Cancer ({(1-prob)*100:.2f}%)"


# 🌐 Route
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    img_path = None

    if request.method == "POST":
        file = request.files["file"]
        
        if file:
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(img_path)

            prediction = predict(img_path)

    return render_template("index.html", prediction=prediction, img_path=img_path)


# 🚀 Run server
if __name__ == "__main__":
    import os

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
    