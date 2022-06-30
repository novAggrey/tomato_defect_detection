import streamlit as st
import streamlit.components.v1 as component
from PIL import Image
import tensorflow
import cv2
import numpy as np
import pandas as pd
pd.set_option("display.max_rows",None)
pd.set_option("display.max_columns",None)
import base64

model = tensorflow.keras.models.load_model("model1.h5")

hide_st_style = """
            <style>
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

def set_bg_hack(main_bg):
    main_bg_ext = "jpg"
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()});
            background-size:cover
        }}
        </style>
        """,
        unsafe_allow_html=True        
    )
set_bg_hack("Images/Background_1.jpg")

heading = """
<div style="background:#F5F5F5; color:gray; border-radius:7px; padding:7px">
<h1 style="text-align:center;color:"">TOMATO DEFECT DETECTION</h1>
</div>
"""
results_0 = """
<div style="background:#F5F5F5; color:gray; border-radius:7px; padding:7px">
<h3 style="text-align:center">Tomato - Healthy</h3>
</div>
"""
results_1 = """
<div style="background:#F5F5F5; color:gray; border-radius:7px; padding:7px">
<h3 style="text-align:center">Tomato - Septoria Disease</h3>
<hr>
<p style="text-align:center">
Septoria leaf spot is caused by a fungus, Septoria lucopersici.
It is one of the most destructive diseases of tomato foliage and is particular severe in areas where wet, humid weather persists for extended periods.
Septoria leaf spot usually appears on the lower laves after the first sets.
</p>
</div>
"""
results_2 = """
<div style="background:#F5F5F5; color:gray; border-radius:7px; padding:7px">
<h3 style="text-align:center">Tomato - Late Blight Disease</h3>
<hr>
<p style="text-align:center">
Late Blight ia a potential devaststing disease of tomato and potato, infecting leaves, stems, tomato fruit and potato tubers.
The disease spreads quickly in fields and can result in total crop failure if untreated.
</p>
</div>
"""
description_2 = """
<div style="background:#F5F5F5; color:gray; border-radius:7px; padding:7px">

</div>
"""
invalid = """
<div style="background:#F5F5F5; color:gray; border-radius:7px; padding:7px">
<h3 style="text-align:center">*** 🚫 INVALID IMAGE 🚫 ***</h3>
</div>
"""
component.html(heading)
file = st.file_uploader("Upload Tomato Leaf Image", type=["PNG","JPG","JPEG"])
if file:
    image = Image.open(file)
    st.image(image, caption='fig. This is the Image you Imported', width=350)
    if st.button("PREDICT"):
        img = tensorflow.keras.preprocessing.image.img_to_array(image)
        if img.shape[1] > 500:
            component.html(invalid)
        else:
            img = cv2.resize(img,(50,50))
            img = tensorflow.expand_dims(img,0)/255
            pred_class = model.predict(img)
            pred = np.argmax(pred_class)

            if pred==0:
                component.html(results_0)

            elif pred==1:
                component.html(results_1)
                    
            elif pred==2:
                component.html(results_2)  