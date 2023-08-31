import numpy as np
import cv2
import pickle
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import xgboost as xgb
#working for decision and xgboost
# Load the model
model = pickle.load(open("C:\\Users\\HIKI\\Desktop\\a_python\\image_api\\xgb_backup.pickle", "rb"))

st.title('Alphabet Recognizer')
st.markdown('''
Write an Alphabet 
''')

SIZE = 256

# Create a canvas component
canvas_result = st_canvas(
    fill_color="#ffffff",
    stroke_width=10,
    stroke_color="#ffffff",
    background_color="#000000",
    height=150,
    width=150,
    drawing_mode='freedraw',
    key="canvas"
)

if canvas_result.image_data is not None:
    img = cv2.resize(canvas_result.image_data.astype('uint8'), (28, 28))
    img_rescaling = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
    st.write('Input Image')
    st.image(img_rescaling)

if st.button('Predict'):
    test_x = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    test_x = test_x.flatten()  # Flatten the image to a 1D array
    pred = model.predict(test_x.reshape(1, -1))  # Reshape the flattened array to match the model's input shape
    
    predicted_alphabet = chr(int(pred[0]) + 65)  # Convert the predicted label to the corresponding alphabet
    
    st.write(f"Result: {predicted_alphabet}")
