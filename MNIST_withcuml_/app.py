import os
import numpy as np
import cv2
import streamlit as st
import pickle
from streamlit_drawable_canvas import st_canvas
def load_model():
    with open('C:\\Users\\HIKI\\Desktop\\a_python\\image_api\\xgb_backup.pickle', 'rb') as file:
        data = pickle.load(file)
    return data
data = load_model()
st.title('My Digit Recognizer')
st.markdown('''
Try to write a digit!
''')

# data = np.random.rand(28,28)
# img = cv2.resize(data, (256, 256), interpolation=cv2.INTER_NEAREST)

SIZE = 192
mode = st.checkbox("Draw (or Delete)?", True)
canvas_result = st_canvas(
    fill_color='#000000',
    stroke_width=20,
    stroke_color='#FFFFFF',
    background_color='#000000',
    width=SIZE,
    height=SIZE,
    drawing_mode="freedraw" if mode else "transform",
    key='canvas')

if canvas_result.image_data is not None:
    img = cv2.resize(canvas_result.image_data.astype('uint8'), (28, 28))
    rescaled = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
    st.write('Model Input')
    st.image(rescaled)

if st.button('Predict'):
    test_x = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    val = data.predict(test_x.reshape(1, 784))

    # Ensure val is a one-dimensional array
    val = val.flatten()

    # Convert the NumPy array to a Pandas DataFrame
    df = pd.DataFrame(val, columns=["Values"])

    st.write(f'result: {np.argmax(val)}')
    st.bar_chart(df)
