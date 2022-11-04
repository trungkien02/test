import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
act_model = load_model("./model_v2.h5")
char_list = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789\
			 ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝàáâãèéêìíòóôõùúýĂăĐđĨĩŨũƠơƯưẠạẢảẤấẦầẨẩẪẫẬậẮắẰằẲẳẴẵẶặ\
			 ẸẹẺẻẼẽẾếỀềỂểỄễỆệỈỉỊịỌọỎỏỐốỒồỔổỖỗỘộỚớỜờỞởỠỡỢợỤụỦủỨứỪừỬửỮữỰựỲỳỴỵỶỷỸỹ "
def resize_(img): 
    w, h = img.shape
    if (32/w < 128/h):
        scale_percent = 32/w
    else:
        scale_percent = 128/h
    height = int(img.shape[0] * scale_percent)
    width = int(img.shape[1] * scale_percent)
    dim = (width, height)
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

    return img
    
def reshape_expand_dim_(img):
    w, h = img.shape
    if w < 32:
            add_zeros = np.ones((32-w, h))*210
            img = np.concatenate((img, add_zeros))
    if h < 128:
        add_zeros = np.ones((32, 128-h))*210
        img = np.concatenate((img, add_zeros), axis=1)
    img = np.expand_dims(img , axis = 2)
    print(img.shape)

    return img

    return img
def predict(img):

    img = img[None, :, :, :]
    prediction = act_model.predict(img)
    
    out = K.get_value(K.ctc_decode(prediction, input_length=np.ones(prediction.shape[0])*prediction.shape[1],
                             greedy=True)[0][0])
    predicted_str = ''
    for x in (out):
        for p in x:  
            if int(p) != -1:
                predicted_str += char_list[int(p)]    
    return predicted_str

st.title("OCR")
st.write("This is a simple OCR app")
uploaded_file = st.file_uploader("Choose an image...", type="png")
if uploaded_file is not None:
    img = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = resize_(img)
    img = reshape_expand_dim_(img)
    img = img/255.
    img = np.array(img)
    st.image(img, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Predicted Text:")
    st.write(predict(img))