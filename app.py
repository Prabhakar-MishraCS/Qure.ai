from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16

# To predict the image
def predict(imagepass):
    model = VGG16()
    image = load_img(imagepass, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    yhat = model.predict(image)
    label = decode_predictions(yhat)
    label = label[0][0]
    return label


import streamlit as st
from PIL import Image
import requests
st.balloons()
def get_image(url):
    img = requests.get(url)
    file = open("sample_image.jpg", "wb")
    file.write(img.content)
    file.close()
    img_file_name = 'sample_image.jpg'
    type(img_file_name)
    return img_file_name

st.title("Image Classification")
st.header("Tech Assessment for Qure.ai")

url = st.text_input("Enter Image Url:")
print("URL ",url)
st.subheader("Or")
uploaded_file = st.file_uploader("Choose a file",type=["jpeg","png"])

if url:
    print("Upload URL")
    image = get_image(url)
    st.image(image)
    classify = st.button("Classify image")
    if classify:
        st.write("")
        st.write("Classifying...")
        label = predict(image)
        st.write('%s (%.2f%%)' % (label[1], label[2]*100))
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image)
    classify = st.button("Classify Image")
    if classify:
        st.write("")
        st.write("Classifying...")
        label = predict(image.tobytes())
        st.write('%s (%.2f%%)' % (label[1], label[2] * 100))


