from dotenv import load_dotenv

load_dotenv()  # Load all the environment variables from .env

import streamlit as st
import os
from PIL import Image
import google.generativeai as genai
import tensorflow as tf
import numpy as np

def model_prediction(test_image) :
    model=tf.keras.models.load_model('eye_model.h5')
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(256,256))
    input_arr=tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    prediction=model.predict(input_arr)
    result_index=np.argmax(prediction)
    return result_index
    

# Configure the Google Generative AI with the API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize the Gemini Pro Vision model
model = genai.GenerativeModel('gemini-pro-vision')

def get_gemini_response(input, image, prompt):
    # Generate content based on the provided input, image, and prompt
    response = model.generate_content([input, image[0], prompt])
    return response.text


def input_image_setup(image_path):
    # Check if the file path exists
    if os.path.exists(image_path):
        # Open the image from the specified path
        image = Image.open(image_path)
        
        # Convert the image into bytes data
        with open(image_path, "rb") as f:
            bytes_data = f.read()
        
        image_parts = [
            {
                "mime_type": "image/jpeg",  # Assuming JPEG format, adjust if needed
                "data": bytes_data
            }
        ]
        return image_parts, image  # Return both the bytes data and the PIL image
    else:
        raise FileNotFoundError("Image file not found at the specified path")

#Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

#Home Page
if(app_mode=="Home"):
    st.header("HEALTH DISEASE RECOGNITION SYSTEM")
    image_path = "home_page.jpg"
    st.image(image_path,use_column_width=True)
    st.markdown(""" Welcome to the Health Disease Recognition System! 🔍
    
    Our mission is to help in identifying health diseases efficiently. Upload an image of a disease, and our system will analyze it to detect any signs of diseases. Together, let's protect our health and ensure a healthier life!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a health with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Health Disease Recognition System!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page. """)

#About Project
elif(app_mode=="About"):
    st.header("About")
    st.markdown("""
                #### About Dataset
                This dataset is recreated using offline augmentation from the original dataset.The original dataset can be found on this github repo.
                This dataset consists of about 4K rgb images of healthy and diseased crop leaves which is categorized into different classes.The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure.
                A new directory containing 50 test images is created later for prediction purpose.
                #### Content
                1. train (4000 images)
                2. test (50 images)
                3. validation (950 images)""")



#Prediction Page
elif(app_mode=="Disease Recognition"):
    st.header("Disease Recognition")
    test_image=st.file_uploader("Choose as image: ")
    if(st.button("Show Image")):
        st.image(test_image,width=4,use_column_width=True)
    if(st.button("Predict")):
        st.write("Our Prediction")
        result_index=model_prediction(test_image)
        class_names=['cataract', 'diabetic_retinopathy', 'glaucoma', 'normal']
        st.success("Model is Predicting it's a {}".format(class_names[result_index]))
        if class_names[result_index] == 'cataract':
                st.write("Treatment",['When your prescription glasses can not clear your vision the only effective treatment for cataracts is surgery.'],
                     "When to consider cataract surgery",["Cataract surgery is typically performed using local anesthesia, so you will be awake but your eye will be numbed. The surgeon will make a small incision in the eye and use ultrasound energy to break up the cloudy lens, which is then removed from the eye. The artificial lens is then implanted in its place."])
        elif class_names[result_index] == 'diabetic_retinopathy':
                     st.write("Treatment",['Proper management of blood sugar levels is crucial in preventing or slowing the progression of diabetic retinopathy. This involves adhering to a healthy diet, regular exercise, and taking prescribed medications as directed by a healthcare provider.',
                                           ' Intravitreal injections of anti-VEGF drugs may be used to reduce swelling.','Laser treatment can help seal leaking blood vessels or reduce abnormal vessel growth.'])

        elif class_names[result_index] == 'glaucoma':
             st.write("Treatment",['The main treatment for glaucoma is eyedrops, which reduce pressure in the eyes. They are usually used 1–4 times a day and should be used as directed, even if you do not notice any problems with your vision. Treatment often starts with prescription eye drops.'],"Eye Drops",
                      ['Xalatan','Travatan Z','Zioptan','Lumigan','Unoprostone isopropyl'])
        
        elif class_names[result_index] == 'normal':
             st.write("YOUR EYES HEALTH IS GOOD")
        else:
                st.success("Model is Predicting it's a {}".format(class_names[result_index]))
                     
                 
    input_prompt = """You are an expert in understanding medical images.
You will receive input images as input & can be of Alzheimer person, skin infections etc. I will tell you about what the disease is, and you have to tell what this disease is and how we can cure it.
"""

# User inputs
input = 'cataract'
image_path = r'E:\Health_care\eye_data\cataract\_0_4015166.jpg'
submit = st.button("Tell me about the image")

# If submit button is clicked
if submit:
    try:
        image_data, image = input_image_setup(image_path)
        response = get_gemini_response(input_prompt, image_data, input)
        
        st.subheader("The Response is")
        st.write(response)
        
        st.subheader("Input Image")
        st.image(image, caption="Input Image", use_column_width=True)
    except FileNotFoundError as e:
        st.error(str(e))

