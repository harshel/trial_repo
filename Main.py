import streamlit as st
import cv2
import numpy as np
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
from sklearn.neighbors import NearestNeighbors
import joblib
import os
from PIL import Image
from enum import Enum
import phonetics
import json
import base64

#Function to load the model
def load_nearest_neighbors_model():
    try:
        # Load the saved NearestNeighbors model
        model_path = 'model.pkl'
        nn = joblib.load(model_path)
        return nn
    except Exception as e:
        print(f"Error while loading the model: {e}")
        return None

# Enum class for search criteria
class SearchCriteria(Enum):
    START_WITH = "Start with"
    CONTAINS = "Contains"
    MATCH_WITH = "Match with"
    PHONETICS = "Phonetics match"


#Json file to save/store the user selected ignore words
IGNORE_FILE = "Ignore_Words.json"


# Function to search for matching logo images based on text and search criteria
def search_logo_images_by_text_1(text, img_folder, Img_to_text_folder, search_criteria):
    matching_images = []

    # Get the list of text files in the text folder
    text_files = os.listdir(Img_to_text_folder)

    # Calculate the input word's metaphone
    input_word_metaphone = phonetics.metaphone(text.lower())

    # Iterate over the text files and check for matching text based on the selected search criteria
    for text_file in text_files:
        text_path = os.path.join(Img_to_text_folder, text_file)
        with open(text_path, 'r', encoding='utf-8') as file:
            file_text = file.read()
            file_text_words = file_text.split()

            if search_criteria == SearchCriteria.START_WITH.value:
                if file_text.lower().startswith(text.lower()):
                    # Get the corresponding image file name based on the text file name
                    image_file = text_file.replace('.txt', '.png')
                    image_path = os.path.join(img_folder, image_file)
                    if os.path.exists(image_path):
                        matching_images.append((image_path, image_file))
            elif search_criteria == SearchCriteria.CONTAINS.value:
                if text.lower() in file_text.lower():
                    # Get the corresponding image file name based on the text file name
                    image_file = text_file.replace('.txt', '.png')
                    image_path = os.path.join(img_folder, image_file)
                    if os.path.exists(image_path):
                        matching_images.append((image_path, image_file))
            elif search_criteria == SearchCriteria.MATCH_WITH.value:
                if text.lower() == file_text.lower():
                    # Get the corresponding image file name based on the text file name
                    image_file = text_file.replace('.txt', '.png')
                    image_path = os.path.join(img_folder, image_file)
                    if os.path.exists(image_path):
                        matching_images.append((image_path, image_file))
            elif search_criteria == SearchCriteria.PHONETICS.value:
                for word in file_text_words:
                    if text.lower() in word.lower() or input_word_metaphone == phonetics.metaphone(word.lower()):
                        # Get the corresponding image file name based on the text file name
                        image_file = text_file.replace('.txt', '.png')
                        image_path = os.path.join(img_folder, image_file)
                        if os.path.exists(image_path):
                            matching_images.append((image_path, image_file))

    return matching_images

#Function to finding phonetic matching results
def search_logo_images_by_phonetics(text, img_folder, Img_to_text_folder, ignore_words):
    matching_phonetics = []

    # Get the list of text files in the text folder
    text_files = os.listdir(Img_to_text_folder)

    # Calculate the input word's metaphone
    input_word_metaphone = phonetics.metaphone(text.lower())

    # Iterate over the text files and check for matching phonetics
    for text_file in text_files:
        text_path = os.path.join(Img_to_text_folder, text_file)
        with open(text_path, 'r', encoding='utf-8') as file:
            file_text = file.read()
            file_text_words = file_text.split()

            for word in file_text_words:
                if input_word_metaphone == phonetics.metaphone(word.lower()):
                    matching_phonetics.append(word)

    # Exclude ignore words from matching phonetics
    if text in ignore_words:
        ignore_list = ignore_words[text]
        matching_phonetics = [word for word in matching_phonetics if word not in ignore_list]

    return matching_phonetics



#Function to search similer trademarks based on input image
def search_logo_images_by_image(query_image, img_folder, nn, image_files, top_k, text_folder):
    # Load pre-trained VGG16 model (without the top fully-connected layers)
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Preprocess the query image
    query_image = cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB)
    query_image = cv2.resize(query_image, (224, 224))
    query_image = preprocess_input(query_image)
    query_image = np.expand_dims(query_image, axis=0)

    # Extract features
    query_features = base_model.predict(query_image)
    query_features = np.ndarray.flatten(query_features)

    # Find similar images
    distances, indices = nn.kneighbors([query_features], n_neighbors=top_k)

    similar_images = []
    for i in range(top_k):
        similar_image_file = image_files[indices[0][i]]
        similar_image_path = os.path.join(img_folder, similar_image_file)
        if os.path.exists(similar_image_path):
            similar_images.append((similar_image_path, similar_image_file))

    if similar_images:  # If similar_images is not empty
        for index, (similar_image_path, similar_image_file) in enumerate(similar_images):
            # Display the logo image
            logo_image = cv2.imread(similar_image_path)
            st.markdown(f'<h4 style="font-size: 18px;">Image: {index + 1} - {similar_image_file}</h4>', unsafe_allow_html=True)

            st.image(logo_image, channels="BGR",caption=f"Image: {index + 1}", use_column_width=True)

            # Get the corresponding text file name
            page_number = similar_image_file.split("_")[0][4:]
            text_file_name = f"page{page_number}_text.txt"
            text_file_path = os.path.join(text_folder, text_file_name)

            # Display the corresponding extracted text
            if os.path.exists(text_file_path):
                with open(text_file_path, "r", encoding="utf-8") as f:
                    extracted_text = f.read()
                st.markdown(f"**Text File Name:** {text_file_name}")
                st.markdown(f"**Extracted Text:**\n{extracted_text}")
            else:
                st.markdown("**Text File Not Found**")
            st.markdown("""<hr style="height:3px;border:none;color:#333;background-color:#333;" /> """,
                        unsafe_allow_html=True)
    else:
        st.markdown(f'<span style="font-size: 20px; font-style: italic;">No matching results found for uploaded image</span>', unsafe_allow_html=True)


    return similar_images, indices, distances


#Function to load ignore words
def load_ignore_words():
    if os.path.exists(IGNORE_FILE):
        with open(IGNORE_FILE, "r") as file:
            try:
                ignore_words = json.load(file)
            except json.JSONDecodeError:
                ignore_words = {}
    else:
        ignore_words = {}
    return ignore_words


#Function to save ignore words
def save_ignore_words(ignore_words):
    with open(IGNORE_FILE, "w") as file:
        json.dump(ignore_words, file)


#Function to download output result
def download_output_to_text(output_images, text_folder):
    # Get the text results as a string
    text_results = ""
    for image_path, image_file in output_images:
        page_number = image_file.split("_")[0][4:]
        text_file_name = f"page{page_number}_text.txt"
        text_file_path = os.path.join(text_folder, text_file_name)

        # Get the corresponding extracted text
        extracted_text = ""
        if os.path.exists(text_file_path):
            with open(text_file_path, "r", encoding="utf-8") as f:
                extracted_text = f.read().strip()

        text_results += f"Image name: {image_file}\n"
        text_results += f"Text file name: {text_file_name}\n"
        text_results += f"Extracted Text:\n{extracted_text}\n\n"

    return text_results

def about():
    st.write("For more information, contact us via email at [info@optimumdataanalytics.com](mailto:info@optimumdataanalytics.com).")

def casstudy():
    st.write("Request us for Case Study via email at [info@optimumdataanalytics.com](mailto:info@optimumdataanalytics.com).")

def ppt():
    st.write("Request us for Demo PPT, via email at [info@optimumdataanalytics.com](mailto:info@optimumdataanalytics.com).")

#Define main function
def main():
    st.sidebar.title("Settings")
    st.sidebar.radio("Select Device", ['cpu', 'cuda'], disabled=True, index=0)
    st.sidebar.slider('Confidence', disabled=True, min_value=0.01, max_value=1.0, value=.21)
    st.sidebar.radio("Select input type: ", ['image', 'video'], disabled=True)

    col1, col2= st.columns([90, 10])
    with col1:
        st.markdown("<h1 style='font-size: 40px;'>Trademark Conflict Identification</h1>", unsafe_allow_html=True)

        st.markdown('<h4 style="text-align: center; color: #008000;">AI Enabled Search Tool</h4>', unsafe_allow_html=True)


    with col2:
        # st.image("ODA_logo_2.png")
        image_path = "Plain_ODA.png"
        hyperlink = "https://www.optimumdataanalytics.com/"

        st.markdown(
            f'<a href="{hyperlink}"><img src="data:image/png;base64,{base64.b64encode(open(image_path, "rb").read()).decode()}" alt="ODA Logo" style="width: 130px; height: auto;"></a>',
            unsafe_allow_html=True
        )

    Home, Case_Study, Demo_PPT, About = st.tabs(["Home", "Case Study", "Demo PPT", "Contact"])

    with Home:
        # Load the NearestNeighbors model
        nn = load_nearest_neighbors_model()

        # Set the path our images, text & image_to_text folders
        img_folder = 'images_24'

        text_folder = "text_24"

        Img_to_text_folder = 'Img_to_text_24'

        image_files = [file for file in os.listdir(img_folder) if file.endswith('.jpg') or file.endswith('.png')]

        # Load ignore words
        ignore_words = load_ignore_words()

        # Search Method Selection
        search_method = st.sidebar.radio("Search Method", ("Upload Image", "Search by Text"))

        if search_method == "Upload Image":
            st.sidebar.title("Upload an Image and Search")
            match_type = st.sidebar.selectbox("Matching Type", ("Exact Image Matching", "Partial Image Matching"))
            uploaded_file = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

            if uploaded_file is not None:
                # Read the uploaded image
                image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)

                # Display the uploaded image
                st.image(image, channels="BGR", caption='Uploaded Image', use_column_width=True)

                if st.sidebar.button("Search by Image", key="search_image_button"):

                    if match_type == "Exact Image Matching":
                        st.markdown("__________")
                        # Exact Image Matching
                        st.markdown('<h3 style="text-align: left; color: navy;">Exact Matched Image:</h3>',
                                    unsafe_allow_html=True)

                        similar_images, indices, distances = search_logo_images_by_image(image, img_folder, nn,
                                                                                         image_files,
                                                                                         top_k=1,
                                                                                         text_folder=text_folder)

                        exact_image_path = os.path.join(img_folder, similar_images[0][1])

                    else:
                        st.markdown("____")
                        # Partial Image Matching
                        st.markdown('<h3 style="text-align: left; color: navy;">Partially Matched Images:</h3>',
                                    unsafe_allow_html=True)
                        #st.header("Partial Image Matching")
                        similar_images, indices, distances = search_logo_images_by_image(image, img_folder, nn,
                                                                                         image_files,
                                                                                         top_k=5,
                                                                                         text_folder=text_folder)

                    # Download the output
                    if st.download_button("Download Results",
                                            data=download_output_to_text(similar_images, text_folder),
                                            file_name="search_results.txt", mime="text/plain"):
                        st.success("Text search results have been saved to: search_results.txt")

        else:
            st.sidebar.title("Enter the Text & Search")
            text_search_option = st.sidebar.selectbox("Search Criteria", (SearchCriteria.START_WITH.value,
                                                                          SearchCriteria.CONTAINS.value,
                                                                          SearchCriteria.MATCH_WITH.value,
                                                                          SearchCriteria.PHONETICS.value))

            if text_search_option == SearchCriteria.PHONETICS.value:
                search_text = st.sidebar.text_input("Enter text", key="phonetics_text_input")
                if search_text:
                    #img_folder = "images_24"
                    #text_folder = "Img_to_text_24"

                    matching_phonetics = search_logo_images_by_phonetics(search_text, img_folder, Img_to_text_folder,
                                                                         ignore_words)


                    if matching_phonetics:

                        st.markdown(
                            f'<h3 style="text-align: left; color: navy;">Results based on <u>{text_search_option}</u> search criteria:</h3>',
                            unsafe_allow_html=True)

                        st.markdown(
                            '<font color="#000000">**Step 1: Search phonetic matched words and display them for selection purpose**</font>',
                            unsafe_allow_html=True)

                        ignore_list = ignore_words.get(search_text, [])
                        selected_checkboxes = []  # List to store the selected checkboxes' values

                        for index, phonetic_word in enumerate(matching_phonetics):
                            checkbox_selected = st.checkbox(label=phonetic_word,
                                                            key=f"checkbox_{search_text}_{index}")
                            if checkbox_selected:
                                if phonetic_word not in ignore_list and phonetic_word not in selected_checkboxes:
                                    selected_checkboxes.append(phonetic_word)

                        st.markdown(
                            '<font color="#FF0000">**Note: Selected words will be excluded in the future searches**</font>',
                            unsafe_allow_html=True)
                        st.markdown(
                            '<font color="#000000">**Step 2: Select & save the unwanted phonetics matched words from the above results**</font>',
                            unsafe_allow_html=True)
                        if st.button("Select & save unwanted phonetics matched words"):
                            if selected_checkboxes:
                                ignore_words[search_text] = ignore_list + selected_checkboxes
                                save_ignore_words(ignore_words)
                                st.write("_Unwanted phonetics keywords saved successfully._")
                            else:
                                st.write("_No new unwanted phonetics keywords selected._")

                            # Get the remaining words after removing the ignored ones
                            remaining_words = [word for word in matching_phonetics if
                                               word not in ignore_words.get(search_text, [])]

                            st.markdown(
                                '<font color="#000000">**Step 3: Display list of reliable phonetics matched words & respective results**</font>',
                                unsafe_allow_html=True)
                            # Store all matching images and extracted text
                            all_matching_images = []
                            if remaining_words or not ignore_words.get(search_text):
                                for word in remaining_words:
                                    # Display each word without the index number and square brackets.
                                    st.write(word)

                                st.markdown("_________________")

                            # Display results for remaining words
                            if remaining_words or not ignore_words.get(search_text):
                                for word in remaining_words:
                                    if word in matching_phonetics:
                                        # Search for matching logo images based on the current word
                                        matching_images = search_logo_images_by_text_1(word, img_folder, Img_to_text_folder,
                                                                                     SearchCriteria.CONTAINS.value)

                                        all_matching_images.extend(matching_images)

                                        if matching_images:
                                            #st.subheader(f"Word: {word}")
                                            for index, (image_path, image_file) in enumerate(matching_images):
                                                logo_image = cv2.imread(image_path)
                                                st.markdown(
                                                    f'<h4 style="font-size: 18px;">Image: {index + 1} - {image_file}</h4>',
                                                    unsafe_allow_html=True)

                                                #st.subheader(f"Image: {index + 1} - {image_file}")
                                                st.image(logo_image, channels="BGR", use_column_width=True)

                                                # Get the corresponding text file name
                                                page_number = image_file.split("_")[0][4:]
                                                text_file_name = f"page{page_number}_text.txt"
                                                text_file_path = os.path.join(text_folder, text_file_name)

                                                # Display the corresponding extracted text
                                                if os.path.exists(text_file_path):
                                                    with open(text_file_path, "r", encoding="utf-8") as f:
                                                        extracted_text = f.read()
                                                    st.markdown(f"**Text File Name:** {text_file_name}")
                                                    st.markdown(f"**Extracted Text:**\n{extracted_text}")

                                                else:
                                                    st.markdown(f"**Text File Not Found for Image:** {image_file}")

                                                st.markdown(
                                                    """<hr style="height:3px;border:none;color:#333;background-color:#333;" /> """,
                                                    unsafe_allow_html=True)
                                        else:
                                            st.write(f"No matching images found for word: {word}")

                                # Download the output

                                if st.download_button("Download Results",
                                                                    data=download_output_to_text(all_matching_images,
                                                                                                text_folder),
                                                                    file_name="search_results.txt", mime="text/plain"):
                                    st.success("Text search results have been saved to: search_results.txt")


                            else:
                                st.write("No matching phonetics found or all matching phonetics have been ignored.")
                    else:
                        # Display a message for no matching phonetics
                        st.markdown(f'<span style="font-size: 20px; font-style: italic;">No matching phonetics found for <u><b>{search_text}</b></u></span>', unsafe_allow_html=True)


            else:
                search_text = st.sidebar.text_input("Enter text", key="text_input")

                if st.sidebar.button("Search by Text", key="search_text_button"):
                    # Search for matching logo images based on text
                    #Img_to_text_folder = 'Img_to_text_24'
                    matching_images = search_logo_images_by_text_1(search_text, img_folder, Img_to_text_folder,
                                                                 text_search_option)

                    if matching_images:  # If matching_images is not empty


                        st.markdown(f'<h3 style="text-align: left; color: navy;">Results based on <u>{text_search_option}</u> search criteria:</h3>',
                                    unsafe_allow_html=True)
                        for index, (image_path, image_file) in enumerate(matching_images):
                            # Display the logo image
                            logo_image = cv2.imread(image_path)
                            st.markdown(f'<h4 style="font-size: 18px;">Image: {index + 1} - {image_file}</h4>',
                                        unsafe_allow_html=True)

                            #st.subheader(f"Image: {index + 1} - {image_file}")
                            st.image(logo_image, channels="BGR", use_column_width=True)

                            # Get the corresponding text file name
                            page_number = image_file.split("_")[0][4:]
                            text_file_name = f"page{page_number}_text.txt"
                            text_file_path = os.path.join(text_folder, text_file_name)

                            # Display the corresponding extracted text
                            if os.path.exists(text_file_path):
                                with open(text_file_path, "r", encoding="utf-8") as f:
                                    extracted_text = f.read()
                                st.markdown(f"**Text File Name:** {text_file_name}")
                                st.markdown(f"**Extracted Text:**\n{extracted_text}")
                            else:
                                st.markdown(f"**Text File Not Found for Image:** {image_file}")

                            st.markdown("""<hr style="height:3px;border:none;color:#333;background-color:#333;" /> """,
                                        unsafe_allow_html=True)

                        #Download the output

                        if st.download_button("Download Results",
                                              data=download_output_to_text(matching_images, text_folder),
                                              file_name="search_results.txt", mime="text/plain"):
                            st.success("Text search results have been saved to: search_results.txt")

                    else:
                        st.markdown(f'<span style="font-size: 20px; font-style: italic;">No matching Results found for <u><b>{search_text}</b></u></span>', unsafe_allow_html=True)

    with About:
        about()
    with Case_Study:
        casstudy()
    with Demo_PPT:
         ppt()

if __name__ == "__main__":
    main()
