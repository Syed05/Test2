import streamlit as st
import torch
import pandas as pd
from PIL import Image, ImageDraw
import base64

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', './best.pt', force_reload=True)

def detect_objects(image):
    # Perform object detection using YOLOv5
    results = model(image)

    # Define colors for each class
    color_map = {
        0: 'red',   # defect class 1
        1: 'blue',  # defect class 2
        2: 'green',  # defect class 3
        3: 'purple'  # defect class 4
    }

    # Define class names
    class_names = {
        0: 'Vertical Crack',
        1: 'Horizontal Crack',
        2: 'Alligator Crack',
        3: 'Pothole'
    }

    # Get bounding boxes, labels and confidences from results
    bboxes = results.xyxy[0][:, :4].cpu().numpy()
    labels = results.xyxy[0][:, -1].cpu().numpy().astype(int)
    confidences = results.xyxy[0][:, 4].cpu().numpy()

    # Draw bounding boxes on image
    img_draw = ImageDraw.Draw(image)
    for bbox, label, confidence in zip(bboxes, labels, confidences):
        # Look up color and class name for current label
        color = color_map.get(label)
        class_name = class_names.get(label)

        # Draw bounding box and class name on image
        img_draw.rectangle(bbox, outline=color, width=3)
        img_draw.text((bbox[0], bbox[1] - 10), f'{class_name} ({confidence:.8f})', fill=color)

    # Convert image to RGB format
    image = image.convert('RGB')

    # Convert results to pandas dataframe
    df = results.pandas().xyxy[0]
    
    # Drop class column and rename label_name column
    df.drop(columns='class', inplace=True)
    df.rename(columns={'label_name': 'label_meaning'}, inplace=True)

    # Add label names to dataframe
    label_names = [class_names[label] for label in labels]
    df['Description'] = label_names

    # Return image and dataframe with results
    return image, df




# Define Streamlit app
def main():
    # Set page title
    st.set_page_config(page_title='Automatic Pavement Defect Detection System')

    # Add logo to sidebar
    logo1 = Image.open('./TU_Dublin_Logo.svg.png').resize((300, 171))
    st.sidebar.image([logo1], use_column_width=False)

    # Set page heading
    st.title('Automatic Pavement Defect Detection System')

    # Upload image file
    image_file = st.file_uploader('Upload Pavement Image', type=['jpg', 'jpeg', 'png'])

    # If image file is uploaded
    if image_file is not None:
        # Load image file
        image = Image.open(image_file)

        # Resize the image to a smaller size
        image = image.resize((512, 512))

        # Display uploaded image
        st.image(image, caption='Pavement Image', use_column_width=False)

        # If 'Detect Objects' button is clicked
        if st.button('Find Defects'):
            # Perform object detection on image
            image, df = detect_objects(image)

            # Display object detection results
            st.write('Number of defects:', len(df))
            st.write('Predicted Coordinates:')
            st.write(df)

            # Resize the image to a smaller size
            image = image.resize((512, 512))

            # Display predicted image with bounding boxes
            st.image(image, caption='Predicted image', use_column_width=False)

            # Download predicted bounding boxes as CSV file
            csv = df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="predicted_boxes.csv">Download predicted bounding boxes as CSV file</a>'
            st.markdown(href, unsafe_allow_html=True)
            
    st.markdown('---')
    st.write('Copyright © Ibrahim Syed. Technological University Dublin. 2023')
    st.write('This material is provided for research purposes only. Any other use, including commercial or non-commercial use, is strictly prohibited without the express written consent of the copyright owner.')
    st.write('For permission to use this material, please contact Ibrahim Syed at ibrahim.syed@tudublin.ie.')

# Run Stream

# Run Streamlit app
if __name__ == '__main__':
    main()
