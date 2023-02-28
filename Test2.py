import streamlit as st
import torch
import pandas as pd
from PIL import Image, ImageDraw
import os

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', './best.pt', force_reload=True)

# Define function for object detection
def detect_objects(image):
    # Perform object detection using YOLOv5
    results = model(image)

    # Define colors for each class
    color_map = {
        0: 'red',   # defect class 1
        1: 'blue',  # defect class 2
        2: 'green',  # defect class 3
        3: 'purple'
    }

    # Get bounding boxes and labels from results
    bboxes = results.xyxy[0][:, :4].cpu().numpy()
    labels = results.xyxy[0][:, -1].cpu().numpy().astype(int)

    # Draw bounding boxes on image
    img_draw = ImageDraw.Draw(image)
    for bbox, label in zip(bboxes, labels):
        # Look up color for current label
        color = color_map.get(label)
        img_draw.rectangle(bbox, outline=color, width=3)
        img_draw.text((bbox[0], bbox[1]), str(label), fill=color)

    # Convert image to RGB format
    image = image.convert('RGB')

    # Convert results to pandas dataframe
    df = pd.DataFrame(results.pandas().xyxy[0])

    # Save results to CSV file
    df.to_csv('results.csv', index=False)

    # Return image and dataframe with results
    return image, df

# Define Streamlit app
# Define Streamlit app
def main():
    # Set page title
    st.set_page_config(page_title='Automatic Pavement Defect Detection System')

    # Add logo to sidebar
    logo1 = Image.open('./TU_Dublin_Logo.svg.png').resize((300, 171))
    #logo2 = Image.open('./pms.png').resize((132, 84))
    st.sidebar.image([logo1], use_column_width=False)

    # Set page heading
    st.title('Automatic Pavement Defect Detection System')

    # Upload image file or folder
    file_type = st.radio("Select source", options=["Single Image", "Folder of Images"])
    if file_type == "Single Image":
        image_file = st.file_uploader('Upload Pavement Image', type=['jpg', 'jpeg', 'png'])
        if image_file is not None:
            images = [Image.open(image_file)]
            detect_and_save(images)
    else:
        folder_path = st.text_input('Enter folder path')
        if folder_path != "":
            folder_path = folder_path.strip()
            images = []
            for filename in os.listdir(folder_path):
                if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
                    image_path = os.path.join(folder_path, filename)
                    images.append(Image.open(image_path))
            detect_and_save(images)

def detect_and_save(images):
    # If there are no images in the folder
    if len(images) == 0:
        st.write('No images found in folder.')
        return

    # Load YOLOv5 model
    model = torch.hub.load('ultralytics/yolov5', 'custom', 'C:/Users/d19126394/Desktop/Yolo/yolov5/best.pt', force_reload=True)

    # Create output folder
    output_folder = 'output'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Define colors for each class
    color_map = {
        0: 'red',   # defect class 1
        1: 'blue',  # defect class 2
        2: 'green',  # defect class 3
        3: 'purple'
    }

    # Process each image
    for i, image in enumerate(images):
        # Resize the image to a smaller size
        image = image.resize((512, 512))

        # Perform object detection on image
        results = model(image)

        # Get bounding boxes and labels from results
        bboxes = results.xyxy[0][:, :4].cpu().numpy()
        labels = results.xyxy[0][:, -1].cpu().numpy().astype(int)

        # Draw bounding boxes on image
        img_draw = ImageDraw.Draw(image)
        for bbox, label in zip(bboxes, labels):
            # Look up color for current label
            color = color_map.get(label)
            img_draw.rectangle(bbox, outline=color, width=3)
            img_draw.text((bbox[0], bbox[1]), str(label), fill=color)

        # Convert image to RGB format
        image = image.convert('RGB')

        # Save image with bounding boxes
        output_filename = os.path.join(output_folder, f'image_{i}.jpg')
        image.save(output_filename)

        # Convert results to pandas dataframe
        df = pd.DataFrame(results.pandas().xyxy[0])

        # Save results to CSV file
        csv_filename = os.path.join(output_folder, f'results_{i}.csv')
        df.to_csv(csv_filename, index=False)

    # Display message when processing is complete
   

# Run Streamlit app
if __name__ == '__main__':
    main()