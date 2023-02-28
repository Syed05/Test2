import streamlit as st
import torch
import pandas as pd
from PIL import Image, ImageDraw

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', './best.pt', force_reload=True)

# Define function for object detection
def detect_objects(image):
    # Perform object detection using YOLOv5
    results = model(image)
    
    # Get bounding boxes and labels from results
    bboxes = results.xyxy[0][:, :4].cpu().numpy()
    labels = results.xyxy[0][:, -1].cpu().numpy().astype(int)
    
    # Draw bounding boxes on image
    img_draw = ImageDraw.Draw(image)
    for bbox, label in zip(bboxes, labels):
        img_draw.rectangle(bbox, outline='red', width=3)
        img_draw.text((bbox[0], bbox[1]), str(label), fill='red')
    
    # Convert image to RGB format
    image = image.convert('RGB')
    
    # Convert results to pandas dataframe
    df = pd.DataFrame(results.pandas().xyxy[0])
    
    # Save results to CSV file
    df.to_csv('results.csv', index=False)
    
    # Return image and dataframe with results
    return image, df

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

    # Upload image file
    image_file = st.file_uploader('Upload Pavement Image', type=['jpg', 'jpeg', 'png'])

    # If image file is uploaded
    if image_file is not None:
        # Load image file
        image = Image.open(image_file)

        # Resize the image to a smaller size
        image = image.resize((512, 512)) # change the size as per your requirement

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
            image = image.resize((512, 512)) # change the size as per your requirement

            # Display predicted image with bounding boxes
            st.image(image, caption='Predicted image', use_column_width=False)


# Run Streamlit app
if __name__ == '__main__':
    main()
