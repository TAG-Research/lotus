import pandas as pd
import base64
from PIL import Image
from io import BytesIO
import json

# Sample function to convert an image to a base64 string
def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Sample DataFrame creation
data = {
    'id': [1, 2],
    'name': ['Image1', 'Image2'],
    'image_path': ['experiments/imageTable/image1.jpg', 'experiments/imageTable/image2.jpg']
}

df = pd.DataFrame(data)

# Add a new column with base64 encoded images
df['image_base64'] = df['image_path'].apply(image_to_base64)

# Remove the original image path if not needed
df.drop(columns=['image_path'], inplace=True)

# Write to JSONL file
with open('output.jsonl', 'w') as file:
    for _, row in df.iterrows():
        json_record = row.to_json()
        file.write(f"{json_record}\n")
