import pandas as pd
import base64
from PIL import Image
from io import BytesIO
import json

# Function to decode base64 string to an image
def base64_to_image(base64_str):
    image_data = base64.b64decode(base64_str)
    return Image.open(BytesIO(image_data))

# Load JSONL file into a DataFrame
records = []
with open('output.jsonl', 'r') as file:
    for line in file:
        record = json.loads(line.strip())
        records.append(record)

df_loaded = pd.DataFrame(records)

# Display DataFrame with images
def render_image(base64_str):
    img = base64_to_image(base64_str)
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    img_b64 = base64.b64encode(buffer.getvalue()).decode()
    return f'<img src="data:image/png;base64,{img_b64}" width="100"/>'

# Apply the rendering function to the 'image_base64' column
df_loaded['image'] = df_loaded['image_base64'].apply(render_image)

# Drop the base64 column if not needed for display
df_loaded.drop(columns=['image_base64'], inplace=True)

# Convert the DataFrame to HTML for display
html = df_loaded.to_html(escape=False, formatters={'image': lambda x: x})

# Write the HTML to a file
with open('output.html', 'w') as f:
    f.write(html)

print("HTML table with images has been written to 'output.html'. Open this file in a web browser to view.")
