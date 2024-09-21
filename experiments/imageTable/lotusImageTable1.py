import pandas as pd
import base64
from PIL import Image
from io import BytesIO
import json
from IPython.display import display, HTML
import lotus
from lotus.utils import image_to_base64, base64_to_image, encode_images
from lotus.models import E5Model, OpenAIModel, CLIPModelRetriever

# Sample function to convert an image to a base64 string with a prefix
# def image_to_base64(image_path):
#     with open(image_path, "rb") as image_file:
#         base64_image = base64.b64encode(image_file.read()).decode('utf-8')
#         return f"data:image/jpeg;base64,{base64_image}"

# # Function to decode base64 string to an image
# def base64_to_image(base64_str):
#     image_data = base64.b64decode(base64_str.split(',')[1])  # Remove prefix before decoding
#     return Image.open(BytesIO(image_data))

# lm = OpenAIModel(api_base="http://localhost:1234/v1", api_key="none")
lm = OpenAIModel(api_key="")
# rm = E5Model()
rm = CLIPModelRetriever()

lotus.settings.configure(lm=lm, rm=rm)
data = {
    'id': [1, 2, 3],
    'name': ['Image1', 'Image2', 'Image3'],
    'image_path': ['experiments/imageTable/image1.jpg', 'experiments/imageTable/image2.jpg', 'experiments/imageTable/image3.jpg']
}

df1 = pd.DataFrame(data)
# Add a new column with base64 encoded images
#df1['image_base64'] = df1['image_path'].apply(image_to_base64)
#df1 = encode_images(df1, 'image_path', 'image_base64')
df1.load_images("image_path")

df1.sem_index("image_base64", "image_index")




data2 = {"Subject": ["Hat", "Dice", "Gopher"]}

df2 = pd.DataFrame(data2)#.sem_index("Subject", "subject_index")

#res = df1.sem_sim_join(df2, left_on="image_base64", right_on="Subject", K=1)
res = df2.sem_sim_join(df1, left_on="Subject", right_on="image_base64", K=1)
print(res)
#res["test1"] = res.groupby("Course Name")["Skill"].transform(lambda x: "\n".join(x))
