import os
import random
import pandas as pd
import json
import lotus
from lotus.models import OpenAIModel, CLIPModelRetriever

def get_random_image_per_folder(directory, translation_file, seed=None):
    # Load the translation JSON file
    with open(translation_file, 'r') as f:
        translations = json.load(f)

    data = []
    visited_folders = set()

    # Iterate over each item in the given directory
    for root, dirs, files in os.walk(directory):
        # Get the parent folder name
        folder_name = os.path.basename(root)
        
        # Create a unique key for the folder using its full path
        folder_key = os.path.normpath(root)
        
        # Skip the folder if it's already been visited
        if folder_key in visited_folders:
            continue

        # Check if there are any image files in the current directory
        image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
        
        if image_files:
            # Re-seed for each folder using a combination of the seed and folder_key
            if seed is not None:
                combined_seed = hash((seed, folder_key))
                random.seed(combined_seed)
            
            # Select a random image file
            random_image = random.choice(image_files)
            
            # Translate the folder name using the translations dictionary
            translated_name = translations.get(folder_name, folder_name)  # Default to folder_name if no translation
            
            # Construct the full path of the image
            image_path = os.path.join(root, random_image)
            
            # Append the original name, translated name, and image path to the data list
            data.append({'Name': folder_name, 'TranslatedName': translated_name, 'ImagePath': image_path})
            
            # Mark the folder as visited using its unique key
            visited_folders.add(folder_key)

    # Create a DataFrame from the collected data
    df_images = pd.DataFrame(data, columns=['Name', 'TranslatedName', 'ImagePath'])
    
    # Remove duplicate rows
    df_images.drop_duplicates(inplace=True)
    
    return df_images

def create_species_dataframe(translation_file):
    # Load the translation JSON file
    with open(translation_file, 'r') as f:
        translations = json.load(f)

    # Extract translated species names
    species_list = list(translations.values())

    # Create a DataFrame with a single column 'species'
    df_species = pd.DataFrame(species_list, columns=['species'])

    return df_species

# Example usage with a seed
directory_path = 'experiments/imageTable/animal_images/dataset'
translation_file = 'experiments/imageTable/animal_images/translation.json'
seed_value = None  # You can change this to any integer to get different results



df_images = get_random_image_per_folder(directory_path, translation_file, seed=seed_value)
df_species = create_species_dataframe(translation_file)

# Display the DataFrames
print("Images DataFrame:")
print(df_images)
print("\nSpecies DataFrame:")
print(df_species)

lm = OpenAIModel(api_key="")
rm = CLIPModelRetriever()
lotus.settings.configure(lm=lm, rm=rm)

df_images.load_images("ImagePath", "animal_image")
res = df_images.sem_filter("The {animal_image} has 4 legs.")

print(res)
res = res.drop('animal_image', axis=1)
 #res.drop('_scores', axis=1, inplace=True)
# res.rename(columns={
#     'TranslatedName': 'truth'
# }, inplace=True)

print(res)

# Generate HTML content
html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Image Results</title>
    <style>
        table {
            width: 100%;
            border-collapse: collapse;
        }
        th, td {
            padding: 10px;
            text-align: left;
            border: 1px solid #ddd;
        }
        img {
            max-width: 100px;
            height: auto;
        }
    </style>
</head>
<body>
    <h1>Image Results</h1>
    <table>
        <thead>
            <tr>
                <th>Name</th>
                <th>Image</th>
            </tr>
        </thead>
        <tbody>
"""

# Add rows to the HTML table
for index, row in res.iterrows():
    # Calculate relative path from HTML file to image
    relative_image_path = os.path.relpath(row['ImagePath'], start=os.path.dirname('experiments/imageTable/animals_results.html'))
    
    html_content += f"""
            <tr>
                <td>{row['TranslatedName']}</td>
                <td><img src="{relative_image_path}" alt="Image"></td>
            </tr>
    """

# Close the HTML tags
html_content += """
        </tbody>
    </table>
</body>
</html>
"""

# Write the HTML content to a file
output_html_file = 'experiments/imageTable/animals_results.html'
with open(output_html_file, 'w') as f:
    f.write(html_content)

print(f"\nThe result has been saved to {output_html_file}")