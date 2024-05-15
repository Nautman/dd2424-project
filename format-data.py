import os
import shutil

DATA_DIR = "oxford-iiit-pet"
IMAGE_DATA_PATH = os.path.join(DATA_DIR, "images")
CATS_OR_DOGS = os.path.join(DATA_DIR, "cats-or-dogs")

# Get all files in the directory of oxford-iiit-pet

cats = ['Abyssinian', 'Bengal', 'Birman', 'Bombay', 'British Shorthair', 'Egyptian Mau', 'Main Coon', 'Persian', 'Ragdoll', 'Russian Blue', 'Siamese', 'Sphynx', 'Maine Coon']
dogs = ['American Bulldog', 'American Pit Bull Terrier', 'Basset Hound', 'Beagle', 'Boxer', 'Chihuahua', 'English Cocker Spaniel', 'English Setter', 'German Shorthaired', 'Great Pyrenees', 'Havanese', 'Japanese Chin', 'Keeshond', 'Leonberger', 'Miniature Pinscher', 'Newfoundland', 'Pomeranian', 'Pug', 'Saint Bernard', 'Samyoed', 'Scottish Terrier', 'Shiba Inu', 'Staffordshire Bull Terrier', 'Wheaten Terrier', 'Yorkshire Terrier', 'samoyed']

def main():
    cats_lower = list(map(lambda x: x.lower(), cats))
    dogs_lower = list(map(lambda x: x.lower(), dogs))
    
    for img_name in os.listdir(IMAGE_DATA_PATH):
        name = img_name.split('.')[0]
        # Replace und
        name = name.split('_')[:-1]
        name = ' '.join(name)
        if name.lower() in cats_lower:
            # Copy img_name to cats folder
            shutil.copy(os.path.join(IMAGE_DATA_PATH, img_name), os.path.join(CATS_OR_DOGS, "cats", img_name))
        elif name.lower() in dogs_lower:
            # Copy img_name to dogs folder
            shutil.copy(os.path.join(IMAGE_DATA_PATH, img_name), os.path.join(CATS_OR_DOGS, "dogs", img_name))
        else:
            print(f"Unknown category for image: {img_name}")
    

            
            
            


if __name__ == "__main__":
    main()