import os
import shutil

# Source and destination directories
source_dir = r'C:/Users/jerie/OneDrive/Documents/GitHub/NNGAN-Minecraft-FVDisco-StyleTransfer/dataset/trainB'
destination_dir = r'C:/Users/jerie/OneDrive/Desktop/assets/minecraft/textures'

def replace_pngs(source, dest):
    print(f'Source directory: {source}')
    print(f'Destination directory: {dest}')

    print('Source files:')
    print(os.listdir(source))

    print('Destination files:')
    for subdir, _, files in os.walk(dest):
        for file in files:
            if file.endswith('.png'):
                dest_file_path = os.path.join(subdir, file)
                source_file_path = os.path.join(source, file)
                
                print(f'Checking if {source_file_path} exists...')  # Debugging statement
                # If a corresponding file exists in the source directory, replace it
                if os.path.exists(source_file_path):
                    print(f'Replacing {dest_file_path} with {source_file_path}')
                    shutil.copy2(source_file_path, dest_file_path)
                else:
                    print(f'{source_file_path} does not exist.')

# Run the replacement function
replace_pngs(source_dir, destination_dir)

