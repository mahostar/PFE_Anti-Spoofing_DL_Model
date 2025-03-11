"""
Script to split the dataset into train, validation and test sets.
"""

import os
import random
import shutil
from itertools import islice

outputFolderPath = "../Dataset/SplitData"
inputFolderPath = "../Dataset/all"
splitRatio = {"train": 0.7, "val": 0.2, "test": 0.1}
classes = ["fake", "real"]

def create_directories():
    """Create necessary directories for split data."""
    try:
        shutil.rmtree(outputFolderPath)
    except OSError:
        os.makedirs(outputFolderPath, exist_ok=True)

    # --------  Directories to Create -----------
    os.makedirs(f"{outputFolderPath}/train/images", exist_ok=True)
    os.makedirs(f"{outputFolderPath}/train/labels", exist_ok=True)
    os.makedirs(f"{outputFolderPath}/val/images", exist_ok=True)
    os.makedirs(f"{outputFolderPath}/val/labels", exist_ok=True)
    os.makedirs(f"{outputFolderPath}/test/images", exist_ok=True)
    os.makedirs(f"{outputFolderPath}/test/labels", exist_ok=True)

def create_data_yaml():
    """Create the data.yaml file for YOLOv8."""
    dataYaml = f'path: ../Data\n\
train: ../train/images\n\
val: ../val/images\n\
test: ../test/images\n\
\n\
nc: {len(classes)}\n\
names: {classes}'

    with open(f"{outputFolderPath}/data.yaml", 'w') as f:
        f.write(dataYaml)
    print("Data.yaml file Created...")

def main():
    create_directories()

    # --------  Get the Names  -----------
    listNames = os.listdir(inputFolderPath)
    uniqueNames = []
    for name in listNames:
        uniqueNames.append(name.split('.')[0])
    uniqueNames = list(set(uniqueNames))

    # --------  Shuffle -----------
    random.shuffle(uniqueNames)

    # --------  Find the number of images for each folder -----------
    lenData = len(uniqueNames)
    lenTrain = int(lenData * splitRatio['train'])
    lenVal = int(lenData * splitRatio['val'])
    lenTest = int(lenData * splitRatio['test'])

    # --------  Put remaining images in Training -----------
    if lenData != lenTrain + lenTest + lenVal:
        remaining = lenData - (lenTrain + lenTest + lenVal)
        lenTrain += remaining

    # --------  Split the list -----------
    lengthToSplit = [lenTrain, lenVal, lenTest]
    Input = iter(uniqueNames)
    Output = [list(islice(Input, elem)) for elem in lengthToSplit]
    print(f'Total Images:{lenData} \nSplit: {len(Output[0])} {len(Output[1])} {len(Output[2])}')

    # --------  Copy the files  -----------
    sequence = ['train', 'val', 'test']
    for i, out in enumerate(Output):
        for fileName in out:
            shutil.copy(f'{inputFolderPath}/{fileName}.jpg', 
                       f'{outputFolderPath}/{sequence[i]}/images/{fileName}.jpg')
            shutil.copy(f'{inputFolderPath}/{fileName}.txt', 
                       f'{outputFolderPath}/{sequence[i]}/labels/{fileName}.txt')

    create_data_yaml()
    print("Split Process Completed...")

if __name__ == "__main__":
    main() 