import os
import shutil
import pandas as pd


def classify_bird(bird_species: str) -> int:
    WATERBIRDS = [
        'Albatross',  # Seabirds
        'Auklet',
        'Cormorant',
        'Frigatebird',
        'Fulmar',
        'Gull',
        'Jaeger',
        'Kittiwake',
        'Pelican',
        'Puffin',
        'Tern',
        'Gadwall',  # Waterfowl
        'Grebe',
        'Mallard',
        'Merganser',
        'Guillemot',
        'Loon'
    ]
    for species in WATERBIRDS:
        if species.lower() in bird_species.split("_"):
            return 0
    else:
        return 1


def main():
    copy_files = True
    images_directory = "data/CUB_200_2011/images/"
    images = pd.read_csv("data/CUB_200_2011/images.txt",
                         sep=" ",
                         header=None,
                         names=['img_id', 'img_filename'],
                         index_col='img_id')
    images["species"] = images["img_filename"].map(lambda filename: filename.split("/")[0].split(".")[1].lower())
    images["birdtype"] = images["species"].map(lambda species: classify_bird(species))
    train_test = pd.read_csv("data/CUB_200_2011/train_test_split.txt",
                             sep=" ",
                             header=None,
                             names=['img_id', 'split'],
                             index_col='img_id',
                             )
    dataset = images.join(other=train_test, on="img_id")
    dataset.to_csv("data/waterbird/annotation.csv")

    if copy_files:
        for index, datapoint in dataset.iterrows():
            if datapoint["birdtype"] == 0:
                birdtype = "waterbird"
            else:
                birdtype = "landbird"
            if datapoint["split"] == 1:
                shutil.copy(images_directory + datapoint["img_filename"],
                            "data/waterbird/train/"
                            + birdtype + "/"
                            + datapoint["img_filename"].split("/")[1])
            else:
                shutil.copy(images_directory + datapoint["img_filename"],
                            "data/waterbird/test/"
                            + birdtype + "/"
                            + datapoint["img_filename"].split("/")[1])


if __name__ == '__main__':
    main()
