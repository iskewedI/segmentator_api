from PIL import Image
import os
import argparse

def rescale_images(directory, size, out):
    try:
        os.makedirs(out)
    except:
        print("Skipping creation of dir as it already exists")

    for img in os.listdir(directory):
        if(not ".jpg" in img and not ".png" in img):
            continue

        im = Image.open(os.path.join(directory, img))

        im_resized = im.resize(size, Image.ANTIALIAS)

        im_resized.save(os.path.join(out, img))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Rescale images")
    parser.add_argument('-d', '--directory', type=str, required=True, help='Directory containing the images')
    parser.add_argument('-o', '--out', type=str, required=True, help='Output directory')
    parser.add_argument('-s', '--size', type=int, nargs=2, required=True, metavar=('width', 'height'), help='Image size')
    args = parser.parse_args()
    rescale_images(args.directory, args.size, args.out)
