### Import the necessary libraries
import cv2
from pylab import *
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from essential_generators import DocumentGenerator
import textwrap
import os


### Create a function to read the images from the folders ###
def load_images_opencv(folder, flag, i, j):
    images = []
    filename = ['{}.jpg'.format(n) for n in range(i, j)]
    for fname in filename:
        # Read the image from the folder
        img = cv2.imread(os.path.join(folder, fname), flag)
        if img is not None:
            # Add the new image at the end of the list
            images.append(img)
    return images


### Create a function to generate rectangular for missing parts ###
def rand_rectangles(img_op, n):
    # Get the dimensions of the image
    h_im, w_im, c = img_op.shape
    img_op = cv2.cvtColor(img_op, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img_op)
    # Create the mask
    mask = Image.new("RGB", (h_im, w_im))
    # Create a copy of the image
    masked_img = img.copy()
    for i in range (n):
        # Create random width and length for the rectangles
        x = np.random.randint(10, h_im)
        y = np.random.randint(10, w_im)
        h = np.random.randint(1, round(h_im/2))
        w = np.random.randint(1, round(w_im/6))

        # Set up the new image surface for drawing
        masked_img_draw = ImageDraw.Draw(masked_img)
        mask_draw = ImageDraw.Draw(mask)

        # Draw the rectangles with white
        mask_draw.polygon([(x, y), (x + w, y), (x + w, y + h), (x, y + h)], fill='white')
        masked_img_draw.polygon([(x, y), (x + w, y), (x + w, y + h), (x, y + h)],
                         fill=(255, 255, 255))


    return masked_img, mask

### Create a function to generate text for missing parts ###
def rand_text(img_op):
    # Get the dimensions of the image
    h, w, c = img_op.shape
    img_op = cv2.cvtColor(img_op, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img_op)
    # Create the mask
    mask = Image.new("RGB", (h, w))
    # Create a copy of the image
    masked_img = img.copy()
    # Generate a random sentence
    gen = DocumentGenerator()
    text = gen.sentence()

    # Find what fonts are available
    # system_fonts = matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf')
    # print(system_fonts)
    fontpath = ["CENTURY.TTF", "FRADMIT.TTF", "MATURASC.TTF", "MAIAN.TTF", "comic.ttf", "ROCKEB.TTF",
                "LATINWD.TTF", "PAPYRUS.TTF", "cambriai.ttf", "TCCEB.TTF", "gadugib.ttf", "SHOWG.TTF",
                "mmrtextb.ttf", "LTYPE.TTF", "HARNGTON.TTF", "BOOKOSI.TTF", "ONYX.TTF", "CHILLER.TTF",
                "LFAXD.TTF", "LTYPEBO.TTF", "BELL.TTF", "JOKERMAN.TTF", "FRADMIT.TTF", "impact.ttf",
                "MAIAN.TTF", "NIAGENG.TTF", "BASKVILL.TTF", "MOD20.TTF", "ELEPHNTI.TTF", "RAGE.TTF",
                "VIVALDII.TTF", "ARIALUNI.TTF", "OCRAEXT.TTF", "ITCEDSCR.TTF", "webdings.ttf", "ERASBD.TTF"]
    font = ImageFont.truetype(fontpath[np.random.randint(1, len(fontpath))], size=np.random.randint(15, 30))
    masked_img_draw = ImageDraw.Draw(masked_img)
    mask_draw = ImageDraw.Draw(mask)

    text = textwrap.fill(text=text, width=30)
    mask_draw.text(xy=(img.size[0] / 2, img.size[1] / 2), text=text, font=font, fill='white', anchor='mm')
    masked_img_draw.text(xy=(img.size[0] / 2, img.size[1] / 2), text=text, font=font, fill='white', anchor='mm')

    return masked_img, mask

### Create a function to generate lines for missing parts ###
def rand_lines(img):
    # Create the mask
    mask = np.full((img.shape[0],img.shape[1], 3), 0, np.uint8) # Black background
    # Create a copy of the image
    masked_img = img.copy()
    for i in range(np.random.randint(7, 20)):
        # Get random x locations to start line
        x1, x2 = np.random.randint(1, img.shape[0]), np.random.randint(1, img.shape[0])
        # Get random y locations to start line
        y1, y2 = np.random.randint(1, img.shape[1]), np.random.randint(1, img.shape[1])
        # Get random thickness of the line
        thickness = np.random.randint(5, 12)
        # Draw a white line on the black mask and on the image copy
        cv2.line(mask, (x1,y1), (x2,y2), (255, 255, 255), thickness)
        cv2.line(masked_img, (x1, y1), (x2, y2), (255, 255, 255), thickness)

    return masked_img, mask


def main():
    # Read the images
    places_lines = load_images_opencv(r'C:\Licenta\images\original_images', 1, 0, 6000)
    places_text = load_images_opencv(r'C:\Licenta\images\original_images', 1, 6000, 9000)
    places_rectangles = load_images_opencv(r'C:\Licenta\images\original_images', 1, 9000, 14000)
    faces_lines = load_images_opencv(r'C:\Licenta\images\original_images', 1, 14000, 16000)
    faces_text = load_images_opencv(r'C:\Licenta\images\original_images', 1, 16000, 17000)
    faces_rectangles = load_images_opencv(r'C:\Licenta\images\original_images', 1, 17000, 19000)


    i = 0
    for j in range(len(places_lines)):
        masked_img, mask = rand_lines(places_lines[j])
        outname_img = r'C:\Licenta\images\damaged_images\%d_masked.jpg' % (i)
        outname_mask = r'C:\Licenta\images\mask_images\%d_mask.jpg' % (i)
        cv2.imwrite(outname_img, masked_img)
        cv2.imwrite(outname_mask, mask)
        i = i + 1

    for j in range(places_text):
        masked_img, mask = rand_text(places_text[j])
        outname_img = r'C:\Licenta\images\damaged_images\%d_masked.jpg' % (i)
        outname_mask = r'C:\Licenta\images\mask_images\%d_mask.jpg' % (i)
        masked_img.save(outname_img)
        mask.save(outname_mask)
        i = i + 1

    for j in range(len(places_rectangles)):
        masked_img, mask = rand_rectangles(places_rectangles[j], np.random.randint(10, 21))
        outname_img = r'C:\Licenta\images\damaged_images\%d_masked.jpg' % (i)
        outname_mask = r'C:\Licenta\images\mask_images\%d_mask.jpg' % (i)
        masked_img.save(outname_img)
        mask.save(outname_mask)
        i = i + 1

    for j in range(len(faces_lines)):
        masked_img, mask = rand_lines(faces_lines[j])
        outname_img = r'C:\Licenta\images\damaged_images\%d_masked.jpg' % (i)
        outname_mask = r'C:\Licenta\images\mask_images\%d_mask.jpg' % (i)
        cv2.imwrite(outname_img, masked_img)
        cv2.imwrite(outname_mask, mask)
        i = i + 1

    for j in range(len(faces_text)):
        masked_img, mask = rand_text(faces_text[j])
        outname_img = r'C:\Licenta\images\damaged_images\%d_masked.jpg' % (i)
        outname_mask = r'C:\Licenta\images\mask_images\%d_mask.jpg' % (i)
        masked_img.save(outname_img)
        mask.save(outname_mask)
        i = i + 1

    for j in range(len(faces_rectangles)):
        masked_img, mask = rand_rectangles(faces_rectangles[j], np.random.randint(10, 21))
        outname_img = r'C:\Licenta\images\damaged_images\%d_masked.jpg' % (i)
        outname_mask = r'C:\Licenta\images\mask_images\%d_mask.jpg' % (i)
        masked_img.save(outname_img)
        mask.save(outname_mask)
        i = i + 1

    return

if __name__ == '__main__':
    main()
