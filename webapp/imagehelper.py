from PIL import Image

def fix_orientation(filename):
    '''
    EXIF tags cause images that were taken via a smartphone to appear rotated since they are landscape images w/ EXIF orientation tags
    This function helps to find the right rotation for these images and resize at the and the image
    '''
    img = Image.open(filename)
    if hasattr(img, '_getexif'):
        exifdata = img._getexif()
        try:
            orientation = exifdata.get(274)
        except:
            # There was no EXIF Orientation Data
            orientation = 1
    else:
        orientation = 1

    if orientation is 1:    # Horizontal (normal)
        pass
    elif orientation is 2:  # Mirrored horizontal
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    elif orientation is 3:  # Rotated 180
        img = img.rotate(180)
    elif orientation is 4:  # Mirrored vertical
        img = img.rotate(180).transpose(Image.FLIP_LEFT_RIGHT)
    elif orientation is 5:  # Mirrored horizontal then rotated 90 CCW
        img = img.rotate(-90).transpose(Image.FLIP_LEFT_RIGHT)
    elif orientation is 6:  # Rotated 90 CCW
        img = img.rotate(-90)
    elif orientation is 7:  # Mirrored horizontal then rotated 90 CW
        img = img.rotate(90).transpose(Image.FLIP_LEFT_RIGHT)
    elif orientation is 8:  # Rotated 90 CW
        img = img.rotate(90)


    # Resize the image to 400x400 pixels
    size = (300, 300)
    img.thumbnail(size, Image.ANTIALIAS)

    #save the result and overwrite the originally uploaded image
    img.save(filename)