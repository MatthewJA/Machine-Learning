#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Load and manipulate the MNIST database of handwritten digits.

The default paths assume that the database is stored in the current directory
as raw binary files mnist_images and mnist_labels.

Information on the MNIST database is found at http://yann.lecun.com/exdb/mnist/

Matthew Alger, Buck Shlegeris
2014 -- 2015
"""

import struct
import pickle
import tkinter

import numpy
import theano

def load_training_labels(db_location="mnist_labels", matrix=False,
    fmt="numpy", validation=False):
    """
    Return a list of labels in database order.

    Labels are integers, or a matrix of 1's and 0's if matrix=True.
    """

    with open(db_location, "rb") as f:
        # Check magic number.
        assert struct.unpack(">I", f.read(4))[0] == 2049

        # Get number of labels.
        label_count = struct.unpack(">I", f.read(4))[0]

        # Read that many labels.
        labels = []
        for i in range(label_count):
            label = struct.unpack(">B", f.read(1))[0]
            labels.append(label)

        nparray = numpy.array(labels)
        if matrix:
            mat = numpy.zeros((label_count, 10))
            mat[numpy.arange(nparray.shape[0]), nparray] = 1
            nparray = mat
            print(mat[:,:4])

        if validation:
            nparray = nparray[50000:,]
        else:
            nparray = nparray[:50000,]

        if fmt == "numpy":
            return nparray
        elif fmt == "theano" and not matrix:
            return theano.tensor.cast(
                theano.shared(
                    numpy.asarray(nparray, dtype=theano.config.floatX),
                    borrow=True
                ),
                "int32"
            )
        elif fmt == "theano" and matrix:
            return theano.shared(
                numpy.asarray(nparray, dtype=theano.config.floatX),
                borrow=True
            )
        else:
            raise ValueError("Invalid format: {}".format(fmt))

def load_training_images(db_location="mnist_images", fmt="numpy",
    validation=False, div=1):
    """
    Return a list of images in database order.

    Images are a 784-dimensional tuple of values in [0, 255].

    If validation is False, then the first 50000 images will be loaded.
    Otherwise, the last 10000 will be loaded.
    """

    # Do we have a pickle?
    try:
        with open(db_location + ".pkl", "rb") as f:
            print("loading from pickle")
            nparray = pickle.load(f)
    except IOError:
        with open(db_location, "rb") as f:
            # Check magic number.
            assert struct.unpack(">I", f.read(4))[0] == 2051

            # Get number of images.
            image_count = struct.unpack(">I", f.read(4))[0]

            # Get number of rows.
            row_count = struct.unpack(">I", f.read(4))[0]

            # Get number of columns.
            column_count = struct.unpack(">I", f.read(4))[0]

            # Read pixels.

            # We will read batches of images to minimise file operations.
            total_images_to_read = image_count
            batch_size = 10000 # Totally arbitrary.
            images = []

            while total_images_to_read > 0:
                if total_images_to_read > batch_size:
                    images_to_read = batch_size
                else:
                    images_to_read = total_images_to_read

                data = f.read(images_to_read*row_count*column_count)
                for im in range(images_to_read):
                    image = []
                    for px in range(row_count*column_count):
                        pixel = struct.unpack(">B",
                            bytes([data[im*row_count*column_count+px]]))[0]
                        image.append(pixel)
                    images.append(tuple(image))

                total_images_to_read -= images_to_read

            assert len(images) == image_count

            nparray = numpy.array(images)/div

            with open(db_location + ".pkl", "wb") as g:
                pickle.dump(nparray, g, -1)

    if validation:
        nparray = nparray[50000:,]
    else:
        nparray = nparray[:50000,]

    if fmt == "numpy":
        return nparray
    elif fmt == "theano":
        return theano.shared(
            numpy.asarray(nparray, dtype=theano.config.floatX),
            borrow=True
        )
    else:
        raise ValueError("Invalid format: {}".format(fmt))

def load_test_labels(db_location="mnist_test_labels", fmt="numpy"):
    return load_training_labels(db_location, fmt)

def load_test_images(db_location="mnist_test_images", fmt="numpy"):
    return load_training_images(db_location, fmt)

def view_image(image, width, height):
    """
    Open a Tkinter window to view an image from the MNIST dataset.
    """
    root = tkinter.Tk()
    root.minsize(width, height)
    root.geometry("{}x{}".format(width*2, height*2))
    root.bind("<Button>", lambda e: e.widget.quit())
    im = tkinter.PhotoImage(width=width, height=height)
    im.put(
        " ".join(
            ("{" + " ".join(
                "#{:02x}{:02x}{:02x}".format(
                    image[i+width*j], image[i+width*j],image[i+width*j])
                    for i in range(width)
                ) + "}"
            ) for j in range(height)))
    w = tkinter.Label(root, image=im)
    w.pack(fill="both", expand=True)
    root.mainloop()