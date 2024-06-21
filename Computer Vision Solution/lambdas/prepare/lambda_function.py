import json
import urllib.parse
import boto3
import os
from PIL import Image

def lambda_handler(event, context):
    s3 = boto3.client('s3')

    bucket = event['bucket']
    key = event['image_path']
    image_filename = event['image_filename']

    # Download image
    image_location = '/tmp/'+image_filename
    s3.download_file(bucket, key, image_location)
    print("Downloading " + image_filename + " to " + image_location)

    # Resize image
    image = Image.open(image_location)
    print("Resizing...")
    image = resize_img(image)
    print("New size: " + str(image.size))
    image.save(image_location)
    image.close()

    # Upload resized image to S3
    s3resized_location = 'resized_images/'+image_filename
    s3.upload_file(image_location, bucket, s3resized_location)

    event['image_path'] = s3resized_location
    return event


def resize_img(image):
    image_width, image_height = image.size
    max_dimension = 1000
    if image_width > max_dimension or image_height > max_dimension:
        scale_factor = max(image_width, image_height) / max_dimension

        # use integer division to get floored value, no decimals
        new_width = int(image_width//scale_factor)
        new_height = int(image_height//scale_factor)
        image = image.resize((new_width, new_height))
    return image



