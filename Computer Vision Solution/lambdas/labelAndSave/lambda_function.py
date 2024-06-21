import json
import os
import boto3
from boto3.dynamodb.types import TypeDeserializer
from PIL import Image, ImageDraw, ExifTags, ImageColor, ImageFont

s3 = boto3.client('s3')
dynamo = boto3.client('dynamodb')

#colors for confidence intervals: 100% to 90% is color number 10 (bright green)
COLORS = {1: (255, 0, 0), 2: (227, 29, 1), 3: (199, 57, 2), 4: (170, 85, 3), 5: (142, 114, 4), 6: (114, 142, 5), 7: (85, 170, 6), 8: (57, 199, 7), 9: (29, 227, 8), 10:(0,255,9)}
font = ImageFont.truetype("/opt/OpenSans-Regular.ttf", 20)

def lambda_handler(event, context):
    bucket = event[0]['bucket']
    table_name = event[0]['table_name']
    image_filename = event[0]['image_filename']
    image_path = event[0]['image_path']
    image_name = image_name = os.path.splitext(image_filename)[0]
    
    # Download image and prepare for labeling
    image_location = s3_img_download(bucket, image_path, image_filename)
    image = Image.open(image_location)
    draw = ImageDraw.Draw(image)
    width, height = image.size
    
    # Get image results from DynamoDB
    image_results = dynamo.get_item(TableName=table_name, Key = {'image_name': {'S': image_name}})
    image_results = dynamo_obj_to_python_obj(image_results['Item'])

    for endpoint in event:
        endpoint_name = endpoint['db_key']
        threshold = float(endpoint['threshold'])
        endpoint_results = image_results[endpoint_name]
        print(endpoint_name)
        print(endpoint_results)
        
        for result in endpoint_results:
            conf_score = float(result['confScore'])
            
            if conf_score >= threshold:
                obj_label = result['objLabel']
                boundBoxLTRB = result['boundBoxLTRB']
                
                left = float(boundBoxLTRB[0])
                top = float(boundBoxLTRB[1])
                right = float(boundBoxLTRB[2])
                bottom = float(boundBoxLTRB[3])
                
                x1 = left * width
                y1 = top * height
                x2 = right * width
                y2 = bottom * height
                
                print_res = (obj_label + ": " + str(round(conf_score, 2)))
                color_choice = int(round(float(conf_score)*100, 2) / 10) + 1
                                
                draw.text((x1,y1),print_res,COLORS[color_choice], font=font)
                for l in range(3):
                    draw.rectangle((x1-l,y1-l,x2+l,y2+l),outline=COLORS[color_choice])

    image.save(image_location,format='JPEG')
    s3resized_location = 'labeled_images/' + image_filename
    s3.upload_file(image_location, bucket, s3resized_location)
    image.close()

    return {
        'statusCode': 200,
        'body': json.dumps('Image processing finished!')
    }

def s3_img_download(bucket, image_path, image_filename):
    # Download resized image from S3 and convert to tensor
    image_location = '/tmp/' + image_filename
    s3.download_file(bucket, image_path, image_location)
    print("Downloading " + image_filename + " to " + image_location)
    return image_location
    
def label_image(results):
    pass

def dynamo_obj_to_python_obj(dynamo_obj):
    deserializer = TypeDeserializer()
    return {
        k: deserializer.deserialize(v) 
        for k, v in dynamo_obj.items()
    }  