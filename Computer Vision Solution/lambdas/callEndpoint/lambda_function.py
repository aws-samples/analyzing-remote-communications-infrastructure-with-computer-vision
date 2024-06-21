import boto3
from boto3.dynamodb.types import TypeSerializer
import os
import json
import cv2

s3 = boto3.client('s3')
dynamo_client = boto3.client('dynamodb')
sagemaker_client = boto3.client('sagemaker-runtime')

def lambda_handler(event, context):
    # Parse event
    bucket = event['bucket']
    key = event['image_path']
    image_filename = event['image_filename']
    labels = event['labels']
    threshold = event['threshold']
    table_name = event['table_name']

    # Download resized image from S3 and convert to tensor
    image_location = '/tmp/'+image_filename
    s3.download_file(bucket, key, image_location)
    print("Downloading " + image_filename + " to " + image_location)
    
    # Convert resized image to tensor, then to python list
    print("Converting image to tensor...")
    img_tensor = image_file_to_tensor(image_location)

    input = {
        'instances': [img_tensor.tolist()]
    }

    # Grab appropriate endpoint given the endpoint config passed in the event
    endpoint_name = event['ep_name']
    print("Accessing Endpoint:")
    print(endpoint_name)
    content_type = 'application/json'
    input = json.dumps(input)
    # body = json.dumps({'instances': img_tensor})
    response = sagemaker_client.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType=content_type,
        Body=input
        )

    detections = response['Body'].read().decode('utf-8')
    detections = json.loads(detections)
    detections = detections['predictions'][0]


    print("Parsing response:")

    # Parse the labels from config. The labels must match the mapping defined during training/labeling in SageMaker notebook and GroundTruth
    category_index = {}
    for index in range(len(labels)):
        category_index[index + 1] = {'id': index + 1, 'name': labels[index]}
    print(category_index)

    # parse results
    results = parse_results(detections, category_index, threshold)
    results = python_obj_to_dynamo_obj(results)
    print(results)
    
    image_name = os.path.splitext(image_filename)[0]
    db_key = event['db_key']
    update_expression = f"set {db_key} = :g"
    dynamo_client.update_item(TableName = table_name, Key = {'image_name':{'S': image_name}}, UpdateExpression=update_expression, ExpressionAttributeValues={':g': results['results']})
    
    return event
    
def image_file_to_tensor(path):
    cv_img = cv2.imread(path, 1).astype('uint8')
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    return cv_img

def parse_results(detections, category_index, threshold):
    detection_boxes = detections['detection_boxes']
    detection_classes = [int(x) for x in detections['detection_classes']]
    detection_scores = detections['detection_scores']
    
    resultsDict = {'results': []}
    for index in range(len(detection_scores)-1):
        objName = category_index[detection_classes[index]]['name']
        print(objName)
        objConf = detection_scores[index]
        top = detection_boxes[index][0]
        left = detection_boxes[index][1]
        bottom = detection_boxes[index][2] 
        right = detection_boxes[index][3]
        
        tempDict = {}
        tempDict['objLabel'] = objName
        tempDict['confScore'] = str(objConf)
        tempDict['boundBoxLTRB'] = [str(left), str(top), str(right), str(bottom)]
        resultsDict['results'].append(tempDict)
    
    return resultsDict
    
def python_obj_to_dynamo_obj(python_obj):
    serializer = TypeSerializer()
    return {
        k: serializer.serialize(v)
        for k, v in python_obj.items()
    }