import json
import urllib.parse
import boto3
import os

ssm = boto3.client('ssm')
s3 = boto3.client('s3')
state_machine = boto3.client('stepfunctions')
sqs_client = boto3.client('sqs')

def lambda_handler(event, context):
    print('## EVENT')
    print(event)
    
    # Parsing
    S3event = event['Records'][0]['body']
    receipt_handle = event['Records'][0]['receiptHandle']
    S3dict= json.loads(S3event)
    S3json = json.dumps(S3dict)
    
    bucket = S3dict['Records'][0]['s3']['bucket']['name'] 
    key = urllib.parse.unquote_plus(S3dict['Records'][0]['s3']['object']['key'])
    image_name = os.path.basename(key)
    
    # Call state machine
    state_machine_input = {
        "bucket": bucket,
        "image_path": key,
        "image_filename": image_name
    }
    state_machine_input = json.dumps(state_machine_input)
    
    state_machine_ARN = ssm.get_parameter(Name='state_machine_ARN')['Parameter']['Value']
    state_machine_response = state_machine.start_execution(
        stateMachineArn=state_machine_ARN,
        input=state_machine_input)
    
    # Delete SQS message from queue so it doesn't get called again
    sqs_URL = ssm.get_parameter(Name='SQS_URL')['Parameter']['Value']
    delete_response = sqs_client.delete_message(
        QueueUrl=sqs_URL,
        ReceiptHandle=receipt_handle
        )
    
    print(delete_response)
    return str(state_machine_response)