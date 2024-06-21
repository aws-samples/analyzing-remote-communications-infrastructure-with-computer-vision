import json
import boto3

ssm = boto3.client('ssm')

def lambda_handler(event, context):
    table_name = ssm.get_parameter(Name='table_name')['Parameter']['Value']
    print("Table name:")
    print(table_name)
    
    endpoint_config = ssm.get_parameter(Name='endpoint_config')['Parameter']['Value']
    endpoint_config = json.loads(endpoint_config)
    
    endpoints = endpoint_config['endpoints_config']
    print("Endpoints:")
    print(endpoints)
    
    for endpoint in endpoints:
        endpoint["image_path"] = event["image_path"]
        endpoint["image_filename"] = event["image_filename"]
        endpoint["bucket"] = event["bucket"]
        endpoint["table_name"] = table_name
    
    event["endpoints"] = endpoints
    event["table_name"] = table_name
    return event
