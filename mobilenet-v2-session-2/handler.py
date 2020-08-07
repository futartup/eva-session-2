try:
    import unzip_requirements
except ImportError:
    pass
import os
import io
import json
import torch
import boto3
import base64
from PIL import Image
from torchvision import transforms
from torchvision.models import  MobileNetV2
from torch.hub import load_state_dict_from_url
from requests_toolbelt.multipart import decoder



S3_BUCKET = os.environ['S3_BUCKET'] if 'S3_BUCKET' in os.environ else 'models-eva'
MODEL_PATH = os.environ['MODEL_PATH'] if 'MODEL_PATH' in os.environ else 'mobilenet_v2-b0353104.pth'

print("Downloading model")
model_full_path = 'https://models-eva.s3.ap-south-1.amazonaws.com/mobilenet_v2-b0353104.pth'
s3 = boto3.client("s3")

try:
    if os.path.isfile(MODEL_PATH) != True:
        #obj = s3.get_object(Bucket=S3_BUCKET, Key=MODEL_PATH)
        #s3.Bucket(S3_BUCKET).download_file(MODEL_PATH, location)
        #print("Creating Bytestream")
        #bytestream = io.BytesIO(obj['Body'].read())
        #decodedd = decoder.b64decode(bytestream)
        model = MobileNetV2()
        state_dict = load_state_dict_from_url(model_full_path, '/tmp', progress=True)
        #print(type(model))

        #mm = mobilenet_v2(False)
        model.load_state_dict(state_dict)
        #m = torch.jit.script(mm)

        # Save to file
        #torch.jit.save(m, '/tmp/mobilenet_v2-b0353104.pth')
     
        # This line is equivalent to the previous
        #m.save("scriptmodule.pt")
        #with open('/tmp/scriptmodule.pt', 'wb') as f:
        #    f.write(bytestream.read())
        
        # Save to io.BytesIO buffer
        #buffer = io.BytesIO()
        #torch.jit.save(m, bytestream)
        #print(bytestream)
        print("Loading model")
      
        #model = torch.jit.load('/tmp/mobilenet_v2-b0353104.pth')
        #model = torch.jit.load(torch.load(bytestream))
        #model = mm.load_state_dict(torch.load(bytestream))

        print(type(model))
        print("Model Loaded...")
except Exception as e:
    print(repr(e))
    raise(e)

def transform_image(image_bytes):
    try:
        preprocess = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
        image = Image.open(io.BytesIO(image_bytes))
        return preprocess(image).unsqueeze(0)
    except Exception as e:
        print(repr(e))
        raise(e)


def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    model.eval()
    output = model(tensor).argmax().item()
    return output


def classify_image(event, context):
    try:
        content_type_header = event['headers']['content-type']
        print(event['body'])
        body = base64.b64decode(event['body'])
        print('BODY LOADED')

        picture = decoder.MultipartDecoder(body, content_type_header).parts[0]
        prediction = get_prediction(image_bytes=picture.content)
        print("The prediction is {}".format(prediction))
        #filename = picture.headers[b'Content-Disposition'].decoder().split(';')[2].split('=')[1]

        return {
            "statusCode": 200,
            "headers": {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Credentials': True
            },
            "body": json.dumps({"predicted": prediction})
        }
    except Exception as e:
        print(repr(e))
        return {
            "statusCode": 500,
            "headers": {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Credentials': True
            },
            "body": json.dumps({"predicted": prediction})
        }
