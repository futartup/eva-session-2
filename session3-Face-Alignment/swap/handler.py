try:
    import unzip_requirements
except ImportError:
    pass
	
import cv2
import dlib
import numpy as np
from faceBlendCommon import *
import math
import boto3
import os
import io
import json
import base64
from requests_toolbelt.multipart import decoder
print('Import END...')


# s3 = boto3.client('s3')
s3 = boto3.resource('s3')
s3_client = boto3.client('s3')

datname = r'shape_predictor_68_face_landmarks.dat'



def get_prediction(image_bytes1,image_bytes2):
    faceDetector=dlib.get_frontal_face_detector()
    landmarkDetector=dlib.shape_predictor(datname)
    print('facedetector and landmarkdetector object done')
    img1 = np.frombuffer(image_bytes1, dtype=np.int8)
    img2 = np.frombuffer(image_bytes2, dtype=np.int8)

    print('img converted to array done')
    im1 = cv2.imdecode(img1, cv2.IMREAD_COLOR)
    im2 = cv2.imdecode(img2, cv2.IMREAD_COLOR)

    points1 = getLandmarks(faceDetector,landmarkDetector,im1)
    points1=np.array(points1)
    # im1 = np.float32(im1)/255

    points2 = getLandmarks(faceDetector,landmarkDetector,im2)
    points2=np.array(points2)
    # im2 = np.float32(im2)/255

    h=600;w=600;

    # Find convex hull
    hullIndex = cv2.convexHull(np.array(points2), returnPoints=False)
    # Create convex hull lists
    hull1 = []
    hull2 = []
    for i in range(0, len(hullIndex)):
        hull1.append(points1[hullIndex[i][0]])
        hull2.append(points2[hullIndex[i][0]])

    # Calculate Mask for Seamless cloning
    hull8U = []
    for i in range(0, len(hull2)):
        hull8U.append((hull2[i][0], hull2[i][1]))

    mask = np.zeros(im2.shape, dtype=im2.dtype) 
    cv2.fillConvexPoly(mask, np.int32(hull8U), (255, 255, 255))
    # Find Centroid
    m = cv2.moments(mask[:,:,1])
    center = (int(m['m10']/m['m00']), int(m['m01']/m['m00']))

    # Find Delaunay traingulation for convex hull points
    sizeImg2 = im2.shape    
    rect = (0, 0, sizeImg2[1], sizeImg2[0])

    dt = calculateDelaunayTriangles(rect, hull2)

    tris1 = []
    tris2 = []
    for i in range(0, len(dt)):
        tri1 = []
        tri2 = []
        for j in range(0, 3):
            tri1.append(hull1[dt[i][j]])
            tri2.append(hull2[dt[i][j]])

        tris1.append(tri1)
        tris2.append(tri2)

    # Simple Alpha Blending
    # Apply affine transformation to Delaunay triangles
    img1Warped = np.copy(im2)
    for i in range(0, len(tris1)):
        warpTriangle(im1, img1Warped, tris1[i], tris2[i])


    # Clone seamlessly.
    output = cv2.seamlessClone(np.uint8(img1Warped), im2, mask, center, cv2.NORMAL_CLONE)

    # print(output.shape)
    data_serial = cv2.imencode('.png', output)[1].tostring()
    # print(data_serial)
    s3.Object('session3-face-alignment-face-swap', 'face-swap.png').put(Body=data_serial,ContentType='image/PNG', ACL='public-read')
    url = "https://{}.s3.amazonaws.com/{}".format('session3-face-alignment-face-swap', 'face-swap.png')

    # print(url)
    return url

def align_image(event, context):
    try:
        content_type_header = event['headers']['content-type']
        #print(event['body'])
        body = base64.b64decode(event["body"])
        print('BODY LOADED')
        picture1 = decoder.MultipartDecoder(body, content_type_header).parts[0]
        picture2 = decoder.MultipartDecoder(body, content_type_header).parts[1]
        print("Picture loaded")	
        prediction = get_prediction(image_bytes1=picture1.content, image_bytes2=picture2.content)
        print("Prediction done")
        print(prediction)
        
        filename = picture1.headers[b'Content-Disposition'].decode().split(';')[1].split('=')[1]
        if len(filename)<4:
            filename = picture1.headers[b'Content-Disposition'].decode().split(';')[2].split('=')[1]

        return {
            "statusCode": 200,
            "headers": {
                'Content-Type':'application/json',
                'Access-Control-Allow-Origin': '*',
                "Access-Control-Allow-Credentials": True
            },
            "body": json.dumps({'file': filename.replace('"', ''), 'predicted class id':prediction})
        }

    except Exception as e:
        print(repr(e))
        return {
            "statusCode": 500,
            "headers": {
                'Content-Type':'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Credentials':True
            },
            "body": json.dumps({"error": repr(e)})
        }
