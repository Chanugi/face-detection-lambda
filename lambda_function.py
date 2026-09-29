import json
import os
import uuid
import datetime
import urllib.request

import boto3
import cv2
import numpy as np
from botocore.config import Config

# ---------------------------------------------------------------------------
# Runs ONCE per container ("cold start"), then reused for later requests.
# ---------------------------------------------------------------------------
print("Loading function")

BUCKET = os.environ["BUCKET_NAME"]      # set in Lambda > Configuration > Environment variables
PREFIX = "output/"                       # folder inside the bucket for result images

s3 = boto3.client("s3", config=Config(signature_version="s3v4"))

# The Haar cascade XML ships inside the opencv package, so no "share/" folder is needed.
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_alt.xml"
)


def get_image_url(event):
    """Works for both a console test event and a browser call through API Gateway."""
    # Console test event: {"imageUrl": "https://..."}
    if event.get("imageUrl"):
        return event["imageUrl"]
    # API Gateway call: https://.../face_detection?image=https://...
    params = event.get("queryStringParameters") or {}
    return params.get("image")


def html_response(status, body):
    return {
        "statusCode": status,
        "headers": {"Content-Type": "text/html"},
        "body": body,
    }


# ---------------------------------------------------------------------------
# Runs on EVERY request.
# ---------------------------------------------------------------------------
def lambda_handler(event, context):
    print("Received event: " + json.dumps(event))

    # 1. Figure out which image to process
    image_url = get_image_url(event)
    if not image_url:
        return html_response(400, "<h3>Missing image URL. Add ?image=https://... to the address.</h3>")

    # 2. Download the image (a User-Agent header avoids being blocked by many sites)
    request = urllib.request.Request(
        image_url, headers={"User-Agent": "Mozilla/5.0 (face-detection-lambda-demo)"}
    )
    with urllib.request.urlopen(request, timeout=10) as resp:
        image_bytes = resp.read()

    # 3. Turn the raw bytes into an OpenCV image (a numpy array of pixels)
    image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return html_response(400, "<h3>Could not read that file as an image.</h3>")

    # 4. Detect faces (the algorithm works on grayscale images)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,   # how much the image is shrunk at each search pass
        minNeighbors=5,    # higher = fewer false detections, may miss some faces
        minSize=(30, 30),  # ignore anything smaller than 30x30 pixels
    )

    # 5. Draw a green box around every face found
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)

    # 6. Encode back to JPEG
    ok, output_image = cv2.imencode(".jpg", image)
    if not ok:
        raise Exception("Error encoding image")

    # 7. Save the result to S3 with a unique name
    key = (
        PREFIX
        + datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        + "-"
        + str(uuid.uuid4())
        + ".jpg"
    )
    s3.put_object(Bucket=BUCKET, Key=key, Body=output_image.tobytes(), ContentType="image/jpeg")

    # 8. Create a temporary (1 hour) link to the result, since the bucket is private
    output_url = s3.generate_presigned_url(
        "get_object", Params={"Bucket": BUCKET, "Key": key}, ExpiresIn=3600
    )

    # 9. Return a tiny web page that shows the annotated image
    return html_response(
        200,
        "<html><body>"
        f"<p>Found {len(faces)} face(s) - deployed automatically with GitHub Actions</p>"
        f"<img src='{output_url}' style='max-width:100%'/>"
        "</body></html>",
    )
