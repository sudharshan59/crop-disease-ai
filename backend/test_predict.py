"""Quick smoke test for the /api/predict endpoint."""
import io
import json
import time
import urllib.request
from PIL import Image

# Create a small test image (green leaf-like)
img = Image.new("RGB", (100, 100), color=(0, 128, 0))
buf = io.BytesIO()
img.save(buf, format="JPEG")
image_data = buf.getvalue()

# Build multipart/form-data manually
boundary = "----TestBoundary123"
body = b""
body += f"--{boundary}\r\n".encode()
body += b'Content-Disposition: form-data; name="file"; filename="test.jpg"\r\n'
body += b"Content-Type: image/jpeg\r\n\r\n"
body += image_data
body += f"\r\n--{boundary}--\r\n".encode()

req = urllib.request.Request(
    "http://127.0.0.1:8000/api/predict",
    data=body,
    headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    method="POST",
)

start = time.time()
try:
    resp = urllib.request.urlopen(req, timeout=300)
    result = json.loads(resp.read())
    elapsed = time.time() - start
    print(f"SUCCESS in {elapsed:.1f}s")
    print(f"Disease: {result['disease']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print(f"Severity: {result['severity']}")
    rec = result["recommendation"]
    print(f"Rec disease: {rec['disease_name']}")
    print(f"Rec crop: {rec['crop_name']}")
    print(f"Chemical: {rec['treatment']['chemical_control'][:80]}...")
    print(f"Organic: {rec['treatment']['organic_control'][:80]}...")
except Exception as e:
    elapsed = time.time() - start
    print(f"FAILED after {elapsed:.1f}s: {e}")
