DepthPro Raw — Metric Depth Estimation API
Apple's DepthPro model deployed on Replicate with raw depth output instead of colorized visualization.
What this returns
Unlike other depth models on Replicate that return colorized images, this model returns:

metadata.json — focal length, field of view, depth range, image dimensions
depth_raw16.png — 16-bit grayscale PNG where each pixel = depth in millimeters
depth_colorized.png — optional turbo colormap visualization for humans

Why raw output matters
Colorized depth images look nice but are useless for code. You can't extract metric depth from an RGB heatmap.
With raw 16-bit output, you can:

Measure real-world distances (meters)
Run RANSAC plane fitting for ground detection
Calculate object heights from photos
Place 3D objects accurately on surfaces

Decoding depth
pythonimport numpy as np
from PIL import Image

depth_img = Image.open("depth_raw16.png")
depth_array = np.array(depth_img).astype(np.float32)

max_depth = 100.0  # from metadata.json → max_depth_meters
depth_meters = depth_array / 65535.0 * max_depth
API Usage
pythonimport replicate

output = replicate.run(
    "YOUR_USERNAME/depth-pro-raw",
    input={
        "image": open("photo.jpg", "rb"),
        "output_format": "both",
        "max_depth_meters": 50,
    }
)
# output[0] = metadata.json URL
# output[1] = depth_raw16.png URL
# output[2] = depth_colorized.png URL
Model

Model: apple/DepthPro-hf
Paper: Depth Pro: Sharp Monocular Metric Depth in Less Than a Second
License: Apple Sample Code License
