import torch
import cv2
import numpy as np
import sys
sys.path.append('/home/student/sam2')

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

# Define model configuration and checkpoint paths
sam2_checkpoint = "/home/student/sam2/checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

# Choose the appropriate device (CUDA, MPS, or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# Build the SAM2 model
sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)

# Initialize mask generator with optimized thresholds
mask_generator = SAM2AutomaticMaskGenerator(
    model=sam2,
    pred_iou_thresh=0.2,  # Allow more overlapping detections
    stability_score_thresh=0.80,  # Capture weaker objects
    stability_score_offset=0.5,  
    mask_threshold=0.02,  # Include more fine details inside objects
    box_nms_thresh=0.1,  # Lower overlap suppression to keep more internal objects
    crop_n_layers=3,  # More cropping levels to refine smaller objects
    crop_nms_thresh=0.2,  # Reduce suppression inside cropped regions
    crop_overlap_ratio=0.3,  
    crop_n_points_downscale_factor=2,  
    min_mask_region_area=2000,  # Detect smaller objects inside others
    output_mode="binary_mask",  
    use_m2m=True,  # Mask-to-mask refinement for better accuracy
    multimask_output=True,  # Keep multiple masks per region
)

def generate_random_color():
    """Generate a random color in BGR format."""
    return tuple(np.random.randint(0, 256, 3).tolist())

# Load the input image
image = cv2.imread("color_frame.jpg")

if image is None:
    raise ValueError("Error: Unable to load the image. Check the file path.")

# Display the enhanced image before passing it to SAM2
cv2.imshow("Image", image)

print("Press 'f' to process the image with SAM2 and 'q' to quit.")
while True:
    key = cv2.waitKey(0) & 0xFF
    if key == ord('f'):
        # Generate masks for the processed image
        masks = mask_generator.generate(image)

        print(f"Generated {len(masks)} masks.")

        for i in range(len(masks)):
            mask = masks[i]['segmentation']

            if mask is not None and np.any(mask):
                mask = (mask * 255).astype(np.uint8)

                # Filter out very small dots
                if cv2.countNonZero(mask) < 500:  # Ignore small specks
                    continue

                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for contour in contours:
                    if cv2.contourArea(contour) > 500:  # Ignore small contours
                        color = generate_random_color()
                        cv2.drawContours(image, [contour], -1, color, 2)

        cv2.imshow("Segmented Image", image)

    elif key == ord('q'):
        break

cv2.destroyAllWindows()
