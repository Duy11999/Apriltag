import cv2
import numpy as np


def find_contour_box(image_path, target_ratio):
    # Load the image

    image = cv2.imread(image_path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    l_b = np.array([0, 11, 0])
    u_b = np.array([255, 255, 255])

    # Apply thresholding to obtain a binary image
    mask = cv2.inRange(hsv, l_b, u_b)

    # Find contours in the binary image
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Iterate through the contours
    for contour in contours:
        # Get the bounding rectangle of the contour
        x, y, w, h = cv2.boundingRect(contour)

        # Calculate the aspect ratio of the bounding rectangle
        aspect_ratio = float(w) / h

        # Check if the aspect ratio matches the target ratio
        if abs(aspect_ratio - target_ratio) < 0.1:
            # Draw a rectangle around the contour
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Display the image with the contour box
            cv2.imshow("Contour Box", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            # Return the coordinates of the bounding rectangle
            return (x, y, x + w, y + h)

    # If no contour with the target ratio is found, return None
    return None


# Example usage
image_path = "../Images/Slab1.jpg"  # Replace with your image path
target_ratio = 0.5  # Replace with your desired aspect ratio

box_coords = find_contour_box(image_path, target_ratio)
if box_coords:
    print("Found contour box with the specified ratio:", box_coords)
else:
    print("No contour box with the specified ratio was found.")
