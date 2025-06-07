import cv2
import numpy as np
import sys

print(f"Using OpenCV version: {cv2.__version__}")

# Create a simple black image (300x300 pixels)
width, height = 300, 300
# Use np.uint8 for image data type
image = np.zeros((height, width, 3), dtype=np.uint8) 

# Add some text to make it clear it's the test image
cv2.putText(image, 'OpenCV Test', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

window_name = 'OpenCV Test Window'

try:
    # Attempt to create and display the window
    cv2.imshow(window_name, image)
    print(f"Successfully called cv2.imshow(). Press any key in the '{window_name}' window to exit.")
    
    # Wait indefinitely until a key is pressed
    key = cv2.waitKey(0) 
    print(f"Key pressed ({key}), closing window.")

except cv2.error as e:
    print(f"\nError: Failed to display OpenCV window.")
    print(f"OpenCV Error Message: {e}")
    print("\nThis usually means OpenCV was installed without GUI support (like GTK+ or Cocoa).")
    print("If you are on Linux (Debian/Ubuntu), try installing dependencies:")
    print("sudo apt-get update && sudo apt-get install -y libgtk2.0-dev pkg-config")
    print("Then, reinstall OpenCV: pip uninstall opencv-python && pip install opencv-python")
    sys.exit(1) # Exit with error code if display failed

finally:
    # Ensure windows are closed even if waitKey is interrupted
    print("Calling cv2.destroyAllWindows()...")
    cv2.destroyAllWindows()
    print("Finished.") 