import cv2
import numpy as np
import pytesseract
import PIL.Image

# Set Tesseract path if needed (for Windows users)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def calculate_sharpness(image):
    """Calculate sharpness based on the Laplacian variance."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var


def detect_object_movement(image):
    """Detect if there is an object in the frame (using simple edge detection)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    return np.count_nonzero(edges) > 500  # Threshold to determine if the object is present


def capture_and_analyze_video():
    cap = cv2.VideoCapture(1)  # Use 0 for default camera

    if not cap.isOpened():
        print("Cannot open camera")
        return

    # Set the desired frame rate
    cap.set(cv2.CAP_PROP_FPS, 30)

    best_frame = None
    max_sharpness = 0
    frames_without_object = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Can't receive frame. Exiting ...")
            break

        sharpness = calculate_sharpness(frame)
        object_present = detect_object_movement(frame)

        # cv2.putText(frame, f"Sharpness: {sharpness:.2f}", (10, 30),
        #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Display the frame
        cv2.imshow('Video Stream', frame)

        # Check if this frame has higher sharpness than the previous best
        if object_present and sharpness > max_sharpness:
            max_sharpness = sharpness
            best_frame = frame.copy()

        # If the object is no longer detected, increment the counter
        if not object_present:
            frames_without_object += 1
        else:
            frames_without_object = 0

        # If the object has passed (e.g., no object detected for 30 frames), stop capturing
        if frames_without_object > 30:  # Adjust threshold if needed
            print("Object surpassed camera. Capturing the best frame.")
            break

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if best_frame is not None:
        print(f"Best frame captured with sharpness: {max_sharpness:.2f}")
        # Save the best frame (optional)
        cv2.imwrite('best_frame.jpg', best_frame)

    return best_frame


def preprocess_for_ocr(frame):
    """Preprocess the frame to enhance text visibility for OCR."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Remove noise with morphological operations
    # kernel = np.ones((1, 1), np.uint8)
    # gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    # gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)

    # # Contrast enhancement
    # gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=10)  # Contrast and brightness

    # # Adaptive thresholding to make the text stand out
    # thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #                                cv2.THRESH_BINARY, 11, 2)

    # # Optional: Denoise the image further
    # denoised = cv2.fastNlMeansDenoising(thresh, None, 30, 7, 21)

    return gray


def perform_ocr(frame):
    """Perform OCR with Tesseract using optimized settings."""
    custom_config = r'--oem 1 --psm 6'  # Use OEM 3 (both legacy and LSTM OCR) and PSM 6 (Assume a block of text)

    text = pytesseract.image_to_string(frame, config=custom_config)
    return text


def perform_box(img):
    my_config = r"--psm 11 --oem 3"

    height, width = img.shape
    boxes = pytesseract.image_to_boxes(img, config=my_config)
    for box in boxes.splitlines():
        box = box.split(" ")
        img = cv2.rectangle(img, (int(box[1]), height - int(box[2])), (int(box[3]), height - int(box[4])), (0, 255, 0),
                            2)
        cv2.imshow("img with box", img)
        cv2.waitKey(0)


def main():
    best_frame = capture_and_analyze_video()

    if best_frame is not None:
        # Preprocess the best frame for OCR
        processed_frame = preprocess_for_ocr(best_frame)
        # qqprocessed_frame=preprocess_with_contour_analysis(best_frame)
        # Display the processed frame
        cv2.imshow('Processed Frame for OCR', processed_frame)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Perform OCR on the processed frame
        ocr_result = perform_ocr(processed_frame)
        perform_box(processed_frame)

        # Display the OCR result
        print("\nOCR Result:\n")
        print(ocr_result)


if __name__ == "main":
    main()