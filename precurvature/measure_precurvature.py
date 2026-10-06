import sys
import cv2
import numpy as np
from scipy.optimize import least_squares

MARKER_SIZE_MM = 60.0  # Physical edge length of the ArUco marker in mm
# Common dictionaries to test automatically
ARUCO_DICTS = [
    cv2.aruco.DICT_4X4_50,
    cv2.aruco.DICT_4X4_100,
    cv2.aruco.DICT_5X5_50,
    cv2.aruco.DICT_5X5_100,
    cv2.aruco.DICT_6X6_50,
    cv2.aruco.DICT_6X6_100,
    cv2.aruco.DICT_ARUCO_ORIGINAL,
]


def detect_aruco_scale(image, marker_length_mm=MARKER_SIZE_MM):
    """Detects an ArUco marker and calculates millimeters per pixel."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    detected_corners = None
    detected_id = None

    for dict_id in ARUCO_DICTS:
        dictionary = cv2.aruco.getPredefinedDictionary(dict_id)
        parameters = cv2.aruco.DetectorParameters()

        # Handle OpenCV version compatibility
        if hasattr(cv2.aruco, "ArucoDetector"):
            detector = cv2.aruco.ArucoDetector(dictionary, parameters)
            corners, ids, _ = detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=parameters)

        if ids is not None and len(ids) > 0:
            # Safely reshape corners to a (4, 2) array
            detected_corners = corners[0].reshape(4, 2)
            # Flatten the ids array to safely extract the first item
            detected_id = int(np.ravel(ids)[0])
            break

    if detected_corners is None:
        raise RuntimeError("No ArUco marker detected. Ensure good lighting and proper framing.")

    # Calculate average edge length in pixels
    side_lengths = [
        np.linalg.norm(detected_corners[0] - detected_corners[1]),
        np.linalg.norm(detected_corners[1] - detected_corners[2]),
        np.linalg.norm(detected_corners[2] - detected_corners[3]),
        np.linalg.norm(detected_corners[3] - detected_corners[0]),
    ]
    mean_side_px = np.mean(side_lengths)
    mm_per_pixel = marker_length_mm / mean_side_px

    return mm_per_pixel, detected_corners, detected_id


def fit_circle(points):
    """Fits a circle to 2D points using nonlinear least squares."""
    pts = np.asarray(points, dtype=np.float64)
    x, y = pts[:, 0], pts[:, 1]

    # Initial guess: center at centroid, radius as mean distance to centroid
    x_m = np.mean(x)
    y_m = np.mean(y)
    r_initial = np.mean(np.sqrt((x - x_m) ** 2 + (y - y_m) ** 2))

    def residuals(params):
        xc, yc, r = params
        return np.sqrt((x - xc) ** 2 + (y - yc) ** 2) - r

    res = least_squares(residuals, [x_m, y_m, r_initial])
    xc, yc, r = res.x
    return xc, yc, abs(r)


def run_precurvature_measurement(image_path):
    orig_img = cv2.imread(image_path)
    if orig_img is None:
        print(f"Error: Unable to load image at '{image_path}'.")
        return

    try:
        mm_per_px, marker_corners, marker_id = detect_aruco_scale(orig_img)
    except RuntimeError as err:
        print(err)
        return

    print(f"Detected ArUco Marker ID: {marker_id}")
    print(f"Scale Factor: {mm_per_px:.5f} mm/pixel")
    print("\nControls:")
    print("  [Left Click]   : Select points")
    print("  [Right Click]  : Remove last point")
    print("  [Scroll Wheel] : Zoom in/out at cursor")
    print("  [Middle Drag]  : Pan image")
    print("  [ + / - ]      : Keyboard zoom fallback")
    print("  [ W/A/S/D ]    : Keyboard pan fallback")
    print("  [Space/Enter]  : Fit circle")
    print("  [R]            : Reset selected points")
    print("  [Q/Esc]        : Exit")

    clicked_points = []
    display_img = orig_img.copy()

    # Viewport variables for zoom and pan
    zoom_scale = 1.0
    pan_x = 0.0
    pan_y = 0.0
    is_panning = False
    last_mouse_pos = (0, 0)
    fitted_circle_data = None

    # Draw detected ArUco marker boundary on the base image
    int_corners = np.int32(marker_corners)
    cv2.polylines(display_img, [int_corners], isClosed=True, color=(0, 255, 0), thickness=2)
    cv2.putText(
        display_img,
        f"ArUco ID {marker_id} (60mm)",
        (int_corners[0][0], int_corners[0][1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )

    window_name = "Tube Precurvature Measurement"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    def mouse_callback(event, x, y, flags, param):
        nonlocal clicked_points, zoom_scale, pan_x, pan_y, is_panning, last_mouse_pos

        # Convert screen coordinates to original image space
        img_x = (x - pan_x) / zoom_scale
        img_y = (y - pan_y) / zoom_scale

        if event == cv2.EVENT_LBUTTONDOWN:
            clicked_points.append((img_x, img_y))
            redraw()
        elif event == cv2.EVENT_RBUTTONDOWN:
            if clicked_points:
                clicked_points.pop()
                redraw()
        elif event == cv2.EVENT_MBUTTONDOWN:
            is_panning = True
            last_mouse_pos = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE:
            if is_panning:
                pan_x += (x - last_mouse_pos[0])
                pan_y += (y - last_mouse_pos[1])
                last_mouse_pos = (x, y)
                redraw()
        elif event == cv2.EVENT_MBUTTONUP:
            is_panning = False
        elif event == cv2.EVENT_MOUSEWHEEL:
            zoom_factor = 1.1 if flags > 0 else 0.9
            # Keep the zoom centered on the current mouse position
            pan_x = x - (x - pan_x) * zoom_factor
            pan_y = y - (y - pan_y) * zoom_factor
            zoom_scale *= zoom_factor
            redraw()

    cv2.setMouseCallback(window_name, mouse_callback)

    def redraw():
        # Create an affine transformation matrix for the current pan and zoom
        M = np.float32([[zoom_scale, 0, pan_x], [0, zoom_scale, pan_y]])
        h, w = orig_img.shape[:2]
        
        # Warp the base image into the viewport
        frame = cv2.warpAffine(display_img, M, (w, h))

        # Map saved image points back to current screen space
        for i, pt in enumerate(clicked_points):
            sx = int(pt[0] * zoom_scale + pan_x)
            sy = int(pt[1] * zoom_scale + pan_y)
            cv2.circle(frame, (sx, sy), 4, (0, 0, 255), -1)
            if i > 0:
                prev_pt = clicked_points[i - 1]
                px = int(prev_pt[0] * zoom_scale + pan_x)
                py = int(prev_pt[1] * zoom_scale + pan_y)
                cv2.line(frame, (px, py), (sx, sy), (0, 165, 255), 1)

        if fitted_circle_data is not None:
            xc, yc, r_px, r_mm, kappa_inv_mm, kappa_inv_m = fitted_circle_data
            
            # Map circle center to screen space
            sxc = int(xc * zoom_scale + pan_x)
            syc = int(yc * zoom_scale + pan_y)
            s_radius = int(r_px * zoom_scale)

            cv2.circle(frame, (sxc, syc), s_radius, (255, 255, 0), 2)
            cv2.circle(frame, (sxc, syc), 5, (255, 0, 0), -1)

            # Draw static text fixed to the screen (not affected by zoom)
            text_lines = [
                f"Radius (R): {r_mm:.2f} mm",
                f"Precurvature (kappa): {kappa_inv_mm:.4f} mm^-1",
            ]
            for idx, line in enumerate(text_lines):
                cv2.putText(frame, line, (30, 40 + idx * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
                cv2.putText(frame, line, (30, 40 + idx * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow(window_name, frame)

    redraw()

    # Main loop with keyboard fallbacks
    pan_step = 50
    while True:
        key = cv2.waitKey(20) & 0xFF
        if key in (27, ord("q")):
            break
        elif key == ord("r"):
            clicked_points.clear()
            fitted_circle_data = None
            redraw()
        elif key in (ord("="), ord("+")): # Zoom in
            center_x, center_y = display_img.shape[1] // 2, display_img.shape[0] // 2
            pan_x = center_x - (center_x - pan_x) * 1.1
            pan_y = center_y - (center_y - pan_y) * 1.1
            zoom_scale *= 1.1
            redraw()
        elif key == ord("-"): # Zoom out
            center_x, center_y = display_img.shape[1] // 2, display_img.shape[0] // 2
            pan_x = center_x - (center_x - pan_x) * 0.9
            pan_y = center_y - (center_y - pan_y) * 0.9
            zoom_scale *= 0.9
            redraw()
        elif key == ord("w"):
            pan_y += pan_step
            redraw()
        elif key == ord("s"):
            pan_y -= pan_step
            redraw()
        elif key == ord("a"):
            pan_x += pan_step
            redraw()
        elif key == ord("d"):
            pan_x -= pan_step
            redraw()
        elif key in (13, 32):  # Enter or Space
            if len(clicked_points) < 3:
                print("Click at least 3 points along the curved portion of the tube.")
                continue

            xc, yc, r_px = fit_circle(clicked_points)
            r_mm = r_px * mm_per_px
            kappa_inv_mm = 1.0 / r_mm
            kappa_inv_m = 1000.0 / r_mm

            print(f"\n--- Results ---")
            print(f"Fitted Radius (R)        : {r_mm:.2f} mm")
            print(f"Precurvature (kappa)     : {kappa_inv_mm:.5f} mm^-1 ({kappa_inv_m:.2f} m^-1)")

            fitted_circle_data = (xc, yc, r_px, r_mm, kappa_inv_mm, kappa_inv_m)
            redraw()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python measure_precurvature.py <path_to_image>")
    else:
        run_precurvature_measurement(sys.argv[1])
