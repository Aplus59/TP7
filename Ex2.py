import cv2 as cv
import numpy as np

def binarize(gray):
    _, bw = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    return bw

def label_regions(binary_255):
    binary01 = (binary_255 > 0).astype(np.uint8)
    num_labels, labels = cv.connectedComponents(binary01, connectivity=8)
    return num_labels, labels

def moments_area_centroid_orientation(labels):
    objects = []
    max_label = labels.max()

    for lab in range(1, max_label + 1):
        mask = (labels == lab).astype(np.uint8)
        m = cv.moments(mask, binaryImage=True)
        m00 = m["m00"]
        if m00 == 0:
            continue

        cx = m["m10"] / m00
        cy = m["m01"] / m00

        mu20 = m["mu20"] / m00
        mu02 = m["mu02"] / m00
        mu11 = m["mu11"] / m00

        theta = 0.5 * np.arctan2(2.0 * mu11, (mu20 - mu02))
        theta_deg = theta * 180.0 / np.pi

        objects.append({
            "label": lab,
            "area": m00,
            "centroid": (cx, cy),
            "theta_deg": theta_deg
        })

    return objects

def compute_statistics(objects):
    areas = np.array([o["area"] for o in objects], dtype=np.float64)
    thetas = np.array([o["theta_deg"] for o in objects], dtype=np.float64)

    return {
        "count": len(objects),
        "area_min": areas.min(),
        "area_max": areas.max(),
        "area_mean": areas.mean(),
        "area_std": areas.std(),
        "theta_mean_deg": thetas.mean(),
        "theta_std_deg": thetas.std()
    }

def draw_results(gray, labels, objects):
    label_norm = (labels.astype(np.float32) / labels.max() * 255).astype(np.uint8)
    label_color = cv.applyColorMap(label_norm, cv.COLORMAP_JET)

    vis = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)

    for o in objects:
        cx, cy = o["centroid"]
        theta = o["theta_deg"] * np.pi / 180.0

        cxi, cyi = int(round(cx)), int(round(cy))
        cv.circle(vis, (cxi, cyi), 4, (0, 0, 255), -1)

        L = 40
        x2 = int(round(cx + L * np.cos(theta)))
        y2 = int(round(cy + L * np.sin(theta)))
        cv.line(vis, (cxi, cyi), (x2, y2), (0, 255, 0), 2)

    return label_color, vis


if __name__ == "__main__":
    img = cv.imread(r".\bin\zebra_1.tif", cv.IMREAD_GRAYSCALE)

    bw = binarize(img)
    num_labels, labels = label_regions(bw)

    objects = moments_area_centroid_orientation(labels)

    for o in objects:
        print(
            f"Label {o['label']} | Area={o['area']:.0f} | "
            f"Centroid=({o['centroid'][0]:.1f},{o['centroid'][1]:.1f}) | "
            f"Theta={o['theta_deg']:.2f} deg"
        )

    stats = compute_statistics(objects)
    print(stats)

    label_color, vis = draw_results(img, labels, objects)

    cv.imshow("Binary", bw)
    cv.imshow("Labels", label_color)
    cv.imshow("Orientation", vis)
    cv.waitKey(0)
    cv.destroyAllWindows()
