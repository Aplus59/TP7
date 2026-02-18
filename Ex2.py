import cv2 as cv
import numpy as np

def label_regions(binary_255):
    binary01 = (binary_255 > 0).astype(np.uint8)
    num_labels, labels = cv.connectedComponents(binary01, connectivity=8)
    return num_labels, labels

def moments_area_centroid(labels):
    objects = []
    max_label = int(labels.max())

    for lab in range(1, max_label + 1):
        mask = (labels == lab).astype(np.uint8)
        m = cv.moments(mask, binaryImage=True)
        m00 = m["m00"]
        if m00 == 0:
            continue

        cx = m["m10"] / m00
        cy = m["m01"] / m00

        objects.append({
            "label": int(lab),
            "area": float(m00),
            "centroid": (float(cx), float(cy)),
        })

    return objects

def compute_statistics(objects):
    if len(objects) == 0:
        return {
            "count": 0,
            "area_min": None,
            "area_max": None,
            "area_mean": None,
            "area_std": None,
            "cx_mean": None,
            "cy_mean": None,
        }

    areas = np.array([o["area"] for o in objects], dtype=np.float64)
    cxs = np.array([o["centroid"][0] for o in objects], dtype=np.float64)
    cys = np.array([o["centroid"][1] for o in objects], dtype=np.float64)

    return {
        "count": int(len(objects)),
        "area_min": float(areas.min()),
        "area_max": float(areas.max()),
        "area_mean": float(areas.mean()),
        "area_std": float(areas.std()),
        "cx_mean": float(cxs.mean()),
        "cy_mean": float(cys.mean()),
    }

def draw_results(binary_255, labels, objects):
    if labels.max() > 0:
        label_norm = (labels.astype(np.float32) / labels.max() * 255).astype(np.uint8)
    else:
        label_norm = np.zeros_like(labels, dtype=np.uint8)

    label_color = cv.applyColorMap(label_norm, cv.COLORMAP_JET)
    vis = cv.cvtColor(binary_255, cv.COLOR_GRAY2BGR)

    for o in objects:
        cx, cy = o["centroid"]
        cxi, cyi = int(round(cx)), int(round(cy))
        cv.circle(vis, (cxi, cyi), 3, (0, 0, 255), -1)
        cv.putText(
            vis, str(o["label"]),
            (cxi + 3, cyi - 3),
            cv.FONT_HERSHEY_SIMPLEX, 0.35,
            (0, 255, 255), 1, cv.LINE_AA
        )

    return label_color, vis

if __name__ == "__main__":
    binary_img = cv.imread(r".\bin\binary.png", cv.IMREAD_GRAYSCALE)

    num_labels, labels = label_regions(binary_img)
    objects = moments_area_centroid(labels)
    stats = compute_statistics(objects)

    for o in objects:
        print(
            f"Label {o['label']} | Area={o['area']:.0f} | "
            f"Centroid=({o['centroid'][0]:.1f},{o['centroid'][1]:.1f})"
        )

    print(stats)

    label_color, vis = draw_results(binary_img, labels, objects)
    cv.imshow("Binary", binary_img)
    cv.imshow("Label image", label_color)
    cv.imshow("Centroids", vis)
    cv.waitKey(0)
    cv.destroyAllWindows()
