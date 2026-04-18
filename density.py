def calculate_density(detections, frame_w, frame_h):
    count = len(detections)
    frame_area = frame_w * frame_h
    if frame_area <= 0:
        return count, 0.0

    total_box_area = 0
    for d in detections:
        x1, y1, x2, y2 = d[0], d[1], d[2], d[3]
        total_box_area += max(0, x2 - x1) * max(0, y2 - y1)

    density_box = total_box_area / frame_area
    return count, density_box


# ✅ ADDED: estimated count from coverage (NO existing code changed)
def estimate_count_from_coverage(detections, frame_area, avg_person_area=2500):
    total_area = 0
    for x1, y1, x2, y2, _ in detections:
        total_area += max(0, x2 - x1) * max(0, y2 - y1)

    coverage = total_area / frame_area if frame_area > 0 else 0
    est_count = int(total_area / avg_person_area)
    return est_count, coverage
