def detect_risk_level(count, density_box, count_thresh=80, density_thresh=0.12):
    if count >= count_thresh or density_box >= density_thresh:
        return "HIGH RISK"
    if count >= int(count_thresh * 0.6) or density_box >= (density_thresh * 0.6):
        return "MEDIUM RISK"
    return "LOW / NORMAL"
