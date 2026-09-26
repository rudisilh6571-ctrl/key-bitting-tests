import argparse
import sys
from dataclasses import dataclass, field

import cv2
import numpy as np
import trimesh
from shapely.geometry import Polygon

# Defaults
DEFAULT_MM_PER_PX = 0.1       # Fallback scale when no key length is given (1 pixel = 0.1 mm)
DEFAULT_THICKNESS = 2.0       # Key thickness in mm
DEFAULT_NUM_CUTS = 0          # Number of bitting cuts; 0 = detect automatically
DEFAULT_DEPTH_STEP = 0.381    # Depth increment per bitting value in mm (Schlage: 0.015")


@dataclass
class Groove:
    """A groove milled along the blade, measured across the blade from the spine (the uncut back edge)."""
    face: str      # "front" = the side facing the camera in the photo, "back" = the other side
    start: float   # mm from the spine to the groove's near edge
    end: float     # mm from the spine to the groove's far edge
    depth: float   # mm into the blade


def parse_grooves(text):
    """Parse "front 1.0-2.2 0.6; back 3-4.5 0.5" into Grooves."""
    grooves = []
    for part in text.replace("\n", ";").split(";"):
        if not part.strip():
            continue
        try:
            face, span, depth = part.split()
            start, end = (float(v) for v in span.split("-"))
            g = Groove(face.lower(), min(start, end), max(start, end), float(depth))
        except ValueError:
            raise ValueError(f"Can't read groove '{part.strip()}'. Use: face from-to depth, "
                             "e.g. 'front 1.0-2.2 0.6'.")
        if g.face not in ("front", "back"):
            raise ValueError(f"Groove face must be 'front' or 'back', not '{face}'.")
        grooves.append(g)
    return grooves


def parse_checks(text):
    """Parse "9:20.45; 20:7.44" (mm from the bow end : measured width) into (pos, width) pairs."""
    checks = []
    for part in text.replace(",", ";").split(";"):
        if not part.strip():
            continue
        try:
            pos, width = (float(v) for v in part.split(":"))
        except ValueError:
            raise ValueError(f"Can't read check width '{part.strip()}'. Use: position:width in mm, "
                             "e.g. '9:20.45'.")
        checks.append((pos, width))
    return checks


@dataclass
class KeyResult:
    mesh: trimesh.Trimesh
    mm_per_px: float
    length_mm: float
    width_mm: float
    blade_width_mm: float = None  # spine to the uncut bitting edge
    correction: dict = None       # outline padding removed using check widths, if any
    cut_depths_mm: list = field(default_factory=list)
    bitting_code: list = field(default_factory=list)
    overlay: np.ndarray = None  # BGR image of the aligned key with detection drawn on it
    warnings: list = field(default_factory=list)


def segment_key(image):
    """Return the key's outer contour and a list of hole contours (e.g. the key-ring hole)."""
    h, w = image.shape[:2]
    # Estimate the background colour from the image border and measure each pixel's distance from it,
    # which works for brass-on-white, silver-on-dark and keys on coloured paper alike.
    border = np.concatenate([image[0], image[-1], image[:, 0], image[:, -1]]).astype(np.float32)
    bg = np.median(border, axis=0)
    dist = np.linalg.norm(image.astype(np.float32) - bg, axis=2)
    dist = cv2.GaussianBlur(dist, (0, 0), 1.5)
    # "Looks like background" = within the background's own noise level. A silver key on grey paper is
    # barely a different colour, so a split between the two (e.g. Otsu) loses most of the metal.
    border_noise = np.percentile(np.linalg.norm(border - bg, axis=1), 99)
    paper = dist < max(20.0, border_noise * 1.3)

    # Background is the paper-like area reachable from the image border without crossing an edge, so
    # glare on the metal (as bright as the paper, but enclosed by the key's outline) stays part of the key.
    gray = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    edges = cv2.dilate(cv2.Canny(gray, 20, 60), np.ones((3, 3), np.uint8))
    open_paper = (paper & (edges == 0)).astype(np.uint8)
    _, labels = cv2.connectedComponents(open_paper, connectivity=4)
    outside = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    background = np.isin(labels, outside[outside != 0]).astype(np.uint8)
    # Win back the thin band the edge barrier took from the background.
    background = cv2.dilate(background, np.ones((5, 5), np.uint8)) & paper

    k = max(3, int(min(h, w) * 0.006)) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(((1 - background) * 255).astype(np.uint8), cv2.MORPH_OPEN, kernel)
    # Paper seen through the key (the key-ring hole) is enclosed, so it isn't background above.
    enclosed = cv2.morphologyEx(((paper & (background == 0)) * 255).astype(np.uint8), cv2.MORPH_OPEN, kernel)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("Could not detect a key in the image.")
    def touches_edge(c):
        bx, by, bw, bh = cv2.boundingRect(c)
        return bx <= 1 or by <= 1 or bx + bw >= w - 1 or by + bh >= h - 1

    # Objects cut off by the frame (table edges, fabric, the paper's own border) can't be the key.
    outer = [i for i in range(len(contours)) if hierarchy[0][i][3] == -1]
    inside = [i for i in outer if not touches_edge(contours[i])]
    if not inside:
        raise ValueError("Could not detect a key in the image. Make sure the whole key is in frame "
                         "with a margin of background around it.")
    best = max(inside, key=lambda i: cv2.contourArea(contours[i]))
    area = cv2.contourArea(contours[best])
    if area < 0.005 * h * w:
        raise ValueError("Could not detect a key in the image (largest object is too small).")

    # Keep the key's own holes (e.g. the key-ring hole). Glare on shiny metal can be as bright as the
    # background and shows up as ragged "holes" too, so only keep ones that are big, round and solid.
    def is_real_hole(c):
        a = cv2.contourArea(c)
        if a < 0.005 * area:
            return False
        circularity = 4 * np.pi * a / (cv2.arcLength(c, True) ** 2)
        solidity = a / max(1e-6, cv2.contourArea(cv2.convexHull(c)))
        return circularity > 0.6 and solidity > 0.9

    key_region = np.zeros((h, w), np.uint8)
    cv2.drawContours(key_region, contours, best, 255, -1)
    candidates, _ = cv2.findContours(enclosed & key_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    holes = [c for c in candidates if is_real_hole(c)]
    return contours[best], holes


def principal_angle(contour):
    """Angle (degrees) of the contour's long axis, from PCA of its points."""
    pts = contour.reshape(-1, 2).astype(np.float64)
    pts -= pts.mean(axis=0)
    _, vecs = np.linalg.eigh(np.cov(pts.T))
    vx, vy = vecs[:, -1]
    angle = np.degrees(np.arctan2(vy, vx))
    # An axis has no direction; pick the smaller turn so an already-level key isn't flipped over.
    return (angle + 90) % 180 - 90


def blade_geometry(aligned_mask):
    """Locate the blade in an aligned mask (bow left, teeth down).

    Returns a dict with the blade's columns (xs) and their top/bottom edges, the spine line
    (top = slope * x + spine_c), the uncut bitting edge line (bottom = slope * x + edge_c), the
    typical blade height and where the bow ends (bow_x). None if there is no recognisable blade.
    """
    cols = np.where(aligned_mask.any(axis=0))[0]
    heights = np.array([np.count_nonzero(aligned_mask[:, x]) for x in cols])
    n = len(cols)
    blade_h = np.median(heights[int(n * 0.6):])
    # Shoulder: last column (from the bow side) that is clearly wider than the blade.
    wide = np.where(heights[: int(n * 0.7)] > blade_h * 1.3)[0]
    start = (wide[-1] + 1) if len(wide) else int(n * 0.3)
    start += int(n * 0.01)
    end = n - int(n * 0.04)  # ignore the tip taper
    if end - start < 8:
        return None
    # The bow proper (not the stop lugs, which are only a bit wider than the blade).
    bow = np.where(heights[: int(n * 0.7)] > max(blade_h * 2, heights.max() * 0.6))[0]
    bow_x = int(cols[bow[-1] + 1]) if len(bow) else int(cols[start])

    xs = cols[start:end]
    top = np.array([np.nonzero(aligned_mask[:, x])[0].min() for x in xs])
    bottom = np.array([np.nonzero(aligned_mask[:, x])[0].max() for x in xs])
    # The key's back (top) edge is straight; use its slope to cancel any leftover tilt, then take
    # the uncut blade edge as the line with that slope through the lowest bottom-edge points.
    back = slice(0, max(2, int(len(xs) * 0.85)))
    slope, spine_c = np.polyfit(xs[back], top[back], 1)
    edge_c = np.percentile(bottom - slope * xs, 97)
    return dict(xs=xs, top=top, bottom=bottom, slope=slope, spine_c=spine_c, edge_c=edge_c,
                blade_h=blade_h, bow_x=bow_x)


def shift(arr, dy, dx):
    """Shift a 2D array so out[y, x] = arr[y - dy, x - dx], filling with zeros."""
    out = np.zeros_like(arr)
    h, w = arr.shape
    out[max(0, dy):h + min(0, dy), max(0, dx):w + min(0, dx)] = \
        arr[max(0, -dy):h + min(0, -dy), max(0, -dx):w + min(0, -dx)]
    return out


def correct_padding(view, view_img, checks, mm_per_px_along):
    """Fix the scale across the key and remove the padding that shadows add to its outline.

    A photo that isn't taken straight down under flat light shows the key too wide by a fixed amount
    on its shadowed side, and a slightly tilted camera also squashes it across. Caliper widths at two
    or more places along the key separate the two: photo_px * mm_per_px_across = true_width + padding.
    The padding comes off the side whose edge is darker (the shadow). The length still sets the scale
    along the key. `view` has the bow on the left and the teeth down.
    """
    if len(checks) < 2:
        raise ValueError("Correcting the outline needs at least two check widths.")
    inside = view > 0
    cols = np.nonzero(inside.any(axis=0))[0]
    photo_px = []
    for pos, _ in checks:
        c, r = cols[0] + int(round(pos / mm_per_px_along)), max(1, int(round(0.5 / mm_per_px_along)))
        xs = range(max(cols[0], c - r), min(cols[-1], c + r) + 1)
        if not len(xs):
            raise ValueError(f"Check position {pos} mm is off the end of the key.")
        photo_px.append(np.median([np.count_nonzero(inside[:, x]) for x in xs]))
    photo_px = np.array(photo_px, dtype=np.float64)
    true = np.array([w for _, w in checks])
    (k, pad), *_ = np.linalg.lstsq(np.c_[photo_px, -np.ones(len(true))], true, rcond=None)
    if k <= 0 or not 0.5 < k / mm_per_px_along < 2:
        raise ValueError("The check widths don't fit this photo. Check where each was measured.")
    warnings = []
    if pad < 0:
        warnings.append("The photo reads narrower than the check widths; only padding can be removed, "
                        "so the outline wasn't trimmed.")
    pad_px = int(round(max(0.0, pad) / k))

    # Judge the shadow on the bow only: it's plain metal on both sides, whereas the blade's edges mix
    # grooves, glare and cuts that would swamp the comparison.
    gray = cv2.cvtColor(view_img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    heights = inside.sum(axis=0)
    bow_region = np.zeros_like(inside)
    bow_region[:, : np.nonzero(heights > heights.max() * 0.6)[0][-1] + 1] = True

    def edge_band(dy):
        # Key pixels within dy of the outline on that side; a shadow makes this band darker.
        band = inside & ~shift(inside, -dy, 0) & bow_region
        return float(gray[band].mean()) if band.any() else 255.0

    spine_band, teeth_band = edge_band(-max(1, pad_px)), edge_band(max(1, pad_px))
    if abs(spine_band - teeth_band) < 0.1 * max(spine_band, teeth_band):
        spine_px, teeth_px = pad_px // 2, pad_px - pad_px // 2  # no clear shadow side: split it
    elif teeth_band < spine_band:
        spine_px, teeth_px = 0, pad_px
    else:
        spine_px, teeth_px = pad_px, 0
    corrected = inside.copy()
    if spine_px:
        corrected &= shift(inside, spine_px, 0)
    if teeth_px:
        corrected &= shift(inside, -teeth_px, 0)

    info = dict(mm_per_px_across=float(k), scale_change=float(k / mm_per_px_along - 1),
                spine=spine_px * k, teeth=teeth_px * k,
                checks=[(pos, w, float(px * k - pad)) for (pos, w), px in zip(checks, photo_px)],
                warnings=warnings)
    return corrected.astype(np.uint8) * 255, info


def analyze_bitting(blade, num_cuts, mm_per_px):
    """Measure cut depths along the bitting edge of the blade found by blade_geometry().

    Returns (depths_mm, roots, span) where roots are (x, y) pixel positions of the cut roots
    and span is (x_start, x_end) of the toothed region.
    """
    xs, bottom, blade_h = blade["xs"], blade["bottom"], blade["blade_h"]
    if len(xs) < num_cuts * 4:
        return [], [], None
    depth = blade["edge_c"] - (bottom - blade["slope"] * xs)
    thr = max(2, 0.04 * blade_h)
    win = max(3, int(len(xs) * 0.01)) | 1
    smooth = np.convolve(depth, np.ones(win) / win, mode="same")
    toothed = np.where(smooth > thr)[0]
    if len(toothed) == 0:
        return [], [], None

    # The toothed span runs from the first cut to the last point where the edge climbs back to full
    # width. Anything after that is the tip bevel, not a cut.
    a = toothed[0]
    recoveries = np.where((smooth[:-1] > thr) & (smooth[1:] <= thr))[0]
    b = (recoveries[-1] + 1) if len(recoveries) else toothed[-1] + 1

    if num_cuts:
        # Known cut count: split the span evenly (real keys use fixed cut spacing).
        seg = (b - a) / num_cuts
        idx = []
        for i in range(num_cuts):
            s, e = a + int(round(i * seg)), a + int(round((i + 1) * seg))
            idx.append(s + int(np.argmax(depth[s:e])))
    else:
        idx = find_cut_roots(smooth[a:b], thr)
        idx = [a + i for i in idx]
        # Report the true deepest point near each smoothed peak.
        idx = [max(a, i - win) + int(np.argmax(depth[max(a, i - win): i + win + 1])) for i in idx]

    depths = [float(depth[j]) * mm_per_px for j in idx]
    roots = [(int(xs[j]), int(bottom[j])) for j in idx]
    return depths, roots, (int(xs[a]), int(xs[b - 1]))


def find_cut_roots(profile, thr):
    """Indices of the distinct notches (local depth maxima) in a smoothed depth profile."""
    n = len(profile)
    r = max(2, int(n * 0.03))
    peaks = [i for i in range(n)
             if profile[i] > thr and profile[i] >= profile[max(0, i - r): i + r + 1].max()]
    # Collapse flat-bottomed cuts (runs of equal maxima) to their centre.
    groups = []
    for i in peaks:
        if groups and i - groups[-1][-1] <= r:
            groups[-1].append(i)
        else:
            groups.append([i])
    peaks = [int(round(np.mean(g))) for g in groups]

    # Two neighbouring maxima are separate cuts only if the edge rises noticeably between them.
    prom = max(thr * 0.5, 0.1 * profile.max())
    merged = True
    while merged and len(peaks) > 1:
        merged = False
        for k in range(len(peaks) - 1):
            p, q = peaks[k], peaks[k + 1]
            if min(profile[p], profile[q]) - profile[p:q + 1].min() < prom:
                keep = p if profile[p] >= profile[q] else q
                peaks[k:k + 2] = [keep]
                merged = True
                break
    return peaks


def process_key_image(image_path, key_length_mm=None, mm_per_px=DEFAULT_MM_PER_PX,
                      thickness=DEFAULT_THICKNESS, num_cuts=DEFAULT_NUM_CUTS,
                      depth_step=DEFAULT_DEPTH_STEP, grooves=(), check_widths=()):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not load the image.")
    warnings = []

    contour, holes = segment_key(image)

    # Rotate so the key's long axis is horizontal.
    angle = principal_angle(contour)
    h, w = image.shape[:2]
    cx, cy = w / 2, h / 2
    rot = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    diag = int(np.hypot(h, w)) + 4
    rot[0, 2] += diag / 2 - cx
    rot[1, 2] += diag / 2 - cy

    def transform(c):
        return cv2.transform(c.reshape(-1, 1, 2).astype(np.float32), rot).reshape(-1, 2)

    outer_pts = transform(contour)
    hole_pts = [transform(hc) for hc in holes]

    mask = np.zeros((diag, diag), np.uint8)
    cv2.drawContours(mask, [outer_pts.astype(np.int32)], -1, 255, -1)
    cv2.drawContours(mask, [p.astype(np.int32) for p in hole_pts], -1, 0, -1)
    aligned_img = cv2.warpAffine(image, rot, (diag, diag), borderValue=(255, 255, 255))

    # Crop to the key.
    x0, y0, bw, bh = cv2.boundingRect(outer_pts.astype(np.int32))
    pad = 20
    x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
    mask = mask[y0:y0 + bh + 2 * pad, x0:x0 + bw + 2 * pad]
    aligned_img = aligned_img[y0:y0 + bh + 2 * pad, x0:x0 + bw + 2 * pad].copy()
    outer_pts -= (x0, y0)
    hole_pts = [p - (x0, y0) for p in hole_pts]

    # Scale.
    length_px = float(bw)
    if key_length_mm:
        mm_per_px = key_length_mm / length_px
    else:
        warnings.append(f"No key length given; assuming {mm_per_px} mm per pixel. "
                        "Measure the key and enter its length for a correctly sized model.")

    # For the bitting analysis, flip a copy so the bow is on the left and the teeth face down.
    # (Flips mirror the image, so they are only used for measuring, never for the model.)
    heights = (mask > 0).sum(axis=0)
    cols = np.nonzero(heights)[0]
    third = max(1, len(cols) // 3)
    flip_h = heights[cols[:third]].mean() < heights[cols[-third:]].mean()
    view = mask[:, ::-1] if flip_h else mask
    view_img = aligned_img[:, ::-1] if flip_h else aligned_img
    vcols = np.nonzero(view.any(axis=0))[0]
    blade = vcols[int(len(vcols) * 0.45): int(len(vcols) * 0.95)]
    top = np.array([np.nonzero(view[:, x])[0].min() for x in blade])
    bottom = np.array([np.nonzero(view[:, x])[0].max() for x in blade])
    flip_v = np.std(np.diff(top)) > np.std(np.diff(bottom))
    if flip_v:
        view = view[::-1]
        view_img = view_img[::-1]
    view_img = np.ascontiguousarray(view_img)

    correction = None
    mm_across = mm_per_px  # scale across the key; differs from along only after a correction
    if check_widths:
        if not key_length_mm:
            raise ValueError("Check widths need the key length as well.")
        view, correction = correct_padding(view, view_img, check_widths, mm_per_px)
        mm_across = correction["mm_per_px_across"]
        warnings.extend(correction.pop("warnings"))
        # Rebuild the model's outline from the corrected mask (undoing the analysis flips).
        mask = view[::-1] if flip_v else view
        mask = np.ascontiguousarray(mask[:, ::-1] if flip_h else mask)
        cs, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        best = max((i for i in range(len(cs)) if hier[0][i][3] == -1), key=lambda i: cv2.contourArea(cs[i]))
        outer_pts = cs[best].reshape(-1, 2).astype(np.float32)
        hole_pts = [cs[i].reshape(-1, 2).astype(np.float32) for i in range(len(cs)) if hier[0][i][3] == best]

    blade = blade_geometry(view)
    depths, roots, span = analyze_bitting(blade, num_cuts, mm_across) if blade else ([], [], None)
    code = [int(round(d / depth_step)) for d in depths]

    # Detection overlay (in the analysis view).
    overlay = view_img.copy()
    vcontours, _ = cv2.findContours(view, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, vcontours, -1, (0, 200, 0), 2)
    if span:
        for x in span:
            cv2.line(overlay, (x, 0), (x, overlay.shape[0]), (255, 150, 0), 1)
    for (x, y), c in zip(roots, code):
        cv2.circle(overlay, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(overlay, str(c), (x - 6, min(overlay.shape[0] - 4, y + 24)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Build the 3D model from the (rotated, unflipped) outline. Image y points down, so negate y
    # to keep the model the same handedness as the key in the photo.
    def to_mm(p):
        q = cv2.approxPolyDP(p.reshape(-1, 1, 2).astype(np.float32), 0.75, True).reshape(-1, 2)
        q = q.astype(np.float64) * (mm_per_px, -mm_across)
        return q

    shell = to_mm(outer_pts)
    hole_rings = [to_mm(p) for p in hole_pts if len(p) >= 3]
    poly = Polygon(shell, [r for r in hole_rings if len(r) >= 3]).buffer(0)
    if poly.geom_type == "MultiPolygon":
        poly = max(poly.geoms, key=lambda g: g.area)
    mesh = trimesh.creation.extrude_polygon(poly, height=thickness)
    offset = -mesh.bounds[0][:2]
    mesh.apply_translation([*offset, 0])

    blade_width = None
    if blade:
        blade_width = (blade["edge_c"] - blade["spine_c"]) * mm_across
    if grooves:
        if not blade:
            raise ValueError("Could not find the blade, so the grooves can't be placed.")

        def view_to_model(pts):
            # Undo the analysis flips, then apply the same pixel -> mm mapping as the outline.
            pts = np.array(pts, dtype=np.float64)
            if flip_v:
                pts[:, 1] = view.shape[0] - 1 - pts[:, 1]
            if flip_h:
                pts[:, 0] = view.shape[1] - 1 - pts[:, 0]
            pts *= (mm_per_px, -mm_across)
            return pts + offset

        mesh = cut_grooves(mesh, grooves, blade, view.shape[1], mm_across, thickness,
                           blade_width, view_to_model)
        for g in grooves:
            y = [blade["spine_c"] + v / mm_across for v in (g.start, g.end)]
            x0, x1 = blade["bow_x"], overlay.shape[1] - 1
            s = blade["slope"]
            pts = np.array([[x0, y[0] + s * x0], [x1, y[0] + s * x1], [x1, y[1] + s * x1],
                            [x0, y[1] + s * x0]], np.int32)
            cv2.polylines(overlay, [pts], True, (255, 0, 255) if g.face == "front" else (0, 165, 255), 2)

    return KeyResult(mesh=mesh, mm_per_px=mm_per_px, length_mm=float(mesh.extents[0]),
                     width_mm=float(mesh.extents[1]), blade_width_mm=blade_width, correction=correction,
                     cut_depths_mm=depths, bitting_code=code, overlay=overlay, warnings=warnings)


def cut_grooves(mesh, grooves, blade, view_width, mm_per_px, thickness, blade_width, view_to_model):
    """Mill each groove along the blade, from the bow to past the tip, parallel to the spine."""
    for g in grooves:
        if g.depth >= thickness:
            raise ValueError(f"A {g.depth} mm deep groove would cut through a {thickness} mm key.")
        if blade_width and g.start >= blade_width:
            raise ValueError(f"Groove at {g.start}-{g.end} mm starts beyond the blade, which is only "
                             f"{blade_width:.2f} mm wide.")
    for a in grooves:
        for b in grooves:
            if (a.face == "front" and b.face == "back" and a.start < b.end and b.start < a.end
                    and a.depth + b.depth >= thickness):
                raise ValueError(f"The front groove at {a.start}-{a.end} mm and the back groove at "
                                 f"{b.start}-{b.end} mm overlap and together cut through the key.")

    x0, x1 = blade["bow_x"], view_width + 10
    slope, spine_c = blade["slope"], blade["spine_c"]
    cutters = []
    for g in grooves:
        # Grooves that start at the spine are open-sided steps; run the cutter past the edge.
        near = g.start - (1.0 if g.start <= 0.05 else 0.0)
        y0, y1 = spine_c + near / mm_per_px, spine_c + g.end / mm_per_px
        corners = view_to_model([[x0, y0 + slope * x0], [x1, y0 + slope * x1],
                                 [x1, y1 + slope * x1], [x0, y1 + slope * x0]])
        margin = 0.5
        cutter = trimesh.creation.extrude_polygon(Polygon(corners).buffer(0), height=g.depth + margin)
        z = thickness - g.depth if g.face == "front" else -margin
        cutter.apply_translation([0, 0, z])
        cutters.append(cutter)
    return trimesh.boolean.difference([mesh, *cutters])


def summarize(result, output_path):
    lines = [f"Key size: {result.length_mm:.1f} x {result.width_mm:.1f} mm "
             f"({result.mm_per_px:.4f} mm/px)"]
    c = result.correction
    if c:
        sides = " and ".join(f"{v:.2f} mm of shadow off the {name} side" for name, v in
                             (("spine", c["spine"]), ("teeth", c["teeth"])) if v >= 0.005)
        lines.append(f"Outline corrected: scale across the key {c['scale_change']:+.1%}; "
                     f"removed {sides or 'no padding'}")
        lines.append("Check widths (measured -> model): " + ", ".join(
            f"{w:.2f} -> {m:.2f} at {pos:g} mm" for pos, w, m in c["checks"]))
    if result.blade_width_mm:
        lines.append(f"Blade width (spine to uncut edge): {result.blade_width_mm:.2f} mm")
    if result.bitting_code:
        lines.append("Bitting code (estimate): " + " ".join(map(str, result.bitting_code)))
        lines.append("Cut depths (mm): " + ", ".join(f"{d:.2f}" for d in result.cut_depths_mm))
    else:
        lines.append("Could not measure the bitting.")
    lines.append(f"3D model saved as '{output_path}'")
    lines.extend(result.warnings)
    return "\n".join(lines)


class KeyForge3DApp:
    def __init__(self, root):
        import tkinter as tk

        self.tk = tk
        self.root = root
        self.root.title("KeyForge3D - Key Shape Extractor and 3D Model Generator")
        self.root.geometry("720x680")

        # Variables
        self.image_path = None
        self.key_length = tk.StringVar(value="")
        self.thickness = tk.StringVar(value=str(DEFAULT_THICKNESS))
        self.num_cuts = tk.StringVar(value=str(DEFAULT_NUM_CUTS))
        self.depth_step = tk.StringVar(value=str(DEFAULT_DEPTH_STEP))
        self.grooves = tk.StringVar(value="")
        self.checks = tk.StringVar(value="")

        # GUI Elements
        self.label = tk.Label(root, text="KeyForge3D: Extract and 3D Print Keys", font=("Arial", 16))
        self.label.pack(pady=10)

        self.upload_button = tk.Button(root, text="Upload Key Image", command=self.upload_image)
        self.upload_button.pack(pady=5)

        self.image_label = tk.Label(root)
        self.image_label.pack(pady=5)

        settings = tk.Frame(root)
        settings.pack(pady=5)
        fields = [("Key length (mm)", self.key_length), ("Thickness (mm)", self.thickness),
                  ("Cuts (0 = auto)", self.num_cuts), ("Depth step (mm)", self.depth_step)]
        for col, (text, var) in enumerate(fields):
            tk.Label(settings, text=text).grid(row=0, column=col, padx=6)
            tk.Entry(settings, textvariable=var, width=10).grid(row=1, column=col, padx=6)
        tk.Label(settings, text="Grooves (face from-to depth, mm from the spine; e.g. "
                                "front 1.0-2.2 0.6; back 3-4.5 0.5)").grid(row=2, column=0, columnspan=4, pady=(8, 0))
        tk.Entry(settings, textvariable=self.grooves, width=60).grid(row=3, column=0, columnspan=4)
        tk.Label(settings, text="Check widths (mm from the bow end:caliper width; e.g. 9:20.45; 20:7.44)"
                 ).grid(row=4, column=0, columnspan=4, pady=(8, 0))
        tk.Entry(settings, textvariable=self.checks, width=60).grid(row=5, column=0, columnspan=4)

        self.process_button = tk.Button(root, text="Process Key and Generate 3D Model",
                                        command=self.process_key, state=tk.DISABLED)
        self.process_button.pack(pady=5)

        self.result_label = tk.Label(root, text="", font=("Arial", 12), justify=tk.LEFT, wraplength=680)
        self.result_label.pack(pady=5)

    def show_image(self, img):
        from PIL import ImageTk

        img = img.copy()
        img.thumbnail((680, 340))
        img_tk = ImageTk.PhotoImage(img)
        self.image_label.config(image=img_tk)
        self.image_label.image = img_tk  # Keep a reference to avoid garbage collection

    def upload_image(self):
        """Allow the user to upload an image of a key."""
        from tkinter import filedialog
        from PIL import Image

        path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.JPG *.JPEG *.PNG *.webp"), ("All Files", "*")])
        if path:
            self.image_path = path
            self.show_image(Image.open(path))
            self.process_button.config(state=self.tk.NORMAL)
            self.result_label.config(text="Image uploaded. Enter the key's length for accurate sizing, "
                                          "then click 'Process Key'.")

    def process_key(self):
        """Process the key image, extract the shape, and generate a 3D model."""
        from tkinter import filedialog, messagebox
        from PIL import Image

        if not self.image_path:
            messagebox.showerror("Error", "Please upload an image first.")
            return

        try:
            length = self.key_length.get().strip()
            result = process_key_image(
                self.image_path,
                key_length_mm=float(length) if length else None,
                thickness=float(self.thickness.get()),
                num_cuts=int(self.num_cuts.get()),
                depth_step=float(self.depth_step.get()),
                grooves=parse_grooves(self.grooves.get()),
                check_widths=parse_checks(self.checks.get()),
            )
            self.show_image(Image.fromarray(cv2.cvtColor(result.overlay, cv2.COLOR_BGR2RGB)))

            output_path = filedialog.asksaveasfilename(
                defaultextension=".stl", initialfile="key_model.stl", filetypes=[("STL", "*.stl")])
            if not output_path:
                self.result_label.config(text="Processed, but no file was saved.")
                return
            result.mesh.export(output_path)
            self.result_label.config(text="Success!\n" + summarize(result, output_path))

        except Exception as e:
            messagebox.showerror("Error", f"Failed to process the key: {str(e)}")
            self.result_label.config(text="Error processing the key. See error message.")


def main():
    parser = argparse.ArgumentParser(description="Extract a key's shape from a photo and export an STL.")
    parser.add_argument("image", nargs="?", help="Key photo. Omit to launch the GUI.")
    parser.add_argument("-o", "--output", default="key_model.stl", help="Output STL path")
    parser.add_argument("-l", "--length", type=float, help="Overall key length in mm (sets the scale)")
    parser.add_argument("--mm-per-px", type=float, default=DEFAULT_MM_PER_PX,
                        help="Scale to use when --length is not given")
    parser.add_argument("-t", "--thickness", type=float, default=DEFAULT_THICKNESS)
    parser.add_argument("-c", "--cuts", type=int, default=DEFAULT_NUM_CUTS,
                        help="Number of cuts if known (0 = detect automatically)")
    parser.add_argument("--depth-step", type=float, default=DEFAULT_DEPTH_STEP)
    parser.add_argument("-g", "--groove", action="append", default=[],
                        help="Groove as 'face from-to depth' in mm, measured from the spine, e.g. "
                             "'front 1.0-2.2 0.6'. Face is front (towards the camera) or back. Repeatable.")
    parser.add_argument("--check", action="append", default=[],
                        help="Caliper width across the key at a distance from the bow end, as 'pos:width' "
                             "in mm, e.g. '9:20.45'. Give two or more to remove shadow padding and fix the scale.")
    parser.add_argument("--overlay", help="Also save the detection overlay image here")
    args = parser.parse_args()

    if args.image is None:
        import tkinter as tk

        root = tk.Tk()
        KeyForge3DApp(root)
        root.mainloop()
        return

    try:
        result = process_key_image(args.image, args.length, args.mm_per_px, args.thickness,
                                   args.cuts, args.depth_step, parse_grooves(";".join(args.groove)),
                                   parse_checks(";".join(args.check)))
    except ValueError as e:
        sys.exit(f"Error: {e}")
    result.mesh.export(args.output)
    if args.overlay:
        cv2.imwrite(args.overlay, result.overlay)
    print(summarize(result, args.output))


if __name__ == "__main__":
    main()
