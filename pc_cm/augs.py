from collections import Counter
import random

import cv2
import numpy as np


def get_bounds(image):
    def _get_bounds(arr):
        for i in range(len(arr)):
            if arr[i] != 0:
                break
        for j in range(len(arr) - 1, -1, -1):
            if arr[j] != 0:
                break
        return (i, j)

    col_s, col_f = _get_bounds(image.sum(axis=0))
    row_s, row_f = _get_bounds(image.sum(axis=1))

    return (row_s, row_f), (col_s, col_f)

def shift_image_numpy(img, n_rows, n_cols):
    shifted = np.zeros_like(img)
    rows, cols = img.shape[:2]

    # Compute coordinate ranges
    r_start = max(0, n_rows)
    r_end = rows + min(0, n_rows)
    c_start = max(0, n_cols)
    c_end = cols + min(0, n_cols)

    # Source coordinates
    src_r_start = max(0, -n_rows)
    src_r_end = rows - max(0, n_rows)
    src_c_start = max(0, -n_cols)
    src_c_end = cols - max(0, n_cols)

    # Copy shifted region
    shifted[r_start:r_end, c_start:c_end] = img[src_r_start:src_r_end, src_c_start:src_c_end]
    return shifted


def equilize(numbers, diff=3):
    edges = []

    for i in range(len(numbers)):
        edges.append((i, i))
        for j in range(i, len(numbers)):
            if abs(numbers[i] - numbers[j]) <= diff:
                edges.append((i, j))

    group_idxs = connected_components(edges)

    results = []
    for group in group_idxs:
        group = [numbers[i] for i in group]
        results.append(
            Counter(group).most_common(1)[0][0]
        )
    return results


from collections import defaultdict


def connected_components(edges: list[tuple[int, int]]) -> list[list[int]]:
    adjacency_lists: dict[int, list[int]] = defaultdict(list)
    for e_i, e_j in edges:
        adjacency_lists[e_i].append(e_j)
        adjacency_lists[e_j].append(e_i)

    graph = adjacency_lists

    _components = defaultdict(set)
    processed_nodes = set()

    for root in graph:
        nodes_to_visit = [root]
        while nodes_to_visit:
            node = nodes_to_visit.pop()
            if node in processed_nodes:
                continue
            processed_nodes.add(node)
            nodes_to_visit.extend(graph[node])
            _components[root].update(graph[node])

    components = [list(nodes) for nodes in _components.values()]
    return components


def merge_coords(coords, image, radius=1):
    edges = []

    for i in range(len(coords)):
        edges.append((i, i))
        for j in range(i, len(coords)):
            if abs(coords[i][0] - coords[j][0]) <= 2 and abs(coords[i][1] - coords[j][1]) <= 2:
                edges.append((i, j))

    group_idxs = connected_components(edges)

    merged = []

    for _group_idxs in group_idxs:
        group = [coords[i] for i in _group_idxs]

        if len(group) == 1:
            merged.append(group[0])
            continue

        sums = []
        for gx, gy in group:
            x_min, x_max = max(0, gx - radius), min(image.shape[0], gx + radius + 1)
            y_min, y_max = max(0, gy - radius), min(image.shape[1], gy + radius + 1)
            region_sum = image[x_min:x_max, y_min:y_max].sum()
            sums.append(region_sum)

        best_coord = group[np.argmin(sums)]
        merged.append(best_coord)

    return merged


def find_corners(grayscale_image):
    blockSize = 2
    ksize = 3
    k = 0.04
    harris_response = cv2.cornerHarris(grayscale_image, blockSize, ksize, k)
    threshold = 0.01 * harris_response.max()
    coords = np.argwhere(harris_response > threshold).tolist()
    merged_coords = merge_coords([(y, x) for x, y in coords], grayscale_image, radius=5)
    merged_coords = merge_coords([(y, x) for x, y in merged_coords], grayscale_image, radius=5)
    return merged_coords


def find_holes(image):
    image = (image == 0).astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(image)
    if num_labels == 1:
        return []
    holes = [labels == i for i in range(1, num_labels)]
    holes = [hole for hole in holes if
             (hole[0, 0] == False and hole[0, -1] == False and hole[-1, 0] == False and hole[-1, -1] == False)]
    return holes




def add_or_delete_part_aug(opposite_views):
    image, other_view_image = opposite_views

    corners = find_corners(image)

    row_bounds, col_bounds = get_bounds(image)
    rows = sorted(equilize([x[0] for x in corners] + list(row_bounds)))
    cols = sorted(equilize([x[1] for x in corners] + list(col_bounds)))

    if len(rows) <= 2:
        return

    deleted = 0

    for i in range(random.randint(1, 4)):
        r_s, c_s = random.randint(0, len(rows) - 2), random.randint(0, len(cols) - 2)
        r_f, c_f = random.randint(r_s + 1, min(len(rows) - 1, r_s + 2)), random.randint(c_s + 1,
                                                                                        min(len(cols) - 1, c_s + 2))
        add = random.random() < 0.825
        if add:
            radius = 3
            s_row, s_col = max(0, rows[r_s] - radius), max(0, cols[c_s] - radius)
            f_row, f_col = min(rows[-1], rows[r_f] + 1 + radius), min(cols[-1], cols[c_f] + 1 + radius)
            area = image[s_row: f_row, s_col: f_col]
            area_sum = area.sum()
            if area_sum > 0.5:
                depth_color = area.max()
                image[rows[r_s]: rows[r_f] + 1, cols[c_s]: cols[c_f] + 2] = depth_color
                other_view_image[rows[r_s]: rows[r_f] + 1, cols[c_s]: cols[c_f] + 2] = (
                    depth_color if random.random() < 0.33 else random.randint(255 - depth_color + 1, 255))

        elif deleted < 2:
            image[rows[r_s]: rows[r_f] + 1, cols[c_s]: cols[c_f] + 2] = 0
            other_view_image[rows[r_s]: rows[r_f] + 1, cols[c_s]: cols[c_f] + 2] = 0
            deleted += 1


def shift_aug(view_to_image):
    img_size = view_to_image["+Z"].shape[0]

    (y_row_s, y_row_f), (x_min, x_max) = get_bounds(view_to_image["+Z"])
    y_min = img_size - y_row_f
    y_max = img_size - y_row_s

    (z_row_s, z_row_f), (_, _) = get_bounds(view_to_image["+Y"])
    z_min = img_size - z_row_f
    z_max = img_size - z_row_s

    n_shift_axis = random.choice((1, 2))
    axes = random.sample(("X", "Y", "Z"), k=n_shift_axis)

    for axis in axes:
        if axis == "X":
            max_shift = round((x_max - x_min) * 0.15)
            possible_shifts = []
            if img_size - 2 - x_max > 0:
                possible_shifts.append(random.randint(1, min(max_shift, img_size - 2 - x_max)))
            if x_min > 2:
                possible_shifts.append(random.randint(-min(max_shift, x_min - 2), -1))

            if not possible_shifts:
                continue
            shift = random.choice(possible_shifts)

            view_to_image["-Z"] = shift_image_numpy(view_to_image["-Z"], 0, shift)
            view_to_image["+Z"] = shift_image_numpy(view_to_image["+Z"], 0, shift)
            view_to_image["+Y"] = shift_image_numpy(view_to_image["+Y"], 0, shift)
            view_to_image["-Y"] = shift_image_numpy(view_to_image["-Y"], 0, shift)
        if axis == "Z":
            max_shift = round((z_max - z_min) * 0.15)
            possible_shifts = []
            if img_size - 2 - z_max > 0:
                possible_shifts.append(random.randint(1, min(max_shift, img_size - 2 - z_max)))
            if z_min > 2:
                possible_shifts.append(random.randint(-min(max_shift, z_min - 2), -1))

            if not possible_shifts:
                continue
            shift = random.choice(possible_shifts)

            view_to_image["+Y"] = shift_image_numpy(view_to_image["+Y"], -shift, 0)
            view_to_image["-Y"] = shift_image_numpy(view_to_image["-Y"], -shift, 0)
            view_to_image["+X"] = shift_image_numpy(view_to_image["+X"], -shift, 0)
            view_to_image["-X"] = shift_image_numpy(view_to_image["-X"], -shift, 0)
        if axis == "Y":
            max_shift = round((y_max - y_min) * 0.15)
            possible_shifts = []
            if img_size - 2 - y_max > 0:
                possible_shifts.append(random.randint(1, min(max_shift, img_size - 2 - y_max)))
            if y_min > 2:
                possible_shifts.append(random.randint(-min(max_shift, y_min - 2), -1))

            if not possible_shifts:
                continue
            shift = random.choice(possible_shifts)

            view_to_image["-Z"] = shift_image_numpy(view_to_image["-Z"], -shift, 0)
            view_to_image["+Z"] = shift_image_numpy(view_to_image["+Z"], -shift, 0)
            view_to_image["+X"] = shift_image_numpy(view_to_image["+X"], 0, -shift)
            view_to_image["-X"] = shift_image_numpy(view_to_image["-X"], 0, -shift)



def remove_through_hole_aug(view_to_image):
    holes = {}
    holes["X"] = find_holes(view_to_image["+X"])
    holes["Y"] = find_holes(view_to_image["+Y"])
    holes["Z"] = find_holes(view_to_image["+Z"])

    candidates = [axis for axis in holes if holes[axis]]

    if not candidates:
        return

    axis = random.choice(candidates)
    hole = random.choice(holes[axis])

    kernel = np.ones((7, 7), np.uint8)
    hole_area = (cv2.dilate(hole.astype(np.uint8), kernel, iterations=1) > 0.5)

    value = view_to_image[f"+{axis}"][hole_area].max()
    view_to_image[f"+{axis}"][hole_area] = value

    value = view_to_image[f"-{axis}"][hole_area].max()
    view_to_image[f"-{axis}"][hole_area] = value


def make_flat_aug(view_to_image):
    n_axis = random.choice((1, 2))
    axes = random.sample(list(view_to_image), k=n_axis)
    for axis in axes:
        image = view_to_image[axis]
        mask = image > 0
        func = random.choice((np.mean, np.max))
        value = func(image[mask])
        view_to_image[axis][mask] = value


def add_through_hole_aug(view_to_image):
    axis = random.choice(("X", "Y", "Z"))

    image = view_to_image[f"+{axis}"]
    bounds = get_bounds(image)
    r_max = min(bounds[0][1] - bounds[0][0], bounds[1][1] - bounds[1][0]) // 2
    if random.random() < 0.2:
        r_max = random.randint(2, 10)

    n_elements = random.randint(1, 2)

    for _ in range(n_elements):
        radius = random.randint(2, max(2, r_max - 2))
        for i in range(5):
            row = random.randint(bounds[0][0] + radius // 2, bounds[0][1] - radius // 2)
            col = random.randint(bounds[1][0] + radius // 2, bounds[1][1] - radius // 2)
            if image[row, col] != 0:
                break

        mask = np.zeros((image.shape[0], image.shape[1], 3))

        if random.random() < 0.5:
            cv2.circle(mask, (col, row), radius, (1, 0, 0), -1)
        else:
            a = random.randint(2, 2 * radius)
            b = random.randint(2, 2 * radius)
            if random.random() < 0.5:
                b, a = a, b
            cv2.rectangle(mask, (col - a // 2, row - a // 2), (col + a // 2, row + a // 2), (1, 0, 0), -1)

        mask = (mask[..., 0] == 1)
        view_to_image[f"+{axis}"][mask] = 0
        view_to_image[f"-{axis}"][mask] = 0


def add_fillet_or_protrusion_aug(view_to_image):
    axis = random.choice(("X", "Y", "Z"))

    image = view_to_image[f"+{axis}"]
    bounds = get_bounds(image)
    r_max = min(bounds[0][1] - bounds[0][0], bounds[1][1] - bounds[1][0]) // 2

    n_elements = random.randint(1, 2)

    for _ in range(n_elements):
        radius = random.randint(2, max(2, r_max - 2))
        for i in range(5):
            row = random.randint(bounds[0][0] + radius // 2, bounds[0][1] - radius // 2)
            col = random.randint(bounds[1][0] + radius // 2, bounds[1][1] - radius // 2)
            if image[row, col] != 0:
                break

        mask = np.zeros((image.shape[0], image.shape[1], 3))

        if random.random() < 0.5:
            cv2.circle(mask, (col, row), radius, (1, 0, 0), -1)
        else:
            a = random.randint(2, 2 * radius)
            b = random.randint(2, 2 * radius)
            if random.random() < 0.5:
                b, a = a, b
            cv2.rectangle(mask, (col - a // 2, row - a // 2), (col + a // 2, row + a // 2), (1, 0, 0), -1)

        mask = (mask[..., 0] == 1)
        value = random.randint(128, 255)
        if random.random() < 0.5:
            view_to_image[f"+{axis}"][mask] = value
        else:
            view_to_image[f"-{axis}"][mask] = value
