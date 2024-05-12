import cv2
import numpy as np

def resize_and_pad(img, size, pad_color=0):
    h, w = img.shape[:2]
    target_h, target_w = size

    aspect_ratio_original = w / h
    aspect_ratio_target = target_w / target_h

    if aspect_ratio_original > aspect_ratio_target:
        new_w = target_w
        new_h = int(new_w / aspect_ratio_original)
    else:
        new_h = target_h
        new_w = int(new_h * aspect_ratio_original)

    new_w = min(new_w, target_w)
    new_h = min(new_h, target_h)

    pad_left = (target_w - new_w) // 2
    pad_right = target_w - new_w - pad_left
    pad_top = (target_h - new_h) // 2
    pad_bot = target_h - new_h - pad_top

    img_resized = cv2.resize(img, (new_w, new_h))

    img_padded = cv2.copyMakeBorder(img_resized, pad_top, pad_bot, pad_left, pad_right,
                                    cv2.BORDER_CONSTANT, value=pad_color)

    return img_padded


def find_homography_and_transform(smaller_img_path, larger_img_path, output_path):
    img1 = cv2.imread(smaller_img_path, cv2.IMREAD_COLOR)
    img2 = cv2.imread(larger_img_path, cv2.IMREAD_COLOR)
    if img1 is None or img2 is None:
        print("Error loading one or both images.")
        return None

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Old but reliable
    sift = cv2.SIFT_create()

    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good_matches = []
    for m,n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    if isinstance(good_matches, tuple):
        good_matches = list(good_matches)

    good_matches.sort(key=lambda x: x.distance)

    points1 = np.zeros((len(good_matches), 2), dtype=np.float32)
    points2 = np.zeros((len(good_matches), 2), dtype=np.float32)

    for i, match in enumerate(good_matches):
        points1[i, :] = kp1[match.queryIdx].pt
        points2[i, :] = kp2[match.trainIdx].pt

    H, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    height, width, channels = img2.shape
    img1_transformed = cv2.warpPerspective(img1, H, (width, height))

    cv2.imwrite(output_path, img1_transformed)

    return output_path

if __name__ == "__main__":
    output_transformed_path = find_homography_and_transform(
        'data/kont_after.jpg', 
        'data/kont_before.jpg', 
        'data/kont_transformed.jpg'
        )