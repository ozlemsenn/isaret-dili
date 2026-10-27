"""Videodan / kameradan el ve vücut noktalarını çıkarma (görüntü işleme kısmı).

Her kare 201 sayılık bir vektöre dönüşür:
  0-74    vücut noktaları 0-24 (belden yukarısı), her biri x, y, z
  75-137  sol el, 21 nokta
  138-200 sağ el, 21 nokta
Bulunamayan parçanın yeri 0 ile doldurulur.
"""
import os

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

N_POSE = 25                       # vücut noktaları 0-24
N_FEAT = (N_POSE + 21 + 21) * 3   # = 201


def make_landmarkers(hand_model="hand_landmarker.task", pose_model="pose_landmarker_full.task"):
    """El ve vücut nokta bulucularını VIDEO modunda oluşturur."""
    hand_opts = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=hand_model),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5)
    pose_opts = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=pose_model),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1)
    return (vision.HandLandmarker.create_from_options(hand_opts),
            vision.PoseLandmarker.create_from_options(pose_opts))


def to_vector(pose_res, hand_res):
    """Bir karenin MediaPipe sonuçlarını 201 sayılık vektöre çevirir."""
    pose = np.zeros((N_POSE, 3), np.float32)
    left = np.zeros((21, 3), np.float32)
    right = np.zeros((21, 3), np.float32)
    has_pose = len(pose_res.pose_landmarks) > 0
    if has_pose:
        lms = pose_res.pose_landmarks[0][:N_POSE]
        pose = np.array([[p.x, p.y, p.z] for p in lms], np.float32)
    for hand, handed in zip(hand_res.hand_landmarks, hand_res.handedness):
        pts = np.array([[p.x, p.y, p.z] for p in hand], np.float32)
        if has_pose:
            # Elin bileği, vücudun sol (15) / sağ (16) bileğinden hangisine yakınsa o eldir.
            d_left = np.linalg.norm(pts[0, :2] - pose[15, :2])
            d_right = np.linalg.norm(pts[0, :2] - pose[16, :2])
            is_left = d_left < d_right
        else:
            is_left = handed[0].category_name == "Left"
        if is_left:
            left = pts
        else:
            right = pts
    return np.concatenate([pose, left, right]).reshape(-1)


def to_mp_image(frame_bgr):
    """OpenCV karesini (BGR) MediaPipe görüntüsüne (RGB) çevirir."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)


def extract_video(path, hand_model="hand_landmarker.task", pose_model="pose_landmarker_full.task"):
    """Bir videoyu (kare_sayısı, 201) boyutunda diziye çevirir."""
    hand_lm, pose_lm = make_landmarkers(hand_model, pose_model)   # her video için yeni: zaman damgası 0'dan başlar
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frames, i = [], 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            img = to_mp_image(frame)
            ts = int(i * 1000 / fps)              # milisaniye; her karede artmak zorunda
            frames.append(to_vector(pose_lm.detect_for_video(img, ts),
                                    hand_lm.detect_for_video(img, ts)))
            i += 1
    finally:
        cap.release()
        hand_lm.close()
        pose_lm.close()
    return np.array(frames, np.float32).reshape(-1, N_FEAT)


def process_one(sample, path, out_dir):
    """Tek videoyu işleyip .npy kaydeder. Dosya zaten varsa atlar (kaldığı yerden devam)."""
    out_path = os.path.join(out_dir, sample + ".npy")
    if os.path.exists(out_path):
        return sample, "atlandı"
    try:
        seq = extract_video(path)
        np.save(out_path, seq.astype(np.float16))
        return sample, len(seq)
    except Exception as e:
        return sample, "HATA: " + str(e)
