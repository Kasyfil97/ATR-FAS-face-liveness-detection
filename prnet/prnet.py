import cv2
import numpy as np
from .api import PRN
from .utils.render_app import get_visibility, get_uv_mask, get_depth_image

class PRNFeatures:
    def __init__(self):
        self.prn = PRN()
    
    def get_landmark(self, image):
        pos = self.prn.process(image)
        landmark = self.prn.get_landmarks(pos)
        return landmark
    
    def face_crop(self, image):
        box = self.prn.face_detect(image)
        if not box:
            return
        left = box[0]['box'][0]
        top = box[0]['box'][1]
        right = box[0]['box'][0] + box[0]['box'][2]
        bottom = box[0]['box'][1] + box[0]['box'][3]
        img_crop = image[top:bottom, left:right].copy()
        return img_crop
    
    @staticmethod
    def align_face(image, landmarks, desired_size):
        """
        Melakukan alignment wajah berdasarkan landmarks yang sudah ada
        
        Parameters:
        - image: numpy array gambar BGR
        - landmarks: numpy array dengan shape (68, 3) berisi koordinat x,y,z tiap landmark
        - desired_size: tuple ukuran output yang diinginkan
        
        Returns:
        - aligned_face: gambar yang sudah di-align
        """
        # Ambil koordinat x,y dari landmarks (abaikan z)
        landmarks_2d = landmarks[:, :2].astype(np.float32)
        
        # Ambil koordinat mata kiri dan kanan
        left_eye = landmarks_2d[36:42]
        right_eye = landmarks_2d[42:48]
        
        # Hitung titik tengah mata
        left_eye_center = left_eye.mean(axis=0)
        right_eye_center = right_eye.mean(axis=0)
        
        # Hitung sudut antara mata
        dy = right_eye_center[1] - left_eye_center[1]
        dx = right_eye_center[0] - left_eye_center[0]
        angle = np.degrees(np.arctan2(dy, dx))
        
        # Hitung titik tengah antara kedua mata
        eye_center = (
            int((left_eye_center[0] + right_eye_center[0]) / 2),
            int((left_eye_center[1] + right_eye_center[1]) / 2)
        )
        
        # Hitung skala berdasarkan jarak antar mata
        desired_eye_distance = desired_size[0] * 0.3
        eye_distance = np.sqrt((dx ** 2) + (dy ** 2))
        scale = desired_eye_distance / eye_distance
        
        # Buat matrix rotasi
        rotation_matrix = cv2.getRotationMatrix2D(eye_center, angle, scale)
        
        # Sesuaikan translation untuk memastikan wajah di tengah
        tx = desired_size[0] * 0.5
        ty = desired_size[1] * 0.35
        rotation_matrix[0, 2] += (tx - eye_center[0])
        rotation_matrix[1, 2] += (ty - eye_center[1])
        
        # Aplikasikan transformasi
        aligned_face = cv2.warpAffine(image, rotation_matrix, desired_size,
                                    flags=cv2.INTER_CUBIC)
        
        return aligned_face

    def face_alignment(self, image, desired_size=(256, 256)):
        landmarks = self.get_landmark(image)
        aligned_face = self.align_face(image, landmarks, desired_size=desired_size)
        return aligned_face
    
    def get_depth_map(self, image, shape=450):
        pos = self.prn.process(image)
        vertices = self.prn.get_vertices(pos)
        depth = get_depth_image(vertices, self.prn.triangles, shape, shape)
        return depth