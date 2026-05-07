import numpy as np
import pyvista as pv
import cv2
import mediapipe as mp
import open3d as o3d
import copy
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ==========================================
# 1. 초기 데이터 설정 (landmark_picker.py를 통해 추출)
# ==========================================
# [코끝, 왼쪽 눈꼬리, 오른쪽 눈꼬리, 인중-윗입술]
GT_POINTS = np.array([
    [110.30, 2.96, 26.00],  # 코끝
    [62.68, 40.29, 54.75],  # 왼쪽 눈꼬리 (환자 기준 왼쪽)
    [159.14, 42.74, 47.05], # 오른쪽 눈꼬리 (환자 기준 오른쪽)
    [109.08, 9.36, 3.64]    # 인중
])

# MediaPipe Face Mesh 인덱스 (위의 GT 순서와 매칭)
MP_INDICES = [4, 33, 263, 0] 

# ==========================================
# 2. 오차 생성 및 가상 카메라 렌더링
# ==========================================
def render_virtual_camera(obj_path, rot_deg=(0, 0, 0), trans=(0, 0, 0)):
    print(f"\n[1/4] 가상 카메라 촬영 중... (오차 적용: 회전 {rot_deg}, 이동 {trans})")
    mesh = pv.read(obj_path)
    
    # 🌟 환자가 고개를 돌리는 상황 시뮬레이션 (인위적 오차)
    mesh.rotate_x(rot_deg[0], inplace=True)
    mesh.rotate_y(rot_deg[1], inplace=True)
    mesh.rotate_z(rot_deg[2], inplace=True)
    mesh.translate(trans, inplace=True)

    plotter = pv.Plotter(off_screen=True, window_size=[1000, 1000])
    plotter.add_mesh(mesh, color="#EAC0A6", smooth_shading=True)
    plotter.set_background("black")
    
    # 카메라를 모델 정면 약간 위쪽에 배치하여 얼굴 전체가 나오게 함
    # 중심점(Origin) 근처를 바라보도록 설정
    plotter.camera_position = [(110, -300, 50), (110, 0, 20), (0, 0, 1)]
    
    img = plotter.screenshot(None, return_img=True)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    cv2.imwrite("virtual_patient.png", img_bgr)
    return img_bgr


# ==========================================
# 3. MediaPipe 추론 및 3D 랜드마크 추출 (최신 Tasks API 버전)
# ==========================================
# 최신 버전에서 mediapipe.solutions를 지원하지 않아 mediapipe.tasks 사용

def extract_mediapipe_landmarks(img_bgr):
    print("[2/4] MediaPipe로 얼굴 특징점 추론 중... (최신 Tasks API 사용)")
    
    # 1. MediaPipe 전용 이미지 포맷으로 변환
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    
    # 2. 'face_landmarker.task' 파일 연동
    base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        num_faces=1
    )
    
    # 3. 모델 로드 및 추론 실행
    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        detection_result = landmarker.detect(mp_image)
        
    if not detection_result.face_landmarks:
        raise ValueError("얼굴을 찾을 수 없습니다. 카메라 각도를 조절하세요.")
        
    landmarks = detection_result.face_landmarks[0]
    
    # 4. 필요한 4개의 점만 추출 (MP_INDICES = [4, 33, 263, 0])
    mp_points = []
    for idx in MP_INDICES:
        lm = landmarks[idx]
        mp_points.append([lm.x, lm.z, -lm.y])
        
    return np.array(mp_points)

# ==========================================
# 4. 좌표 복원 (Scale Matching & mm 변환)
# ==========================================
def restore_coordinates(mp_points, gt_points):
    print("[3/4] MediaPipe 랜드마크를 실제 물리적 크기(mm)로 복원 중...")
    
    # 양쪽 눈꼬리 사이의 물리적 거리 계산 (GT 기준)
    # 인덱스 1: 왼쪽 눈꼬리, 인덱스 2: 오른쪽 눈꼬리
    gt_eye_dist = np.linalg.norm(gt_points[1] - gt_points[2])   # 3d 모델에서 추출한 실제 환자의 눈꼬리 사이 거리 (mm)
    mp_eye_dist = np.linalg.norm(mp_points[1] - mp_points[2])   # MediaPipe가 추론한 가상의 눈꼬리 사이 거리 (0~1)
    
    # [ Scale 맞추기 ] MediaPipe의 상대적인 좌표에 스케일을 곱해 실제 mm 단위로 증폭
    scale_factor = gt_eye_dist / mp_eye_dist    # ex) 실제 거리가 100mm인데 mp 거리가 0.5면 200이 됨
    restored_points = mp_points * scale_factor  # 모든 점 좌표에 scale_factor를 곱해 크기를 맞춰줌
    
    # [ 영점 조절 ] 중심점을 코끝(인덱스 0)으로 맞춰 정렬 준비 (코 끝을 (0,0,0)으로 맞춰줌)
    gt_centered = gt_points - gt_points[0]
    restored_centered = restored_points - restored_points[0]
    
    return restored_centered, gt_centered

# ==========================================
# 5. 랜드마크 기반 정합 (SVD) 및 오차 계산 <-- 로직 수정 필요 (오차 최소 2mm 이내로)
# ==========================================
def calculate_registration_error(source_pts, target_pts):
    print("[4/4] 랜드마크 기반 정합(Registration) 수행 및 오차 계산 중...")
    
    # Numpy 점 데이터 -> Open3D의 PointCloud 객체 변환
    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(source_pts)
    
    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(target_pts)
    
    # 4개의 점이 순서대로 1:1 대응된다는 것을 엔진에 명시합니다.
    # 형태: [[source_idx, target_idx], ...]
    corres = o3d.utility.Vector2iVector(np.array([
        [0, 0], # 코끝
        [1, 1], # 왼쪽 눈꼬리
        [2, 2], # 오른쪽 눈꼬리
        [3, 3]  # 인중
    ]))
    
    # 대응 정보(corres)를 넣어서 변환 행렬(T)을 한 번에 계산
    # Iteration을 거쳐 계산 X (ICP와 다름)
    # SVD(Singular Value Decomposition) 기법 사용하여 4x4 변환 행렬 반환
    estimation = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    transformation = estimation.compute_transformation(source_pcd, target_pcd, corres)
    
    # 계산된 변환 행렬을 적용하여 소스 데이터를 타겟에 맞춤
    source_pcd.transform(transformation)
    
    # 남은 오차(RMSE) 계산
    distances = np.linalg.norm(np.asarray(source_pcd.points) - np.asarray(target_pcd.points), axis=1)
    rmse = np.sqrt(np.mean(distances**2))
    
    print("\n" + "="*50)
    print("✅ 정합 검증 완료!")
    print(f"-> 계산된 최종 오차(RMSE): {rmse:.4f} mm")
    if rmse < 5.0:
        print("-> 훌륭합니다! 실제 수술 네비게이션으로도 활용 가능한 수준의 오차율입니다.")
    else:
        print("-> 오차가 다소 큽니다. 가상 카메라의 투영 왜곡이나 MediaPipe 인식 오차일 수 있습니다.")
    print("="*50)
    
    return transformation, rmse

def visualize_registration_result(obj_path, gt_points, mp_restored_centered, transformation):
    print("\n[5/5] Open3D 시각화 창을 띄웁니다...")
    
    # 1. 원본 3D 피부 메쉬 로드
    mesh = o3d.io.read_triangle_mesh(obj_path)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.8, 0.7, 0.6]) # 시각화를 위한 스킨톤 적용
    
    # 2. GT 포인트 생성 (빨간색 구)
    # GT_POINTS는 원점 이동 전의 절대 좌표를 그대로 사용합니다.
    gt_geometries = []
    for pt in gt_points:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=2.0)
        sphere.translate(pt)
        sphere.paint_uniform_color([1.0, 0.0, 0.0]) # Red
        gt_geometries.append(sphere)
        
    # 3. MediaPipe 정합 포인트 생성 (파란색 구)
    # 중심이 맞춰진 moving_pts(mp_restored_centered)에 Transformation 적용
    mp_pcd = o3d.geometry.PointCloud()
    mp_pcd.points = o3d.utility.Vector3dVector(mp_restored_centered)
    mp_pcd.transform(transformation)
    
    # 변환된 포인트들을 다시 원본 메쉬의 위치(코끝 절대 좌표)로 이동시켜 원상 복구
    mp_pcd.translate(gt_points[0])
    
    mp_geometries = []
    for pt in np.asarray(mp_pcd.points):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=2.0)
        sphere.translate(pt)
        sphere.paint_uniform_color([0.0, 0.0, 1.0]) # Blue
        mp_geometries.append(sphere)
        
    # 4. 렌더링
    print("🔴 빨간색 구: Ground Truth (실제 3D 모델의 랜드마크)")
    print("🔵 파란색 구: MediaPipe 추론 후 ICP 정합된 랜드마크")
    
    o3d.visualization.draw_geometries(
        [mesh] + gt_geometries + mp_geometries,
        window_name="XR Markerless Registration Result",
        width=1280, height=720
    )


def visualize_before_and_after(obj_path, gt_points, moving_pts, fixed_pts, test_rot, test_trans, transform_matrix):
    print("\n[5/5] 좌우 분할 및 랜드마크 포인트 시각화 창을 띄웁니다...")
    
    OFFSET_X = 250.0  
    
    # 🌟 헬퍼 함수: 좌표 리스트를 받아 Open3D 구(Sphere) 객체 리스트로 변환
    def create_spheres(points, color, offset=(0,0,0), radius=2.5):
        spheres = []
        for pt in points:
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
            sphere.translate(pt + np.array(offset))
            sphere.paint_uniform_color(color)
            spheres.append(sphere)
        return spheres

    # ==========================================
    # ⬅️ [왼쪽] 정합 이전 (Pre-registration)
    # ==========================================
    target_left = o3d.io.read_triangle_mesh(obj_path)
    target_left.compute_vertex_normals()
    target_left.paint_uniform_color([0.7, 0.7, 0.7]) 
    target_left.translate(-gt_points[0]) 
    
    source_left = o3d.io.read_triangle_mesh(obj_path)
    R = source_left.get_rotation_matrix_from_xyz((
        np.radians(test_rot[0]), 
        np.radians(test_rot[1]), 
        np.radians(test_rot[2])
    ))
    source_left.rotate(R, center=(0, 0, 0))
    source_left.translate(test_trans)
    patient_nose_tip = np.dot(R, gt_points[0]) + np.array(test_trans)
    source_left.translate(-patient_nose_tip)
    source_left.compute_vertex_normals()
    source_left.paint_uniform_color([1.0, 0.4, 0.4]) 
    
    # [Point 추가] 왼쪽 메쉬용 포인트 (GT와 MediaPipe 원본)
    gt_spheres_left = create_spheres(fixed_pts, color=[1.0, 0.0, 0.0]) # 🔴 빨간색: GT
    mp_spheres_left = create_spheres(moving_pts, color=[0.0, 0.0, 1.0]) # 🔵 파란색: MediaPipe

    # ==========================================
    # ➡️ [오른쪽] 정합 이후 (Post-registration)
    # ==========================================
    target_right = copy.deepcopy(target_left)
    target_right.translate((OFFSET_X, 0, 0))
    
    source_right = copy.deepcopy(source_left)
    source_right.transform(transform_matrix) 
    source_right.translate((OFFSET_X, 0, 0))
    source_right.paint_uniform_color([0.2, 0.6, 1.0]) 
    
    # [Point 추가] 오른쪽 메쉬용 포인트 (MediaPipe 포인트에도 Transform 행렬 적용!)
    moving_pcd = o3d.geometry.PointCloud()
    moving_pcd.points = o3d.utility.Vector3dVector(moving_pts)
    moving_pcd.transform(transform_matrix) # 파란 점들도 메쉬와 함께 이동시킴
    transformed_moving_pts = np.asarray(moving_pcd.points)
    
    gt_spheres_right = create_spheres(fixed_pts, color=[1.0, 0.0, 0.0], offset=(OFFSET_X, 0, 0))
    mp_spheres_right = create_spheres(transformed_moving_pts, color=[0.0, 0.0, 1.0], offset=(OFFSET_X, 0, 0))

    # ==========================================
    # 화면 렌더링
    # ==========================================
    print("\n" + "="*50)
    print("🔴 빨간색 점: Ground Truth (실제 목표 위치)")
    print("🔵 파란색 점: MediaPipe 특징점 (Source)")
    print("💡 정합 후(오른쪽)에서 빨간 점과 파란 점이 가까울수록 정합 오차가 적은 것입니다.")
    print("="*50)
    
    # 렌더링 리스트에 메쉬 4개 + 구(Spheres) 모두 추가
    geometries = [target_left, source_left, target_right, source_right]
    geometries.extend(gt_spheres_left + mp_spheres_left + gt_spheres_right + mp_spheres_right)
    
    o3d.visualization.draw_geometries(
        geometries,
        window_name="XR Markerless Registration: Mesh & Points",
        width=1600, height=800
    )

# ==========================================
# 실행부
# ==========================================
if __name__ == "__main__":
    # 🎯 지영님의 3D 피부 모델 파일명 (절대좌표 모델)
    SKIN_OBJ_PATH = "data/neurosync_skin_abs.obj"
    
    # 테스트용 인위적 오차 (예: 카메라 앞에서 환자가 고개를 X축 15도 숙임)
    TEST_ROTATION = (15, 0, 0)
    TEST_TRANSLATION = (0, 0, 0)
    
    try:
        # 1. 렌더링
        rendered_image = render_virtual_camera(SKIN_OBJ_PATH, TEST_ROTATION, TEST_TRANSLATION)
        
        # 2. 특징점 추론
        raw_mp_points = extract_mediapipe_landmarks(rendered_image)
        
        # 3. 스케일 복원
        moving_pts, fixed_pts = restore_coordinates(raw_mp_points, GT_POINTS)
        
        # 4. 정합 및 오차 계산
        transform_matrix, final_error = calculate_registration_error(moving_pts, fixed_pts)
        
        # 5. 3D 결과 시각화
        visualize_before_and_after(
            SKIN_OBJ_PATH, 
            GT_POINTS, 
            moving_pts,
            fixed_pts,
            TEST_ROTATION, 
            TEST_TRANSLATION, 
            transform_matrix
        )
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        print("경로나 모델 파일 상태를 확인해주세요. 카메라 뷰를 조정해야 할 수도 있습니다.")
