import numpy as np
import pyvista as pv
import SimpleITK as sitk
import os

# ==========================================
# 1. 데이터 로드 함수
# ==========================================
def load_dicom_volume(folder_path):
    print(f"-> [DICOM] 로드 중: {folder_path}")
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(folder_path)
    dicom_names = reader.GetGDCMSeriesFileNames(folder_path, series_ids[0])
    reader.SetFileNames(dicom_names)
    
    image = reader.Execute()
    vol_array = np.transpose(sitk.GetArrayFromImage(image), (2, 1, 0))
    return vol_array, image.GetSpacing()

def load_nifti_volume(file_path):
    print(f"-> [NIfTI] 로드 중: {file_path}")
    image = sitk.ReadImage(file_path)
    vol_array = np.transpose(sitk.GetArrayFromImage(image), (2, 1, 0))
    return vol_array, image.GetSpacing()

# ==========================================
# 2. 메쉬 추출 함수
# ==========================================
def create_mesh(volume_array, spacing, threshold, smooth_iter, pass_band, largest_only=True):
    """
    3D 볼륨 배열 데이터(예: CT, MRI)로부터 임계값(Threshold)을 기준으로 
    표면 메쉬(Isosurface Mesh)를 추출하고 후처리하는 함수입니다.
    """
    
    # 1. 3D 격자(Grid) 구조 생성
    # PyVista에서 3D 볼륨 데이터를 다루기 위한 빈 컨테이너(ImageData)를 만듭니다.
    grid = pv.ImageData()
    
    # 2. 볼륨 데이터의 차원(Shape) 설정
    # (x, y, z) 형태의 격자 크기를 정의합니다.
    grid.dimensions = volume_array.shape
    
    # 3. 물리적 스케일(Spacing) 적용 (px -> mm)
    # 의료 영상의 Voxel이 갖는 실제 물리적 간격(mm 단위)을 설정하여 
    # 메쉬가 현실의 물리적 크기(Absolute Scale)를 갖도록 보정합니다.
    grid.spacing = spacing
    
    # 4. 볼륨 데이터 매핑
    # 3D 넘파이 배열을 1차원으로 쫙 펴서(flatten) grid의 각 포인트에 값으로 할당합니다.
    # order="F" (Fortran 순서)를 사용하는 이유는 PyVista(VTK 기반)가 메모리를 읽는 순서와 
    # NumPy 배열의 x, y, z 축 인덱싱 방향을 올바르게 맞추기 위함입니다.
    grid.point_data["values"] = volume_array.flatten(order="F")

    # 5. 등위면(Isosurface) 추출 (핵심 단계)
    # 특정 임계값(예: 뼈를 추출할 때는 350 HU 이상, 피부는 -500 HU 등)을 만족하는 
    # 경계면을 따라 다각형(Polygon)을 생성합니다. (내부적으로 Marching Cubes 알고리즘 사용)
    mesh = grid.contour([threshold])
    
    # 6. 메인 컴포넌트(가장 큰 덩어리) 추출
    # largest_only가 True일 경우, 촬영 노이즈나 파편화되어 허공에 떠 있는 
    # 작은 메쉬 조각들을 전부 날려버리고 가장 부피가 큰 뼈대/피부 본체 하나만 남깁니다.
    if largest_only:
        mesh = mesh.connectivity(largest=True)
        
    # 7. 표면 스무딩 (Taubin Smoothing)
    # 픽셀 단위로 추출되어 계단 현상(Aliasing)이 있는 메쉬 표면을 부드럽게 다듬어 줍니다.
    # Taubin 알고리즘은 부피의 손실(수축 현상)을 최소화하면서 표면만 매끄럽게 펴주는 데 유리합니다.
    # n_iter는 반복 횟수, pass_band는 값이 작을수록 더 강력하게 주름을 펴줍니다.
    mesh = mesh.smooth_taubin(n_iter=smooth_iter, pass_band=pass_band)
    
    return mesh

# ==========================================
# 3. 시각화 함수
# ==========================================
def visualize_4way(skull, skin, tumor):
    print("\n🌟 4분할 3D 렌더링 창을 띄웁니다...")
    plotter = pv.Plotter(shape=(2, 2), window_size=[1200, 900])
    plotter.set_background("#222222")

    plotter.subplot(0, 0)
    plotter.add_text("1. Skull (Bone)", font_size=12, color="ivory")
    plotter.add_mesh(skull, color="ivory", specular=0.5)

    plotter.subplot(0, 1)
    plotter.add_text("2. Skin (Face)", font_size=12, color="#EAC0A6")
    plotter.add_mesh(skin, color="#EAC0A6", smooth_shading=True, specular=0.2)

    plotter.subplot(1, 0)
    plotter.add_text("3. Tumor (Target)", font_size=12, color="red")
    plotter.add_mesh(tumor, color="red", opacity=0.9)

    plotter.subplot(1, 1)
    # 타이틀을 Absolute로 수정
    plotter.add_text("4. Combined View (Absolute)", font_size=12, color="white")
    plotter.add_mesh(skin, color="#EAC0A6", opacity=0.2, smooth_shading=True) 
    plotter.add_mesh(skull, color="ivory", opacity=0.4, specular=0.5)        
    plotter.add_mesh(tumor, color="red", opacity=1.0)                        

    plotter.link_views()
    plotter.show()

# ==========================================
# 메인 파이프라인 실행부
# ==========================================
if __name__ == "__main__":
    # 🎯 데이터 경로 설정
    DICOM_DIR = r"data\tumor"
    NIFTI_PATH = r"data\Segmentation.nii"
    
    print("=== NeuroSync 3D Pipeline (절대 좌표 유지) 시작 ===")
    
    # 1. 데이터 로드
    dicom_vol, dicom_spacing = load_dicom_volume(DICOM_DIR)
    nifti_vol, nifti_spacing = load_nifti_volume(NIFTI_PATH)

    # 2. 메쉬 추출
    print("\n[1/3] 두개골 메쉬 생성 중...")
    skull_mesh = create_mesh(dicom_vol, dicom_spacing, threshold=350, smooth_iter=100, pass_band=0.05, largest_only=True)
    
    print("[2/3] 피부 메쉬 생성 중...")
    # skin의 경우 스무딩을 하게되면 landmark로 활용될 수 있는 눈꼬리 등이 뭉개져보이기 때문에 smooth_iter를 0으로 주었습니다.
    # 해당 landmark를 사용하지 않을경우 smooth_iter를 50으로 설정합니다.
    skin_mesh = create_mesh(dicom_vol, dicom_spacing, threshold=-500, smooth_iter=0, pass_band=0.005, largest_only=True)
    
    print("[3/3] 종양 메쉬 생성 중...")
    tumor_mesh = create_mesh(nifti_vol, nifti_spacing, threshold=0.5, smooth_iter=50, pass_band=0.01, largest_only=False)


    # 3. OBJ 파일로 각각 저장
    print("\n-> 3D 모델(OBJ) 저장을 시작합니다...")
    base_dir = os.path.dirname(NIFTI_PATH)
    
    skull_mesh.save(os.path.join(base_dir, "skull_abs.obj"))
    skin_mesh.save(os.path.join(base_dir, "skin_abs.obj"))
    tumor_mesh.save(os.path.join(base_dir, "tumor_abs.obj"))
    print("✅ OBJ 파일 3종 저장 완료!")

    # 4. 사용자 정의 함수로 결과 확인
    visualize_4way(skull_mesh, skin_mesh, tumor_mesh)