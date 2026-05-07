import numpy as np
import pyvista as pv
import SimpleITK as sitk
import os

def load_dicom_volume(folder_path):
    print(f"[{folder_path}] 경로에서 DICOM 시리즈를 조립합니다...")
    
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(folder_path)
    if not series_ids:
        raise ValueError("DICOM 데이터를 찾을 수 없습니다.")
        
    dicom_names = reader.GetGDCMSeriesFileNames(folder_path, series_ids[0])
    reader.SetFileNames(dicom_names)
    
    image = reader.Execute()
    volume_array = sitk.GetArrayFromImage(image)
    
    spacing = image.GetSpacing() 
    volume_array = np.transpose(volume_array, (2, 1, 0))
    
    print(f"-> 로드 완료! 가져온 슬라이스(장) 수: {volume_array.shape[2]}장")
    print(f"-> 물리적 간격(Spacing): {spacing}")
    return volume_array, spacing

def extract_and_save_skull(volume_array, spacing, threshold=500, output_filename="refined_skull.obj"):
    """
    임계값을 적용해 뼈를 추출하고, 표면을 매끄럽게 다듬어 OBJ로 저장합니다.
    """
    print(f"\n[모델 생성] 임계값 {threshold}으로 두개골 메쉬 생성 중...")
    
    grid = pv.ImageData()
    grid.dimensions = volume_array.shape
    grid.spacing = spacing
    grid.point_data["values"] = volume_array.flatten(order="F")

    full_surface = grid.contour([threshold])
    
    print("-> 노이즈 제거 및 최대 연결 요소 추출 중...")
    skull_mesh = full_surface.connectivity(largest=True)
    
    # ==========================================
    # 🌟 표면 스무딩 (Taubin Smoothing) 추가
    # ==========================================
    print("-> 표면을 매끄럽게 다듬는 중 (Taubin Smoothing)...")
    # n_iter: 반복 횟수 (100~200 정도가 적당하며, 높을수록 더 부드러워짐)
    # pass_band: 모양을 유지하는 강도 (보통 0.01 ~ 0.1 사이 사용)
    skull_mesh = skull_mesh.smooth_taubin(n_iter=100, pass_band=0.05)
    
    # 🌟 파일 저장 (OBJ 형식)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(base_dir, output_filename)
    skull_mesh.save(output_path)
    
    print(f"✅ 저장 완료: {output_path}")
    print(f"-> 추출된 정점(Vertex) 개수: {skull_mesh.points.shape[0]}개")
    
    return output_path

def visualize_mesh(obj_path):
    """저장된 OBJ 파일을 화면에 렌더링합니다."""
    print("\n3D 렌더링 창을 띄웁니다. (종료하려면 창을 닫으세요)")
    mesh = pv.read(obj_path)
    
    plotter = pv.Plotter()
    # 🌟 "bone" 대신 PyVista가 인식하는 "ivory" 또는 "#FFFFF0" 사용
    plotter.add_mesh(mesh, color="ivory", opacity=1, specular=0.5)
    plotter.set_background("#222222")
    plotter.show()

if __name__ == "__main__":
    # 🎯 DICOM 폴더 경로
    DICOM_FOLDER = r"data\tumor"
    
    # 뼈 임계값 (CT 기준 보통 200~500, 필요시 조절)
    BONE_THRESHOLD = 350
    OUTPUT_OBJ_NAME = "refined_skull.obj"

    # 1. DICOM 로드 및 뼈 추출
    vol_array, vol_spacing = load_dicom_volume(DICOM_FOLDER)
    saved_obj_path = extract_and_save_skull(vol_array, vol_spacing, 
                                            threshold=BONE_THRESHOLD, 
                                            output_filename=OUTPUT_OBJ_NAME)
    
    # 2. 저장된 결과 눈으로 확인하기
    visualize_mesh(saved_obj_path)