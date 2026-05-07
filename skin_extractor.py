import numpy as np
import pyvista as pv
import SimpleITK as sitk
import os

def load_dicom_volume(folder_path):
    """DICOM 폴더를 읽어 3D 볼륨과 물리적 간격(Spacing)을 반환합니다."""
    print(f"[{folder_path}] 경로에서 데이터를 로드합니다...")
    
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
    print(f"-> 로드 완료! (슬라이스: {volume_array.shape[2]}장, Spacing: {spacing})")
    return volume_array, spacing

def extract_skin_mesh(volume_array, spacing, threshold=-300, output_filename="skin_model.obj"):
    """
    임계값(기본 -300)을 사용하여 피부 표면을 면(Mesh)으로 추출하고 OBJ 파일로 저장합니다.
    """
    print(f"\n임계값 {threshold} 기준으로 피부(Skin) 면(Mesh)을 추출합니다...")
    grid = pv.ImageData()
    grid.dimensions = volume_array.shape
    grid.spacing = spacing 
    grid.point_data["values"] = volume_array.flatten(order="F")

    # 1. 1차 추출 (침대, 노이즈 등이 포함된 상태)
    mesh = grid.contour([threshold])
    print(f"-> 1차 추출 완료! (면: {mesh.n_cells}개)")
    
    # 2. 노이즈 제거 (가장 큰 덩어리 하나만 남김)
    print("-> 허공의 노이즈와 CT 침대 등을 제거하는 중...")
    mesh = mesh.connectivity(largest=True)
    
    # ==========================================
    # 🌟 3. 표면 스무딩 (Taubin Smoothing) 추가
    # ==========================================
    print("-> 피부 표면을 매끄럽게 다듬는 중 (Taubin Smoothing)...")
    # n_iter를 150~200 정도로 주면 피부가 훨씬 도자기처럼 매끄러워집니다.
    #mesh = mesh.smooth_taubin(n_iter=50, pass_band=0.005)
    
    print(f"-> 최종 정제 완료! (면: {mesh.n_cells}개)")
    
    # 4. OBJ 파일 저장
    mesh.save(output_filename)
    save_path = os.path.abspath(output_filename)
    print(f"-> XR 매칭용 3D 모델 저장 완료: \n   [{save_path}]")
    
    return mesh

def visualize_skin(mesh):
    """추출된 피부 메쉬를 화면에 부드럽게 렌더링합니다."""
    print("\n피부 3D 렌더링 창을 띄웁니다. (종료하려면 창을 닫으세요)")
    plotter = pv.Plotter(window_size=[800, 800])
    
    # 🌟 살구색 피부 톤 적용 및 부드러운 음영 처리
    plotter.add_mesh(
        mesh, 
        color="#EAC0A6",          # 피부색과 유사한 헥스 코드
        smooth_shading=True,      # 폴리곤을 매끄럽게
        specular=0.2,             # 뼈에 비해 빛 반사율을 낮춤
        opacity=1.0
    )
    
    plotter.add_text("3D Skin Mesh (For Camera Landmark Matching)", font_size=12)
    plotter.set_background("#222222")
    plotter.show()

if __name__ == "__main__":
    # ==========================================
    # 환경 설정
    # ==========================================
    DICOM_FOLDER = r"data\tumor"
    
    # 💡 피부 추출용 임계값 (얼굴 형태가 찌그러지면 -400 ~ -150 사이에서 조절)
    SKIN_THRESHOLD = -500  
    
    # 저장될 파일 이름
    OUTPUT_FILE = "patient_face.obj" 
    # ==========================================

    # 실행 파이프라인
    vol_array, vol_spacing = load_dicom_volume(DICOM_FOLDER)
    skin_mesh = extract_skin_mesh(
        vol_array, 
        vol_spacing, 
        threshold=SKIN_THRESHOLD, 
        output_filename=OUTPUT_FILE
    )
    visualize_skin(skin_mesh)