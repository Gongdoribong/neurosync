import numpy as np
import pyvista as pv
import SimpleITK as sitk
import os

def load_nifti_volume(file_path):
    """NIfTI 파일을 읽어 3D 볼륨과 물리적 간격(Spacing)을 반환합니다."""
    print(f"\n[{file_path}] 데이터를 로드합니다...")
    
    if not os.path.exists(file_path):
        print(f"오류: 파일을 찾을 수 없습니다 -> {file_path}")
        return None, None

    image = sitk.ReadImage(file_path)
    volume_array = sitk.GetArrayFromImage(image)
    spacing = image.GetSpacing() 
    
    volume_array = np.transpose(volume_array, (2, 1, 0))
    print(f"-> 로드 완료! (볼륨 크기: {volume_array.shape}, Spacing: {spacing})")
    return volume_array, spacing

def extract_skin_mesh(volume_array, spacing, threshold=-150, output_filename="skin_model.obj"):
    """
    임계값을 사용하여 피부 표면을 면(Mesh)으로 추출하고 OBJ 파일로 저장합니다.
    """
    print(f"임계값 {threshold} 기준으로 피부(Skin) 메쉬를 추출합니다...")
    grid = pv.ImageData()
    grid.dimensions = volume_array.shape
    grid.spacing = spacing 
    grid.point_data["values"] = volume_array.flatten(order="F")

    # 1. 1차 추출 (침대, 노이즈 등이 포함된 상태)
    mesh = grid.contour([threshold])
    if mesh.n_cells == 0:
        print("-> 경고: 추출된 메쉬가 없습니다. 임계값을 확인해주세요.")
        return None
    print(f"-> 1차 추출 완료! (면: {mesh.n_cells}개)")
    
    # 2. 노이즈 제거 (가장 큰 덩어리 하나만 남김)
    print("-> 허공의 노이즈와 CT 침대 등을 제거하는 중...")
    mesh = mesh.connectivity(largest=True)
    
    # 3. 표면 스무딩 (Taubin Smoothing)
    # 카메라 랜드마크 매칭을 위한 특징점 추출 시, 표면이 부드러울수록 정합 안정성이 높아집니다.
    print("-> 피부 표면을 매끄럽게 다듬는 중 (Taubin Smoothing)...")
    mesh = mesh.smooth_taubin(n_iter=50, pass_band=0.005)
    
    print(f"-> 최종 정제 완료! (면: {mesh.n_cells}개)")
    
    # 4. OBJ 파일 저장
    mesh.save(output_filename)
    save_path = os.path.abspath(output_filename)
    print(f"-> 3D 모델 저장 완료: [{save_path}]")
    
    return mesh

if __name__ == "__main__":
    # ==========================================
    # 환경 설정
    # ==========================================
    DATA_DIR = os.path.join("Control_Subjects", "T1-Weighted MRI")
    
    # 피부 추출용 임계값
    # 주의: MRI는 CT(HU)와 달리 절대적인 밝기 기준이 없으므로 데이터에 따라 조정이 필요할 수 있습니다.
    SKIN_THRESHOLD = 20
    # ==========================================

    # subj1 ~ subj4 까지 반복 처리
    for i in range(1, 5):
        subject_name = f"subj{i}"
        nifti_file_path = os.path.join(DATA_DIR, f"{subject_name}.nii")
        output_file_path = os.path.join(DATA_DIR, f"{subject_name}_face.obj") 
        
        print(f"{'='*50}")
        print(f"진행 중: {subject_name}")
        print(f"{'='*50}")

        vol_array, vol_spacing = load_nifti_volume(nifti_file_path)
        
        # 파일이 정상적으로 로드된 경우에만 추출 진행
        if vol_array is not None:
            print(f"[{subject_name}] 값 범위 -> Min: {vol_array.min()}, Max: {vol_array.max()}, Mean: {vol_array.mean()}")
            skin_mesh = extract_skin_mesh(
                vol_array, 
                vol_spacing, 
                threshold=SKIN_THRESHOLD, 
                output_filename=output_file_path
            )
            
    print("\n모든 환자 데이터의 피부 메쉬 추출 및 저장이 완료되었습니다.")