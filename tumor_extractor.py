import numpy as np
import pyvista as pv
import SimpleITK as sitk
import os

def extract_nifti_to_obj(file_path, output_filename="Segmentation.obj"):
    print(f"[{file_path}] 파일을 로드합니다...")
    
    # 1. NIfTI 파일 읽기 (단일 파일 로드)
    image = sitk.ReadImage(file_path)
    volume_array = sitk.GetArrayFromImage(image)
    spacing = image.GetSpacing()
    
    # SimpleITK (Z, Y, X) -> PyVista (X, Y, Z) 변환
    volume_array = np.transpose(volume_array, (2, 1, 0))
    
    print(f"-> 로드 완료! (크기: {volume_array.shape}, Spacing: {spacing})")
    
    # 2. PyVista ImageData(그리드) 생성
    print("\n3D 메쉬(OBJ)로 변환을 시작합니다...")
    grid = pv.ImageData()
    grid.dimensions = volume_array.shape
    grid.spacing = spacing
    grid.point_data["values"] = volume_array.flatten(order="F")
    
    # 🌟 3. 표면 추출 (Contour)
    # 라벨링 마스크는 보통 0(배경)과 1(라벨)로 되어 있으므로, 
    # 임계값을 0.5로 설정하여 경계면을 정확히 추출합니다.
    mesh = grid.contour([0.5])
    
    # 4. 스무딩 (옵션: 라벨링 굴곡이 너무 각져 보이지 않게 다듬기)
    print("-> 표면 스무딩 처리 중...")
    mesh = mesh.smooth_taubin(n_iter=50, pass_band=0.01)
    
    # 5. OBJ 파일 저장 (원본 .nii 파일이 있는 폴더에 저장)
    output_dir = os.path.dirname(file_path)
    output_path = os.path.join(output_dir, output_filename)
    
    mesh.save(output_path)
    print(f"\n✅ 저장 완료: {output_path}")
    
    return output_path, mesh

def visualize_mesh(mesh):
    """저장된 라벨링 모델을 화면에 렌더링합니다."""
    print("\n3D 렌더링 창을 띄웁니다. (종료하려면 창을 닫으세요)")
    plotter = pv.Plotter()
    
    # 세그멘테이션 데이터가 눈에 잘 띄도록 연한 붉은색(salmon) 적용
    plotter.add_mesh(mesh, color="salmon", opacity=0.9, specular=0.3)
    plotter.set_background("#222222")
    plotter.show()

if __name__ == "__main__":
    # 파일 경로
    NIFTI_PATH = r"data\Segmentation.nii"
    
    # 변환 및 저장 실행
    saved_path, seg_mesh = extract_nifti_to_obj(NIFTI_PATH, output_filename="Segmentation.obj")
    
    # 생성된 3D 메쉬 눈으로 확인하기
    visualize_mesh(seg_mesh)