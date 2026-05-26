import numpy as np
import pyvista as pv
import SimpleITK as sitk
import os

def extract_nifti_to_obj(file_path, output_filename="Segmentation.obj"):
    print(f"\n[{file_path}] 파일을 로드합니다...")
    
    # 1. NIfTI 파일 읽기
    image = sitk.ReadImage(file_path)
    volume_array = sitk.GetArrayFromImage(image)
    spacing = image.GetSpacing()
    
    # SimpleITK (Z, Y, X) -> PyVista (X, Y, Z) 변환
    volume_array = np.transpose(volume_array, (2, 1, 0))
    
    print(f"-> 로드 완료! (크기: {volume_array.shape}, Spacing: {spacing})")
    
    # 2. PyVista ImageData(그리드) 생성
    print("-> 3D 메쉬(OBJ)로 변환을 시작합니다...")
    grid = pv.ImageData()
    grid.dimensions = volume_array.shape
    grid.spacing = spacing
    grid.point_data["values"] = volume_array.flatten(order="F")
    
    # 3. 표면 추출 (Contour)
    mesh = grid.contour([0.5])
    
    # 4. 스무딩
    print("-> 표면 스무딩 처리 중...")
    mesh = mesh.smooth_taubin(n_iter=50, pass_band=0.01)
    
    # 5. OBJ 파일 저장
    output_dir = os.path.dirname(file_path)
    output_path = os.path.join(output_dir, output_filename)
    
    mesh.save(output_path)
    print(f"✅ 저장 완료: {output_path}")
    
    return output_path, mesh

def visualize_multiple_meshes(mesh_list):
    """여러 개의 저장된 라벨링 모델을 화면에 동시에 렌더링합니다."""
    if not mesh_list:
        print("\n시각화할 메쉬가 없습니다.")
        return

    print("\n3D 렌더링 창을 띄웁니다. (종료하려면 창을 닫으세요)")
    plotter = pv.Plotter()
    
    # 시각적 구분을 위한 색상 팔레트
    colors = ["salmon", "lightblue", "lightgreen", "gold"]
    
    for i, mesh in enumerate(mesh_list):
        color = colors[i % len(colors)]
        plotter.add_mesh(mesh, color=color, opacity=0.9, specular=0.3, label=f"Seg-{i+1}")
        
    plotter.set_background("#222222")
    plotter.add_legend() # 어떤 색상이 어떤 세그멘테이션인지 표시
    plotter.show()

if __name__ == "__main__":
    # 기본 폴더 경로 설정
    BASE_DIR = r"Control_Subjects\Full-Head Segmentation"
    
    # 생성된 메쉬들을 담을 리스트
    processed_meshes = []
    
    # 1부터 4까지 반복 처리
    for i in range(1, 5):
        nifti_name = f"Segmentation-{i}.nii"
        obj_name = f"Segmentation-{i}.obj"
        
        nifti_path = os.path.join(BASE_DIR, nifti_name)
        
        # 파일 존재 여부 확인
        if not os.path.exists(nifti_path):
            print(f"\n⚠️ 파일을 찾을 수 없습니다: {nifti_path}")
            continue
            
        # 변환 및 저장 실행
        saved_path, seg_mesh = extract_nifti_to_obj(nifti_path, output_filename=obj_name)
        processed_meshes.append(seg_mesh)
    
    # 생성된 3D 메쉬들 한 번에 시각화
    visualize_multiple_meshes(processed_meshes)