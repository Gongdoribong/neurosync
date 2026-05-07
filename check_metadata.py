import SimpleITK as sitk
import numpy as np

def check_dicom_metadata(folder_path):
    print(f"[{folder_path}] 데이터의 메타데이터를 분석합니다...\n")
    
    # 1. DICOM 시리즈 로드
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(folder_path)
    if not series_ids:
        raise ValueError("❌ 오류: 해당 폴더에서 DICOM 데이터를 찾을 수 없습니다.")
        
    dicom_names = reader.GetGDCMSeriesFileNames(folder_path, series_ids[0])
    reader.SetFileNames(dicom_names)
    
    # 이미지 객체 생성 (배열 변환 없이 메타데이터만 쏙 빼냅니다)
    image = reader.Execute()
    
    # 🌟 2. 핵심 메타데이터 3가지 추출
    spacing = image.GetSpacing()
    origin = image.GetOrigin()
    direction = image.GetDirection()
    
    # 3. 결과 출력 (Direction은 9개의 숫자로 나오므로 3x3 행렬로 변환해서 보여줍니다)
    direction_matrix = np.array(direction).reshape(3, 3)
    
    print("=== 🌟 DICOM 물리적 메타데이터 (mm 변환의 3요소) ===")
    print(f"1. Spacing (간격) : {spacing}")
    print("   -> 한 칸(Voxel)의 실제 크기 (X, Y, Z mm)\n")
    
    print(f"2. Origin (원점)  : {origin}")
    print("   -> CT 기계 공간상에서 환자 데이터가 시작되는 절대 기준점 (X, Y, Z mm)\n")
    
    print(f"3. Direction (방향 행렬):")
    for row in direction_matrix:
        print(f"   [{row[0]:6.3f}, {row[1]:6.3f}, {row[2]:6.3f}]")
    print("   -> 환자의 촬영 자세(회전 상태)를 나타내는 3x3 코사인 행렬")
    print("==================================================\n")
    
    return spacing, origin, direction_matrix

if __name__ == "__main__":
    # 🎯 지영 님의 현재 연구 데이터 경로
    DICOM_FOLDER = r"data\tumor"
    
    # 메타데이터 추출 실행
    check_dicom_metadata(DICOM_FOLDER)