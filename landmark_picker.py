import pyvista as pv

def on_point_picked(point):
    """지점을 선택하면 터미널에 해당 지점의 3D 좌표를 출력합니다."""
    # point는 선택된 지점의 [x, y, z] 좌표 리스트입니다.
    if point is not None:
        print(f"🎯 픽(Pick)된 3D 좌표: [{point[0]:.2f}, {point[1]:.2f}, {point[2]:.2f}]")

def main():
    file_path = "data/neurosync_skin_nose.obj"
    print(f"[{file_path}] 파일을 불러오는 중...")
    
    try:
        # 3D 피부 모델 읽기
        mesh = pv.read(file_path)
    except Exception as e:
        print(f"오류: {file_path} 파일을 찾을 수 없거나 읽을 수 없습니다. ({e})")
        return

    plotter = pv.Plotter(window_size=[1000, 800])
    
    # 1. 모델 렌더링
    plotter.add_mesh(mesh, color="#EAC0A6", smooth_shading=True, opacity=0.8)
    
    # 2. 조작 안내 텍스트 추가
    instructions = (
        "[3D Landmark Picker]\n"
        "1. 마우스로 모델을 회전/확대하여 원하는 부위를 찾으세요.\n"
        "2. 해당 지점에 마우스를 올리고 키보드 'P'를 누르세요.\n\n"
        "👉 목표: 코끝, 왼쪽 눈꼬리, 오른쪽 눈꼬리, 턱끝 (총 4개)"
    )
    plotter.add_text(instructions, font_size=12, color="white", position="upper_left")
    
    # pickable_window=False로 설정하면 배경이 아닌 메쉬 표면에서만 좌표가 잡힙니다.
    plotter.enable_point_picking(
        callback=on_point_picked, 
        show_point=True, 
        color='red', 
        point_size=15,
        pickable_window=False
    )
    
    print("\n창이 열리면 4개 지점에서 'P' 키를 눌러 좌표를 추출한 뒤 저에게 알려주세요!")
    plotter.show()

if __name__ == "__main__":
    main()