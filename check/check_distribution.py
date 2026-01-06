import os
from pathlib import Path
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import json
import sys

# 상위 디렉토리를 path에 추가
sys.path.append(str(Path(__file__).parent.parent))
from utils.config import DOTA_DATA_ROOT

def parse_meta_file(meta_path):
    """메타 파일에서 GSD 정보 추출"""
    gsd = None
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            for line in f:
                if line.startswith('gsd:'):
                    try:
                        gsd = float(line.split(':')[1].strip())
                    except:
                        pass
    return gsd

def parse_label_file(label_path):
    """라벨 파일에서 객체 정보 추출 (OBB 형식)"""
    objects = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 9:
                    # x1 y1 x2 y2 x3 y3 x4 y4 class_name difficulty
                    try:
                        coords = [float(parts[i]) for i in range(8)]
                        x_coords = coords[0::2]  # x1, x2, x3, x4
                        y_coords = coords[1::2]  # y1, y2, y3, y4

                        class_name = parts[8]
                        difficulty = int(parts[9]) if len(parts) > 9 else 0

                        # Bounding box 크기 계산 (픽셀)
                        width_pixel = max(x_coords) - min(x_coords)
                        height_pixel = max(y_coords) - min(y_coords)
                        area_pixel = width_pixel * height_pixel

                        objects.append({
                            'class': class_name,
                            'difficulty': difficulty,
                            'width_pixel': width_pixel,
                            'height_pixel': height_pixel,
                            'area_pixel': area_pixel,
                        })
                    except:
                        continue
    return objects

def analyze_size_distribution(data_root, splits=['train', 'val'], min_samples=100):
    """픽셀 크기와 실제 크기 분포 분석"""

    print("="*80)
    print("Object Size Distribution Analysis")
    print("="*80)

    # 전체 데이터 수집
    class_data = defaultdict(lambda: {
        'pixel_widths': [],
        'pixel_heights': [],
        'pixel_areas': [],
        'real_widths': [],
        'real_heights': [],
        'real_areas': [],
        'count': 0
    })

    total_objects = 0
    total_with_gsd = 0

    for split in splits:
        print(f"\nProcessing {split} split...")

        meta_dir = Path(data_root) / split / 'meta'
        label_dir = Path(data_root) / split / 'labelTxt' / f'DOTA-v2.0_{split}'

        if not meta_dir.exists():
            print(f"Meta directory not found: {meta_dir}")
            continue

        for meta_file in sorted(meta_dir.glob('*.txt')):
            image_id = meta_file.stem

            # GSD 파싱
            gsd = parse_meta_file(meta_file)

            # 라벨 파싱
            label_file = label_dir / f'{image_id}.txt'
            objects = parse_label_file(label_file)

            for obj in objects:
                total_objects += 1
                class_name = obj['class']

                # 픽셀 크기는 항상 저장
                class_data[class_name]['pixel_widths'].append(obj['width_pixel'])
                class_data[class_name]['pixel_heights'].append(obj['height_pixel'])
                class_data[class_name]['pixel_areas'].append(obj['area_pixel'])

                # GSD가 있으면 실제 크기도 계산
                if gsd is not None:
                    total_with_gsd += 1
                    real_width = obj['width_pixel'] * gsd  # meters
                    real_height = obj['height_pixel'] * gsd  # meters
                    real_area = obj['area_pixel'] * (gsd ** 2)  # square meters

                    class_data[class_name]['real_widths'].append(real_width)
                    class_data[class_name]['real_heights'].append(real_height)
                    class_data[class_name]['real_areas'].append(real_area)

                class_data[class_name]['count'] += 1

    print(f"\nTotal objects processed: {total_objects}")
    print(f"Objects with GSD: {total_with_gsd} ({total_with_gsd/total_objects*100:.2f}%)")

    # 통계 계산 및 출력
    print("\n" + "="*80)
    print("Size Distribution Statistics")
    print("="*80)

    results = {}

    # 샘플 수가 충분한 클래스만 분석
    valid_classes = [cls for cls, data in class_data.items()
                     if len(data['real_widths']) >= min_samples]

    print(f"\nClasses with sufficient samples (>= {min_samples}): {len(valid_classes)}")
    print(f"Excluded classes: {set(class_data.keys()) - set(valid_classes)}")

    for class_name in sorted(valid_classes, key=lambda x: class_data[x]['count'], reverse=True):
        data = class_data[class_name]

        pixel_widths = np.array(data['pixel_widths'])
        pixel_heights = np.array(data['pixel_heights'])
        pixel_areas = np.array(data['pixel_areas'])

        real_widths = np.array(data['real_widths'])
        real_heights = np.array(data['real_heights'])
        real_areas = np.array(data['real_areas'])

        results[class_name] = {
            'count': data['count'],
            'count_with_gsd': len(real_widths),
            'pixel_size': {
                'width': {
                    'mean': float(np.mean(pixel_widths)),
                    'std': float(np.std(pixel_widths)),
                    'median': float(np.median(pixel_widths)),
                    'min': float(np.min(pixel_widths)),
                    'max': float(np.max(pixel_widths)),
                },
                'height': {
                    'mean': float(np.mean(pixel_heights)),
                    'std': float(np.std(pixel_heights)),
                    'median': float(np.median(pixel_heights)),
                    'min': float(np.min(pixel_heights)),
                    'max': float(np.max(pixel_heights)),
                },
                'area': {
                    'mean': float(np.mean(pixel_areas)),
                    'std': float(np.std(pixel_areas)),
                    'median': float(np.median(pixel_areas)),
                    'min': float(np.min(pixel_areas)),
                    'max': float(np.max(pixel_areas)),
                }
            },
            'real_size': {
                'width': {
                    'mean': float(np.mean(real_widths)),
                    'std': float(np.std(real_widths)),
                    'median': float(np.median(real_widths)),
                    'min': float(np.min(real_widths)),
                    'max': float(np.max(real_widths)),
                },
                'height': {
                    'mean': float(np.mean(real_heights)),
                    'std': float(np.std(real_heights)),
                    'median': float(np.median(real_heights)),
                    'min': float(np.min(real_heights)),
                    'max': float(np.max(real_heights)),
                },
                'area': {
                    'mean': float(np.mean(real_areas)),
                    'std': float(np.std(real_areas)),
                    'median': float(np.median(real_areas)),
                    'min': float(np.min(real_areas)),
                    'max': float(np.max(real_areas)),
                }
            }
        }

        print(f"\n{'='*80}")
        print(f"Class: {class_name}")
        print(f"{'='*80}")
        print(f"Total samples: {data['count']} (with GSD: {len(real_widths)})")

        print(f"\n📏 Pixel Size (pixels):")
        print(f"  Width:  {np.mean(pixel_widths):7.2f} ± {np.std(pixel_widths):7.2f} "
              f"[{np.min(pixel_widths):7.2f} - {np.max(pixel_widths):7.2f}]")
        print(f"  Height: {np.mean(pixel_heights):7.2f} ± {np.std(pixel_heights):7.2f} "
              f"[{np.min(pixel_heights):7.2f} - {np.max(pixel_heights):7.2f}]")
        print(f"  Area:   {np.mean(pixel_areas):7.2f} ± {np.std(pixel_areas):7.2f} "
              f"[{np.min(pixel_areas):7.2f} - {np.max(pixel_areas):7.2f}]")

        print(f"\n🌍 Real Size (meters):")
        print(f"  Width:  {np.mean(real_widths):7.2f} ± {np.std(real_widths):7.2f} "
              f"[{np.min(real_widths):7.2f} - {np.max(real_widths):7.2f}]")
        print(f"  Height: {np.mean(real_heights):7.2f} ± {np.std(real_heights):7.2f} "
              f"[{np.min(real_heights):7.2f} - {np.max(real_heights):7.2f}]")
        print(f"  Area:   {np.mean(real_areas):7.2f} ± {np.std(real_areas):7.2f} "
              f"[{np.min(real_areas):7.2f} - {np.max(real_areas):7.2f}] m²")

    # 결과 저장
    output_file = 'size_distribution_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Results saved to {output_file}")

    return class_data, results, valid_classes

def plot_size_distributions(class_data, valid_classes, output_dir='plots'):
    """크기 분포 시각화 - 클래스별 크기 비교"""

    Path(output_dir).mkdir(exist_ok=True)

    # 스타일 설정
    sns.set_style("whitegrid")

    print("\n" + "="*80)
    print("Generating plots...")
    print("="*80)

    # 주요 클래스만 선택 (샘플 수가 많은 순서대로)
    main_classes = valid_classes[:10]

    # 1. 실제 크기 분포 - Violin Plot (Width, Height, Area)
    fig, axes = plt.subplots(3, 1, figsize=(14, 18))

    # Width 분포
    ax = axes[0]
    width_data = [class_data[cls]['real_widths'] for cls in main_classes]
    positions = range(len(main_classes))
    parts = ax.violinplot(width_data, positions=positions, widths=0.7,
                          showmeans=True, showmedians=True)
    ax.set_xticks(positions)
    ax.set_xticklabels(main_classes, rotation=45, ha='right')
    ax.set_ylabel('Real Width (meters)', fontsize=12)
    ax.set_title('Real Width Distribution by Class (GSD-normalized)', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Height 분포
    ax = axes[1]
    height_data = [class_data[cls]['real_heights'] for cls in main_classes]
    parts = ax.violinplot(height_data, positions=positions, widths=0.7,
                          showmeans=True, showmedians=True)
    ax.set_xticks(positions)
    ax.set_xticklabels(main_classes, rotation=45, ha='right')
    ax.set_ylabel('Real Height (meters)', fontsize=12)
    ax.set_title('Real Height Distribution by Class (GSD-normalized)', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Area 분포
    ax = axes[2]
    area_data = [class_data[cls]['real_areas'] for cls in main_classes]
    parts = ax.violinplot(area_data, positions=positions, widths=0.7,
                          showmeans=True, showmedians=True)
    ax.set_xticks(positions)
    ax.set_xticklabels(main_classes, rotation=45, ha='right')
    ax.set_ylabel('Real Area (m²)', fontsize=12)
    ax.set_title('Real Area Distribution by Class (GSD-normalized)', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.set_yscale('log')  # Area는 범위가 넓으니 log scale

    plt.tight_layout()
    plt.savefig(f'{output_dir}/real_size_violin_plot.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_dir}/real_size_violin_plot.png")
    plt.close()

    # 2. 실제 크기 분포 - Box Plot (더 깔끔한 비교)
    fig, axes = plt.subplots(3, 1, figsize=(14, 18))

    # Width Box Plot
    ax = axes[0]
    bp = ax.boxplot(width_data, positions=positions, widths=0.6,
                    patch_artist=True, showfliers=False)
    for patch in bp['boxes']:
        patch.set_facecolor('skyblue')
        patch.set_alpha(0.7)
    ax.set_xticks(positions)
    ax.set_xticklabels(main_classes, rotation=45, ha='right')
    ax.set_ylabel('Real Width (meters)', fontsize=12)
    ax.set_title('Real Width Distribution by Class (GSD-normalized)', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Height Box Plot
    ax = axes[1]
    bp = ax.boxplot(height_data, positions=positions, widths=0.6,
                    patch_artist=True, showfliers=False)
    for patch in bp['boxes']:
        patch.set_facecolor('lightcoral')
        patch.set_alpha(0.7)
    ax.set_xticks(positions)
    ax.set_xticklabels(main_classes, rotation=45, ha='right')
    ax.set_ylabel('Real Height (meters)', fontsize=12)
    ax.set_title('Real Height Distribution by Class (GSD-normalized)', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Area Box Plot
    ax = axes[2]
    bp = ax.boxplot(area_data, positions=positions, widths=0.6,
                    patch_artist=True, showfliers=False)
    for patch in bp['boxes']:
        patch.set_facecolor('lightgreen')
        patch.set_alpha(0.7)
    ax.set_xticks(positions)
    ax.set_xticklabels(main_classes, rotation=45, ha='right')
    ax.set_ylabel('Real Area (m²)', fontsize=12)
    ax.set_title('Real Area Distribution by Class (GSD-normalized)', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/real_size_box_plot.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_dir}/real_size_box_plot.png")
    plt.close()

    # 3. 픽셀 크기 vs 실제 크기 비교 (주요 클래스만)
    vehicle_classes = [cls for cls in main_classes if 'vehicle' in cls or cls in ['plane', 'ship', 'helicopter']]

    if len(vehicle_classes) > 0:
        fig, axes = plt.subplots(2, 1, figsize=(14, 12))

        # 픽셀 Width
        ax = axes[0]
        pixel_width_data = [class_data[cls]['pixel_widths'] for cls in vehicle_classes]
        positions = range(len(vehicle_classes))
        bp = ax.boxplot(pixel_width_data, positions=positions, widths=0.6,
                        patch_artist=True, showfliers=False)
        for patch in bp['boxes']:
            patch.set_facecolor('orange')
            patch.set_alpha(0.7)
        ax.set_xticks(positions)
        ax.set_xticklabels(vehicle_classes, rotation=45, ha='right')
        ax.set_ylabel('Pixel Width (pixels)', fontsize=12)
        ax.set_title('Pixel Width Distribution - Vehicle Classes (WITHOUT GSD normalization)',
                     fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        # 실제 Width
        ax = axes[1]
        real_width_data = [class_data[cls]['real_widths'] for cls in vehicle_classes]
        bp = ax.boxplot(real_width_data, positions=positions, widths=0.6,
                        patch_artist=True, showfliers=False)
        for patch in bp['boxes']:
            patch.set_facecolor('skyblue')
            patch.set_alpha(0.7)
        ax.set_xticks(positions)
        ax.set_xticklabels(vehicle_classes, rotation=45, ha='right')
        ax.set_ylabel('Real Width (meters)', fontsize=12)
        ax.set_title('Real Width Distribution - Vehicle Classes (WITH GSD normalization)',
                     fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{output_dir}/pixel_vs_real_comparison.png', dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {output_dir}/pixel_vs_real_comparison.png")
        plt.close()

    # 4. 통계 요약 테이블 시각화
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.axis('tight')
    ax.axis('off')

    # 테이블 데이터 준비
    table_data = [['Class', 'Count',
                   'Pixel Width\n(mean±std)', 'Real Width\n(mean±std)',
                   'Pixel Height\n(mean±std)', 'Real Height\n(mean±std)',
                   'Pixel Area\n(mean±std)', 'Real Area\n(mean±std)']]

    for cls in main_classes:
        pw = class_data[cls]['pixel_widths']
        ph = class_data[cls]['pixel_heights']
        pa = class_data[cls]['pixel_areas']
        rw = class_data[cls]['real_widths']
        rh = class_data[cls]['real_heights']
        ra = class_data[cls]['real_areas']

        row = [
            cls,
            f"{len(rw):,}",
            f"{np.mean(pw):.1f}±{np.std(pw):.1f}",
            f"{np.mean(rw):.1f}±{np.std(rw):.1f}m",
            f"{np.mean(ph):.1f}±{np.std(ph):.1f}",
            f"{np.mean(rh):.1f}±{np.std(rh):.1f}m",
            f"{np.mean(pa):.0f}±{np.std(pa):.0f}",
            f"{np.mean(ra):.0f}±{np.std(ra):.0f}m²"
        ]
        table_data.append(row)

    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.18, 0.08, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # 헤더 스타일링
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    plt.title('Size Statistics Summary by Class', fontsize=16, fontweight='bold', pad=20)
    plt.savefig(f'{output_dir}/size_statistics_table.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_dir}/size_statistics_table.png")
    plt.close()

    print(f"\n✅ All plots saved to {output_dir}/")

if __name__ == '__main__':

    # 분석 실행
    class_data, results, valid_classes = analyze_size_distribution(
        DOTA_DATA_ROOT,
        splits=['train', 'val'],
        min_samples=100
    )

    # 시각화
    plot_size_distributions(class_data, valid_classes, output_dir='plots')

    print("\n" + "="*80)
    print("Analysis Complete!")
    print("="*80)
