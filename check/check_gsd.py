import os
from pathlib import Path
import numpy as np
from collections import defaultdict
from utils.config import DOTA_DATA_ROOT
import json

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
    """라벨 파일에서 객체 정보 추출"""
    objects = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 9:
                    # x1 y1 x2 y2 x3 y3 x4 y4 class_name difficulty
                    class_name = parts[8]
                    difficulty = int(parts[9]) if len(parts) > 9 else 0
                    objects.append({
                        'class': class_name,
                        'difficulty': difficulty
                    })
    return objects

def analyze_gsd_dataset(data_root, splits=['train', 'val']):
    """GSD 정보가 있는 데이터셋 분석"""

    results = {}

    for split in splits:
        print(f"\n{'='*60}")
        print(f"Analyzing {split.upper()} split")
        print(f"{'='*60}")

        meta_dir = Path(data_root) / split / 'meta'
        label_dir = Path(data_root) / split / 'labelTxt' / f'DOTA-v2.0_{split}'

        # 통계 수집
        total_images = 0
        images_with_gsd = 0
        images_without_gsd = 0

        gsd_values = []

        # 클래스별 통계
        class_stats_with_gsd = defaultdict(int)
        class_stats_without_gsd = defaultdict(int)
        total_instances_with_gsd = 0
        total_instances_without_gsd = 0

        # difficulty 별 통계 (GSD 있는 경우)
        difficulty_stats_with_gsd = defaultdict(int)

        # 각 이미지 처리
        if not meta_dir.exists():
            print(f"Meta directory not found: {meta_dir}")
            continue

        for meta_file in sorted(meta_dir.glob('*.txt')):
            image_id = meta_file.stem
            total_images += 1

            # GSD 파싱
            gsd = parse_meta_file(meta_file)

            # 라벨 파싱
            label_file = label_dir / f'{image_id}.txt'
            objects = parse_label_file(label_file)

            if gsd is not None:
                images_with_gsd += 1
                gsd_values.append(gsd)

                # 객체 통계
                for obj in objects:
                    class_stats_with_gsd[obj['class']] += 1
                    difficulty_stats_with_gsd[obj['difficulty']] += 1
                    total_instances_with_gsd += 1
            else:
                images_without_gsd += 1
                for obj in objects:
                    class_stats_without_gsd[obj['class']] += 1
                    total_instances_without_gsd += 1

        # 결과 저장
        results[split] = {
            'total_images': total_images,
            'images_with_gsd': images_with_gsd,
            'images_without_gsd': images_without_gsd,
            'gsd_coverage': images_with_gsd / total_images * 100 if total_images > 0 else 0,
            'gsd_stats': {
                'min': float(np.min(gsd_values)) if gsd_values else None,
                'max': float(np.max(gsd_values)) if gsd_values else None,
                'mean': float(np.mean(gsd_values)) if gsd_values else None,
                'median': float(np.median(gsd_values)) if gsd_values else None,
                'std': float(np.std(gsd_values)) if gsd_values else None,
            },
            'instances_with_gsd': total_instances_with_gsd,
            'instances_without_gsd': total_instances_without_gsd,
            'class_distribution_with_gsd': dict(class_stats_with_gsd),
            'class_distribution_without_gsd': dict(class_stats_without_gsd),
            'difficulty_distribution_with_gsd': dict(difficulty_stats_with_gsd)
        }

        # 출력
        print(f"\n📊 Image Statistics:")
        print(f"  Total images: {total_images}")
        print(f"  Images with GSD: {images_with_gsd} ({images_with_gsd/total_images*100:.2f}%)")
        print(f"  Images without GSD: {images_without_gsd} ({images_without_gsd/total_images*100:.2f}%)")

        if gsd_values:
            print(f"\n📏 GSD Statistics:")
            print(f"  Min GSD: {np.min(gsd_values):.6f}")
            print(f"  Max GSD: {np.max(gsd_values):.6f}")
            print(f"  Mean GSD: {np.mean(gsd_values):.6f}")
            print(f"  Median GSD: {np.median(gsd_values):.6f}")
            print(f"  Std GSD: {np.std(gsd_values):.6f}")
            print(f"  GSD Range: {np.max(gsd_values) - np.min(gsd_values):.6f}")

        print(f"\n🎯 Instance Statistics:")
        print(f"  Total instances (with GSD): {total_instances_with_gsd}")
        print(f"  Total instances (without GSD): {total_instances_without_gsd}")

        if total_instances_with_gsd > 0:
            print(f"\n📦 Class Distribution (Images with GSD):")
            sorted_classes = sorted(class_stats_with_gsd.items(), key=lambda x: x[1], reverse=True)
            for class_name, count in sorted_classes:
                percentage = count / total_instances_with_gsd * 100
                print(f"  {class_name:20s}: {count:6d} ({percentage:5.2f}%)")

        if total_instances_with_gsd > 0:
            print(f"\n💪 Difficulty Distribution (Images with GSD):")
            for difficulty, count in sorted(difficulty_stats_with_gsd.items()):
                percentage = count / total_instances_with_gsd * 100
                difficulty_label = "Easy" if difficulty == 0 else "Hard"
                print(f"  {difficulty_label:10s}: {count:6d} ({percentage:5.2f}%)")

    return results

if __name__ == '__main__':

    print("="*60)
    print("DOTA v2.0 Dataset - GSD Coverage Analysis")
    print("="*60)

    results = analyze_gsd_dataset(DOTA_DATA_ROOT, splits=['train', 'val'])

    # 결과를 JSON으로 저장
    output_file = 'gsd_analysis_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Results saved to {output_file}")

    # 전체 요약
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    total_images_all = sum(r['total_images'] for r in results.values())
    total_with_gsd_all = sum(r['images_with_gsd'] for r in results.values())
    total_instances_all = sum(r['instances_with_gsd'] for r in results.values())

    print(f"Total images (train+val): {total_images_all}")
    print(f"Images with GSD: {total_with_gsd_all} ({total_with_gsd_all/total_images_all*100:.2f}%)")
    print(f"Total instances with GSD: {total_instances_all}")