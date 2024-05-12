import json
import os
from PIL import Image

def process_annotations(json_path, images_dir, output_dir):
    # Завантаження даних аннотації
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Перевірка наявності каталогу виводу
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Список для збереження даних про зображення
    images_data = []

    for annotation in data['annotations']:
        image_info = next((img for img in data['images'] if img['id'] == annotation['image_id']), None)
        if image_info:
            img_path = os.path.join(images_dir, image_info['file_name'])
            try:
                with Image.open(img_path) as img:
                    bbox = annotation['bbox']
                    # Вирізання зображення за баундінг боксом
                    cropped_image = img.crop((bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]))
                    # Створення назви файлу для збереженого зображення
                    output_file_name = f"{annotation['image_id']}_{annotation['id']}.png"
                    output_file_path = os.path.join(output_dir, output_file_name)
                    cropped_image.save(output_file_path)
                    
                    # Записуємо інформацію для конфігураційного файлу
                    images_data.append({
                        'id': annotation['id'],
                        'file_name': output_file_name,
                        'category_id': annotation['category_id']
                    })
            except Exception as e:
                print(f"Error processing image {img_path}: {e}")

    # Запис конфігураційного файлу
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(images_data, f, indent=4)

# Використання функції
json_path = 'init_dataset/annotations.json'
images_dir = 'init_dataset/images'
output_dir = 'dataset/train'
process_annotations(json_path, images_dir, output_dir)