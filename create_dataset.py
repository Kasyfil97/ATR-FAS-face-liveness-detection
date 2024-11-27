import cv2
import argparse
import torch
import gc
import numpy as np
from PIL import Image
from preprocessing import Preprocessing
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from functools import partial
class DatasetCreator:
    def __init__(self, batch_size: int = 8):
        self.preprocessor = Preprocessing()
        self.batch_size = batch_size
    
    def process_one_image(self, image_path, output_dir, is_real: bool):
        try:
            image = cv2.imread(str(image_path))
            crop_face = self.preprocessor.crop_face(image, align=True, return_shape=256)
            if crop_face is None or crop_face.size == 0:
                print(f'no face detected in {image_path}')
                return
            
            depth_map = self.preprocessor.depth_maps(crop_face) if is_real else self.preprocessor.spoof_depth_map()
            depth_map = Image.fromarray((depth_map*255.).astype(np.uint8))

            label_path = output_dir / 'label'
            label_path.mkdir(parents=True, exist_ok=True)
            depth_map.save(label_path / image_path.name)

            crop_face = Image.fromarray(crop_face)
            image_for_diff = self.preprocessor.transform(crop_face)
            return image_for_diff
        except Exception as e:
            print(f'error processing {image_path}: {e}')
            return
        
    def process_batch(self, image_paths: list, output_dir: Path, is_real: bool):
        with ThreadPoolExecutor(max_workers=self.batch_size) as executor:
            process_one_image = partial(self.process_one_image, output_dir=output_dir, is_real=is_real)
            results = executor.map(process_one_image, image_paths)
            results = list(result for result in results if result is not None)
        
        if not results:
            return
        
        stack_images = self.preprocessor.stack_image(results)
        diff_norm = self.preprocessor.differential_norm(stack_images)

        input_path = output_dir / 'input'
        input_path.mkdir(parents=True, exist_ok=True)

        for i, image_diff in enumerate(diff_norm):
            with torch.no_grad():
                tensor_normalized = (image_diff * 255).clamp(0, 255).to(torch.uint8)
                image_numpy = tensor_normalized.permute(1, 2, 0).cpu().numpy()
                Image.fromarray(image_numpy).save(input_path / f'frame_{i+1}.png')

        del stack_images, diff_norm, results
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def process_dataset(self, input_dir: Path, output_dir: Path):
        for label_folder in input_dir.iterdir():
            if not label_folder.is_dir():
                continue
            is_real = label_folder.name =='real'
            
            for image_folder in label_folder.iterdir():
                if not image_folder.is_dir():
                    continue
                image_paths = list(image_folder.glob('*.jpg')) + list(image_folder.glob('*.png'))

                self.process_batch(image_paths, output_dir / label_folder.name, is_real)



if __name__ == "__main__":
    arg = argparse.ArgumentParser()
    arg.add_argument("--input_dir", type=str, required=True, help = 'path to input directory')
    arg.add_argument("--output_dir", type=str, required=True, help = 'path to output directory')
    args = arg.parse_args()

    DatasetCreator().process_dataset(Path(args.input_dir), Path(args.output_dir))

