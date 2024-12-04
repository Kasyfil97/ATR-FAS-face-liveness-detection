import os
import datetime
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
import logging

# Set up logging
logging.basicConfig(filename='dataset.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

class DatasetCreator:
    def __init__(self, batch_size: int = 8):
        self.preprocessor = Preprocessing()
        self.batch_size = batch_size
    
    def process_one_image(self, 
                          image_path, 
                          output_dir, 
                          is_real: bool, 
                          align_face = True, 
                          return_shape = 256, 
                          input_crop = False):
        
        try:
            if input_crop:
                image = cv2.imread(str(image_path))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                crop_face = self.preprocessor.crop_face(image, align=align_face, return_shape=return_shape)
                if crop_face is None or crop_face.size == 0:
                    logging.warning(f'No face detected in {image_path}')
                    return
            else:
                crop_face = cv2.imread(str(image_path))
                crop_face = cv2.cvtColor(crop_face, cv2.COLOR_BGR2RGB)
            try:
                depth_map = self.preprocessor.depth_maps(crop_face) if is_real else self.preprocessor.spoof_depth_map()
                depth_map = Image.fromarray((depth_map*255.).astype(np.uint8))

                label_path = output_dir / 'label'
                label_path.mkdir(parents=True, exist_ok=True)
                depth_map.save(label_path / image_path.name)

            except Exception:
                logging.error(f'No depth map for {image_path}')
            
            crop_face = Image.fromarray(crop_face)
            image_for_diff = self.preprocessor.transform(crop_face)
            return image_for_diff
        except Exception as e:
            logging.error(f'Error processing {image_path}: {e}')
            return
        
    def process_batch(self, 
                      image_paths: list, 
                      output_dir: Path, 
                      is_real: bool, 
                      align_face = True, 
                      input_crop = False, 
                      return_shape = 256):
        with ThreadPoolExecutor(max_workers=self.batch_size) as executor:
            process_one_image = partial(self.process_one_image, 
                                        output_dir=output_dir, 
                                        align_face=align_face, 
                                        is_real=is_real, 
                                        return_shape=return_shape, 
                                        input_crop=input_crop)
            
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

    def process_dataset(self, 
                        input_dir: Path, 
                        output_dir: Path, 
                        align_face = True,
                        input_crop = False,
                        return_shape = 256):
        
        os.makedirs(output_dir, exist_ok=True)
        for label_folder in input_dir.iterdir():
            if not label_folder.is_dir():
                continue
            is_real = label_folder.name
            
            for image_folder in label_folder.iterdir():
                if not image_folder.is_dir():
                    continue
                image_paths = list(image_folder.glob('*.jpg')) + list(image_folder.glob('*.png'))

                self.process_batch(image_paths, output_dir / image_folder.name, align_face, is_real, input_crop, return_shape)



if __name__ == "__main__":
    date = datetime.datetime.now().strftime("%Y-%m-%d")
    arg = argparse.ArgumentParser()
    arg.add_argument("--input_dir", type=str, required=True, help = 'path to input directory')
    arg.add_argument("--output_dir", type=str, required=True, help = 'path to output directory')
    arg.add_argument("--align_face", type=bool, default=True, help = 'whether to align face')
    args = arg.parse_args()

    logging.info(f'Starting dataset processing {date}: {args.input_dir} -> {args.output_dir}')
    DatasetCreator().process_dataset(Path(args.input_dir), Path(args.output_dir), args.align_face)
    logging.info('Finished dataset processing')
