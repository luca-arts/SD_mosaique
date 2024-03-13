from datetime import datetime

import split_image
import torch
import yaml
from diffusers import AutoPipelineForImage2Image, StableDiffusionImg2ImgPipeline
from diffusers.utils import load_image, make_image_grid
from PIL import Image
import numpy as np
import split_image.split
import os
from pathlib import Path
import argparse


class SDMosaique:
    def __init__(self, input_image, prompt):
        self.input_image = input_image
        self.prompt = prompt
        self.model_id_or_path = "runwayml/stable-diffusion-v1-5"
        self.folder_structure = None
        self.create_folder_structure()
        self.amount_of_rows = 15
        self.amount_of_cols = 15
        self.cell_resolution = (512, 512)
        self.pipe = None

    def up_scale_pixels(self, img: Image.Image):
        # todo do we want to have a different upscaling algorithm?
        upscale = img.resize(self.cell_resolution)
        return upscale

    def prep_pipeline(self):
        device = "cuda"
        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(self.model_id_or_path, torch_dtype=torch.float16)
        self.pipe.safety_checker = lambda images, clip_input: (images, [False] * len(images))  # allows for NSFW images
        self.pipe = self.pipe.to(device)

    def generate_img(self, prompt: str, base_image: Image.Image, preview: bool = False):
        
        image = self.pipe(prompt=prompt, image=base_image, strength=0.75, guidance_scale=7.5, num_inference_steps=50).images[
            0]
        if preview:
            grid = make_image_grid([base_image, image], rows=1, cols=2)
            grid.show()
        return image

    def test_generate_img(self):
        init_image = load_image(
            'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png')

        prompt = "cat wizard, gandalf, lord of the rings, detailed, fantasy, cute, adorable, Pixar, Disney, 8k"

        img = self.generate_img(prompt, init_image, True)
        img.save("gen_img.png")

    def load_pil_image(self, source: str) -> np.ndarray:
        """
        Opens an image from specified source and returns a numpy array
        with image rgb data
        """
        with Image.open(source) as im:
            im_arr = np.asarray(im)
        return im_arr

    def stitch_images(self):
        # Determine the size of the output image
        output_shape = (self.amount_of_rows * self.cell_resolution[0], self.amount_of_cols * self.cell_resolution[1], 3)

        # Initialize an empty array to store the output image
        output_image = np.zeros(output_shape, dtype=np.uint8)

        # Loop over each upscaled "generated image pixel" and insert it into the output image
        for i in range(self.amount_of_rows):
            for j in range(self.amount_of_cols):
                # Compute the index of the upscaled pixel in the list
                index = i * self.amount_of_cols + j

                gen_img_pixel = self.find_matching_file_by_index(self.folder_structure.gen_imgs_path, index)
                # Insert the upscaled pixel into the output image
                output_image[i * self.cell_resolution[0]:(i + 1) * self.cell_resolution[0],
                    j * self.cell_resolution[1]:(j + 1) * self.cell_resolution[1]] = Image.open(gen_img_pixel)

        # Convert the output image to a PIL Image object and save it to disk
        output_image = Image.fromarray(output_image)
        output_image.save(os.path.join(self.folder_structure.result_path,self.input_image))

    def find_matching_file_by_index(self, path, index):
        # Initialize an empty string to store the matching filename
        matching_filename = ""

        # Loop over each file in the folder
        for filename in os.listdir(path):

            # Check if the current file is an image (based on its file extension)
            if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
                filename_stem = Path(filename).stem

                # Check if the current filename ends with the target index (as a string)
                if filename_stem.endswith(str(index)):
                    # Update the matching filename
                    matching_filename = filename

                    # Stop the loop (since we've found the matching filename)
                    break

        # Return the matching filename (or an empty string if no match was found)
        return os.path.join(path, matching_filename)

    def generate_upscaled_split_images(self):
        """
        this function takes an input image, splits it into x rows & cols and upscales each cell ,
        :return:
        """
        face_im_arr = self.load_pil_image(self.input_image)
        width = face_im_arr.shape[0]
        height = face_im_arr.shape[1]
        print("input image width = {}, height={}".format(width, height))

        # split the image and save it in ./imgs
        split_image.split_image(self.input_image, self.amount_of_rows, self.amount_of_cols, should_square=False,
                                should_cleanup=False, output_dir=self.folder_structure.imgs_path, should_quiet=True)

        # Loop over each file in the folder
        for filename in os.listdir(self.folder_structure.imgs_path):

            # Check if the current file is an image (based on its file extension)
            if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
                # Construct the full path to the image file
                image_path = os.path.join(os.getcwd(), self.folder_structure.imgs_path, filename)
                image_save_path = os.path.join(os.getcwd(), self.folder_structure.upscale_imgs_path, filename)
                image = Image.open(image_path)
                upscaled_img = self.up_scale_pixels(image)
                upscaled_img.save(image_save_path)

    def generate_img2img_folder(self):
        self.prep_pipeline()
        for filename in os.listdir(self.folder_structure.upscale_imgs_path):
            # Check if the current file is an image (based on its file extension)
            if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
                print("generating for tile {}".format(filename))
                # Construct the full path to the image file
                image_path = os.path.join(os.getcwd(), self.folder_structure.upscale_imgs_path, filename)
                image_save_path = os.path.join(os.getcwd(), self.folder_structure.gen_imgs_path, filename)

                image = Image.open(image_path)

                gen_img = self.generate_img(self.prompt, image, False)

                gen_img.save(image_save_path)

    def create_folder_structure(self):
        # Get the current date and time
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d_%H-%M-%S")

        # Create a root folder with the date and time as its name
        root_name = f"results/run_{date_str}"
        self.folder_structure = FolderStructure(root_name)
        self.folder_structure.create_folders()

        # Create a config object with the specified properties
        config = Config(date_str, self.prompt, self.input_image)

        # Create a YAML config file with the specified properties
        self.folder_structure.create_config(config)


class Config:
    def __init__(self, date, prompt, image_link):
        self.date = date
        self.prompt = prompt
        self.image_link = str(image_link)


class FolderStructure:
    def __init__(self, root_name):
        self.root_name = root_name
        self.root_path = os.path.join(os.getcwd(), root_name)
        self.gen_imgs_path = os.path.join(self.root_path, "gen_imgs")
        self.upscale_imgs_path = os.path.join(self.root_path, "upscale_imgs")

        self.imgs_path = os.path.join(self.root_path, "imgs")
        self.result_path = os.path.join(self.root_path, "result")

    def create_folders(self):
        os.makedirs(self.root_path, exist_ok=True)
        os.makedirs(self.gen_imgs_path, exist_ok=True)
        os.makedirs(self.upscale_imgs_path, exist_ok=True)
        os.makedirs(self.imgs_path, exist_ok=True)
        os.makedirs(self.result_path, exist_ok=True)

    def create_config(self, config):
        config_path = os.path.join(self.root_path, "config.yaml")
        with open(config_path, "w") as f:
            yaml.dump(config.__dict__, f)


def main(input_image: Path, prompt: str):
    sd_mosaique = SDMosaique(input_image, prompt)
    sd_mosaique.generate_upscaled_split_images()

    sd_mosaique.generate_img2img_folder()

    sd_mosaique.stitch_images()


if __name__ == "__main__":
    # Create an argument parser object
    parser = argparse.ArgumentParser(description="Process an image and a string.")

    # Add arguments to the parser
    parser.add_argument("image", help="Path to the starting image file", default="wio.jpg", nargs='?')
    parser.add_argument("prompt", help="A prompt to steer the images being generated",
                        default="cat wizard, gandalf, lord of the rings, detailed, fantasy, cute, adorable, Pixar, "
                                "Disney, 8k", nargs='?')

    # Parse the command-line arguments
    args = parser.parse_args()
    image_path = Path(args.image)
    if image_path.exists():
        # Call the main function with the parsed arguments
        main(image_path, args.prompt)
    else:
        print("the image is not found")
