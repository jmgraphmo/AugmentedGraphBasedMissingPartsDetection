import os
import re
import io
import base64
from pathlib import Path
from PIL import Image
from openai import OpenAI

import tempfile 

client = OpenAI()
model= "gpt-image-1"

def main():
    os.makedirs("data/generatedimages", exist_ok=True)

    mask_directory="data/masks"
    directory_entries = os.listdir(mask_directory)


    #add all mask file names to list 
    # standard is base_mask_part_png
    #remove extension and split on mask to get base and part
    mask_filenames= []
    for filename in directory_entries:
        if filename.lower().endswith(".png"):
            mask_filenames.append(filename)

    mask_filenames.sort()


    for mask_filename in mask_filenames:
        namee = mask_filename[:-4]
        token="_mask_"

        if token not in namee:
            print("no proper mask naming")
            continue

        base_name,part_name = namee.split(token,1)

        images_directory = Path("data/images")

        #find matching image for mask
        original_path = None
        for extension in (".jpg", ".jpeg", ".png"):
            path = images_directory / (base_name+ extension)
            if path.exists():
                original_path= str(path)
                break
        if original_path is None:
            print("no matching image for mask")
            continue
        
        mask_path = os.path.join("data/masks", mask_filename)


        # the following part ( until prompt ) aims to fit mask and image to the required size for openai , does not work perfectly
        try:
            with Image.open(original_path) as original_image, Image.open(mask_path) as mask_image:
                original_width, original_height = original_image.size

                if mask_image.size != (original_width, original_height):
                    print("mask and image not same size")
                    continue
                
                scale = min(1024 / original_width, 1024 / original_height)
                
            
                canvas_image = Image.new("RGB", (1024, 1024), (255, 255, 255))
                new_original = original_image.resize((int(round(original_width * scale)), int(round(original_height * scale))), Image.BICUBIC)
                canvas_image.paste(new_original, ((1024 - int(round(original_width * scale)))// 2, (1024 - int(round(original_height * scale)))// 2))

                canvas_mask = Image.new("RGBA", (1024, 1024), (0, 0, 0, 255))
                new_mask = mask_image.resize((int(round(original_width * scale)), int(round(original_height * scale))), Image.NEAREST)
                canvas_mask.paste(new_mask, ((1024 - int(round(original_width * scale)))// 2, (1024 - int(round(original_height * scale)))// 2))

                
                
            prompt = (
                f"Remove the bicycle part(s) '{part_name}' in the masked region and fill in the background naturally. "
                "Do not change anything outside the masked region. "
                "Maintain the original perspective, lighting, and textures."
            )
            
            

            with tempfile.NamedTemporaryFile(suffix=".png") as tmp_image, tempfile.NamedTemporaryFile(suffix=".png") as tmp_mask:
                canvas_image.save(tmp_image.name)
                canvas_mask.save(tmp_mask.name)

                with open(tmp_image.name, "rb") as image_file, open(tmp_mask.name, "rb") as mask_file:
                    result = client.images.edit(model=model,image=image_file,mask=mask_file,prompt=prompt,size="1024x1024",)



            # decode output
            base64_payload = result.data[0].b64_json
            edited_bytes = base64.b64decode(base64_payload)
            edited_full = Image.open(io.BytesIO(edited_bytes)).convert("RGB")

            # crop back to original aspect and size
            edited_region = edited_full.crop(((1024 - int(round(original_width * scale)))// 2, (1024 - int(round(original_height * scale)))// 2, (1024 - int(round(original_width * scale)))// 2 + int(round(original_width * scale)), (1024 - int(round(original_height * scale)))// 2 + int(round(original_height * scale))))
            edited_back_to_original = edited_region.resize((original_width, original_height), Image.BICUBIC)

            # keep the original extension for later (easier to replace )
            original_extension = Path(original_path).suffix or ".jpg"
            output_name = f"{base_name}_removed_{part_name}{original_extension}"
            output_path = os.path.join("data/generatedimages", output_name)

            # save
            if original_extension.lower() in (".jpg", ".jpeg"):
                edited_back_to_original.save(output_path, quality=95)
            else:
                edited_back_to_original.save(output_path)


            
            
            
        except Exception as error:
            print(f"following error for file{mask_filename}:{error}")
    
    
    print("all images created")


if __name__ == "__main__":
    main()
