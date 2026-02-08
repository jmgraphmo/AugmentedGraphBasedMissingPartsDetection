import os
import json
import random
from PIL import Image, ImageDraw

#names from annotations 
bike_parts=["front_light","back_pedal","lock","saddle","back_light","front_wheel","back_mudguard","front_pedal","back_wheel","back_hand_break","chain","front_handbreak","bell","dress_guard","kickstand","front_mudguard","back_reflector","steer","front_handle","gear_case","back_handle","dynamo"]


#create rgba mask , transparent rectangle where boundingbox is
def create_mask(width, height, bbox):
   
   
    mask = Image.new("RGBA", (width, height), (0, 0, 0, 255))
    draw = ImageDraw.Draw(mask)
    draw.rectangle([bbox["left"],bbox["top"], bbox["left"]+bbox["width"],bbox["top"]+bbox["height"]], fill=(0, 0, 0, 0))
    return mask

def main():
    

    os.makedirs("data/masks", exist_ok=True)

    #seed used by sakr
    random.seed(42)

    with open("data/processed/final_annotations_without_occluded.json") as f :
        annotations= json.load(f)
    images_dict = annotations["images"]

    #file for the names of testset files
    with  open("data/splits/test_set.txt", "r") as f :
        testset_files= [line.strip() for line in f if line.strip()]

    random.shuffle(testset_files) 


    #go through test set , try to create a mask for each part if part is available
    #stop at 10 created masks and go to next part
    #could go outside amount of files , needs to be manually checked
    i = 0 

    for target_part in bike_parts:
        created = 0

        while created < 10 and i < len(testset_files):
            image_name = testset_files[i]
            i= i+1
            image_info = images_dict.get(image_name)
            if image_info is None:
                continue

            image_path = os.path.join("data/images_vlm", image_name)
            if not os.path.exists(image_path):
                continue

            bbox = None
            for part in image_info.get("available_parts",[]):
                part_name = part.get("part_name","").strip().lower().replace(" ", "_")
                if part_name == target_part:
                    bbox= part["absolute_bounding_box"]
                    break
            if bbox is None:
                continue

            with Image.open(image_path) as image:
                width, height = image.size
                
            mask = create_mask(width, height, bbox)

            #create name for vlm to identify which part should be removed.
            base, _ = os.path.splitext(image_name)
            mask_filename = f"{base}_mask_{target_part}.png"
            mask.save(os.path.join("data/masks", mask_filename))

            created = created+1

        print(f"Created masks for {target_part} at index{i}/{len(testset_files)}")

    print("Masks saved to: data/masks")

if __name__ == "__main__":
    main()
