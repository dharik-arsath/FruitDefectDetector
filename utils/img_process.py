def get_image_size(img_size: str):
    hwc_list: list = img_size.split(",")
    height, width, channel = list(map(int, hwc_list))
    return height, width, channel