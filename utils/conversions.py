
def convert_pixel_distance_to_meters(pixel_distance, refrence_height_meters, reference_height_pixels):
    return (pixel_distance * refrence_height_meters) / reference_height_pixels


def convert_meters_to_pixel_distances(meters, reference_height_meters , reference_height_pixels):
    return (meters * reference_height_pixels) / reference_height_meters
