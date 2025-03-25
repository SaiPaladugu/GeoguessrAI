import os
import random
import requests
import numpy as np
from PIL import Image, ImageStat
from io import BytesIO
from dotenv import load_dotenv

load_dotenv()

# Google Street View API key
API_KEY = os.getenv("GOOGLE_MAPS_API")

# Pictures per city
TOTAL_PICTURES = 200

# City boundaries (approximate lat/lon ranges)
CITY_BOUNDS = {
    "Ottawa": [(45.30, 45.50), (-75.85, -75.55)],  # Lat: 45.30 - 45.50, Lon: -75.85 - -75.55
    "Dubai": [(25.10, 25.40), (55.20, 55.50)],  # Lat: 25.10 - 25.40, Lon: 55.20 - 55.50
    "Tokyo": [(35.60, 35.80), (139.60, 139.90)]  # Lat: 35.60 - 35.80, Lon: 139.60 - 139.90
}

# Image folder path
BASE_FOLDER = "street_view_images"

# Function to detect "Sorry, No Imagery Here" images
def is_no_imagery_image(image, brightness_threshold=220, stddev_threshold=15, unique_colors_threshold=1000):
    """Detects if an image is an error screen from Google Street View API."""
    grayscale = image.convert("L")
    stat = ImageStat.Stat(grayscale)
    
    avg_brightness = stat.mean[0]
    brightness_stddev = stat.stddev[0]
    unique_colors = len(set(image.getdata()))

    # Print debugging info
    print(f"📌 Checking image eligibility:")
    print(f" - Avg Brightness: {avg_brightness:.2f} (Threshold: {brightness_threshold})")
    print(f" - Brightness Std Dev: {brightness_stddev:.2f} (Threshold: {stddev_threshold})")
    print(f" - Unique Colors: {unique_colors} (Threshold: {unique_colors_threshold})")

    # Check against defined thresholds
    return avg_brightness > brightness_threshold and brightness_stddev < stddev_threshold and unique_colors < unique_colors_threshold

# Function to fetch and save Street View images
def fetch_street_view(api_key, city, location, heading=0, pitch=0, fov=90, image_size=(640, 640)):
    base_url = "https://maps.googleapis.com/maps/api/streetview"
    params = {
        'size': f"{image_size[0]}x{image_size[1]}",
        'location': f"{location[0]},{location[1]}",
        'heading': heading,
        'pitch': pitch,
        'fov': fov,
        'key': api_key
    }
    
    response = requests.get(base_url, params=params)
    
    if response.status_code == 200:
        image = Image.open(BytesIO(response.content))
        
        # Check if it's a "No Imagery" error image
        if is_no_imagery_image(image):
            print(f"[⚠] Skipped invalid image for {city}: {location}")
            return False  # Skip this image
        
        # Save image in city-specific folder
        city_folder = os.path.join(BASE_FOLDER, city)
        os.makedirs(city_folder, exist_ok=True)
        
        filename = f"{location[0]}_{location[1]}.jpg"
        image.save(os.path.join(city_folder, filename))
        print(f"[✔] Saved image for {city}: {location}")
        return True  # Successful save
    else:
        print(f"[✖] Failed to fetch image for {city}: {location}, Status: {response.status_code}")
        return False  # Failed fetch

# Function to generate random lat/lon within city bounds
def generate_random_location(city):
    lat_min, lat_max = CITY_BOUNDS[city][0]
    lon_min, lon_max = CITY_BOUNDS[city][1]
    return (random.uniform(lat_min, lat_max), random.uniform(lon_min, lon_max))

# Function to delete all images inside Ottawa, Tokyo, and Dubai folders
def delete_all_images():
    """
    Deletes all images inside the Ottawa, Tokyo, and Dubai folders.
    This function does NOT delete the folders themselves.
    """
    for city in CITY_BOUNDS.keys():
        city_folder = os.path.join(BASE_FOLDER, city)
        if os.path.exists(city_folder):
            for filename in os.listdir(city_folder):
                file_path = os.path.join(city_folder, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            print(f"[🗑] Deleted all images in {city_folder}")
        else:
            print(f"[⚠] Folder {city_folder} does not exist.")

# Main script to generate and save images
if __name__ == "__main__":
    for city in CITY_BOUNDS.keys():
        count = 0
        while count < TOTAL_PICTURES - 1:  # Ensure 50 successful
            location = generate_random_location(city)
            if fetch_street_view(API_KEY, city, location):
                count += 1  # Only count successful requests
                
# delete_all_images()