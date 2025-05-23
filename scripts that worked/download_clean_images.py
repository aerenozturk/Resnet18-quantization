from bing_image_downloader import downloader

# Define the categories you want to download
categories = ["cat", "dog", "car", "airplane", "bicycle"]

# Loop through each category and download separately
for category in categories:
    downloader.download(category, 
                        limit=100, 
                        output_dir=f'./dataset/{category}', 
                        adult_filter_off=True, 
                        force_replace=False, 
                        timeout=60, 
                        verbose=True)

