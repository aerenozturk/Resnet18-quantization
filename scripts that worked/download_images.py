from bing_image_downloader import downloader

# Download images using the downloader function
downloader.download("cat, dog, car, airplane, bicycle", 
                    limit=100, 
                    output_dir='./dataset', 
                    adult_filter_off=True, 
                    force_replace=False, 
                    timeout=60, 
                    verbose=True)
