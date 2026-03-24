from PIL import Image

# Create a pure green image (1920x1080)
width, height = 1920, 1080
green = (0, 255, 0)  # RGB: pure green
image = Image.new('RGB', (width, height), green)

# Save to data folder
image.save('data/green_bg.png')
print("Green background image saved to data/green_bg.png")