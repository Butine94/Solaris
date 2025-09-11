from PIL import Image, ImageDraw, ImageFont

# Test 1: Basic colored rectangle
img = Image.new("RGB", (640, 480), (255, 0, 0))  # Red
img.save("test_red.png")

# Test 2: Add white text
img = Image.new("RGB", (640, 480), (50, 50, 50))  # Dark gray
draw = ImageDraw.Draw(img)
font = ImageFont.load_default()
draw.text((50, 50), "TEST TEXT", fill=(255, 255, 255), font=font)
img.save("test_text.png")

print("Check test_red.png and test_text.png")


if __name__ == "__main__":
    main()