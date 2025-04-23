from transformers import CLIPProcessor, CLIPModel

from PIL import Image

def test_clip_installation():
    try:
        # Load the CLIP model and processor
        clip_model_name = "openai/clip-vit-base-patch32"
        processor = CLIPProcessor.from_pretrained(clip_model_name)
        clip_model = CLIPModel.from_pretrained(clip_model_name)

        # Prepare dummy text and a dummy image
        dummy_text = ["a photo of a cat"]
        dummy_image = Image.new("RGB", (224, 224))  # Create a dummy image (you can use a real image if you prefer)

        # Test with dummy data (text and dummy image)
        inputs = processor(text=dummy_text, images=dummy_image, return_tensors="pt", padding=True)
        outputs = clip_model(**inputs)

        # Print the output to verify it's working
        print("CLIP model test successful.")

    except Exception as e:
        print(f"An error occurred while loading or using the CLIP model: {e}")

