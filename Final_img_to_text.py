from PIL import Image, ImageEnhance, ImageFilter
from pytesseract import pytesseract
import enum
import os


class OS(enum.Enum):
   # Mac = 0
    Windows = 1


class Language(enum.Enum):
    ENG = 'eng'
    MAR = 'mar'
    ENG_MAR = 'eng+mar'


class ImageReader:
    def __init__(self, os: OS):
        # Tesseract
        #if os == OS.Mac:
         #   print('Running on: MAC\n')
        if os == OS.Windows:
            windows_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
            pytesseract.tesseract_cmd = windows_path
            print('Running on: WINDOWS\n')

    def preprocess_image(self, image: Image) -> Image:
        # Apply preprocessing steps to enhance OCR accuracy
        image = image.convert('L')  # Convert to grayscale
        image = image.filter(ImageFilter.SHARPEN)  # Sharpen the image
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2)  # Increase contrast
        image = image.point(lambda x: 0 if x < 128 else 255, '1')  # Thresholding

        return image

    def extract_text(self, image: str, lang: Language) -> str:
        img = Image.open(image)
        img = self.preprocess_image(img)  # Preprocess the image
        extracted_text = pytesseract.image_to_string(img, lang=lang.value)
        return extracted_text

    def process_images_from_folder(self, folder_path: str, lang: Language, output_folder: str):
        # Create the output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)

        # Iterate over the images in the folder
        for filename in os.listdir(folder_path):
            if filename.endswith('.png') or filename.endswith('.jpg'):
                image_path = os.path.join(folder_path, filename)

                # Extract text from the image
                extracted_text = self.extract_text(image_path, lang)

                # Process the text
                processed_text = ' '.join(extracted_text.split())

                # Print the extracted text
                print(f"Image: {filename}")
                print("Extracted Text:")
                print(processed_text)
                print("-------------------")

                # Create the output text file path
                output_file_path = os.path.join(output_folder, os.path.splitext(filename)[0] + '.txt')

                # Save the processed text to the output text file
                with open(output_file_path, 'w', encoding='utf-8') as file:
                    file.write(processed_text)

                print(f"Processed image: {filename}. Extracted text saved to: {output_file_path}")

                # Display the extracted text
                #print("Displaying extracted text:")
                #print(processed_text)
                print("-------------------")

                # Wait for user input before proceeding to the next image
                input("Press Enter to continue...")


if __name__ == '__main__':
    ir = ImageReader(OS.Windows)
    image_folder = 'C:/Users/ODA/PycharmProjects/TradeMark_Final_files/images_24'
    output_folder = 'C:/Users/ODA/PycharmProjects/TreadMark/Img_to_text_24'
    ir.process_images_from_folder(image_folder, lang=Language.ENG_MAR, output_folder=output_folder)
