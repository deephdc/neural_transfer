import neural_transfer.config as cfg
import os
from PIL import Image
from fpdf import FPDF
        
        
# Input and result images from the segmentation
files = ['{}/content_image.png'.format(cfg.DATA_DIR),
         '{}/result_image.png'.format(cfg.DATA_DIR)]

# Merge images and add a color legend
def merge_images():
    for path in files:
        img = Image.open(path)
        img.thumbnail((310, 210), Image.ANTIALIAS)
        img.save(path)


# Put images and accuracy information together in one pdf file
def create_pdf():
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Results:', ln=1)
    pdf.set_font('Arial', '',12)
    pdf.cell(0, 10, 'Original image:', ln=1)
    pdf.image(files[0], 10, 30)
    
    pdf.cell(0, 170, 'Result image:', ln=1)
    pdf.image(files[1], 10, 120)
    
    results = '{}/prediction_results.pdf'.format(cfg.DATA_DIR)
    pdf.output(results,'F')
    return results