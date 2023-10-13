![banner](./assets/logo_with_text.png)

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->

# OCR Components

Includes:

| No. | Name                                                                  | Description                       | Output Type | Input Type | Usage Example                                               |
| --- | --------------------------------------------------------------------- | --------------------------------- | ----------- | ---------- | ----------------------------------------------------------- |
| 1   | [ParsePdf](geniusrise_bolts/parse_pdf.py)                             | Classifies and processes PDFs     | Batch       | PDF        | [ParsePdf Usage](#parsepdf-usage)                           |
| 2   | [ConvertImage](geniusrise_bolts/convert_image.py)                     | Converts image formats            | Batch       | Image      | [ConvertImage Usage](#convertimage-usage)                   |
| 3   | [ImageClassPredictor](geniusrise_bolts/image_class_predictor.py)      | Classifies images into categories | Batch       | Image      | [ImageClassPredictor Usage](#imageclasspredictor-usage)     |
| 4   | [TrainImageClassifier](geniusrise_bolts/train_image_classifier.py)    | Trains an image classifier        | Batch       | Image      | [TrainImageClassifier Usage](#trainimageclassifier-usage)   |
| 5   | [TROCRImageOCR](geniusrise_bolts/trocr_image_ocr.py)                  | Performs OCR on images            | Batch       | Image      | [TROCRImageOCR Usage](#trocrimageocr-usage)                 |
| 6   | [FineTuneTROCR](geniusrise_bolts/fine_tune_trocr.py)                  | Fine-tunes TROCR model            | Batch       | Image      | [FineTuneTROCR Usage](#fine_tune_trocr-usage)               |
| 7   | [TROCRImageOCRAPI](geniusrise_bolts/trocr_image_ocr_api.py)           | OCR API using TROCR               | API         | Image      | [TROCRImageOCRAPI Usage](#trocrimageocrapi-usage)           |
| 8   | [Pix2StructImageOCR](geniusrise_bolts/pix2struct_image_ocr.py)        | OCR using Pix2Struct              | Batch       | Image      | [Pix2StructImageOCR Usage](#pix2structimageocr-usage)       |
| 9   | [Pix2StructImageOCRAPI](geniusrise_bolts/pix2struct_image_ocr_api.py) | OCR API using Pix2Struct          | API         | Image      | [Pix2StructImageOCRAPI Usage](#pix2structimageocrapi-usage) |
| 10  | [FineTunePix2Struct](geniusrise_bolts/fine_tune_pix2struct.py)        | Fine-tunes Pix2Struct model       | Batch       | Image      | [FineTunePix2Struct Usage](#fine_tune_pix2struct-usage)     |
