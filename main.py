import json

from doctr.io import DocumentFile
from doctr.models import ocr_predictor


doc = DocumentFile.from_pdf("local_files/clg syllabus.pdf")
print(f"Number of pages: {len(doc)}")

predictor = ocr_predictor(pretrained=True)

print(predictor)


result = predictor(doc)

synthetic_pages = result.synthesize()

json_export = result.export()
with open('local_files/output.json', 'w') as f:
    json.dump(json_export, f)