import docx2txt
import fitz  # PyMuPDF

def extract_text_from_pdf(file):
    pdf_text = ""
    doc = fitz.open(stream=file.read(), filetype="pdf")
    for page in doc:
        pdf_text += page.get_text()
    return pdf_text

def extract_text_from_docx(file):
    return docx2txt.process(file)

def extract_text_from_txt(file):
    return file.read().decode("utf-8")

def load_documents(files):
    merged_text = ""
    for file in files:
        filename = file.name.lower()
        if filename.endswith(".pdf"):
            merged_text += extract_text_from_pdf(file)
        elif filename.endswith(".docx"):
            merged_text += extract_text_from_docx(file)
        elif filename.endswith(".txt"):
            merged_text += extract_text_from_txt(file)
        merged_text += "\n"
    return merged_text
