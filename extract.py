import zipfile
import re
import sys

def extract_docx(docx_path, out_path):
    z = zipfile.ZipFile(docx_path)
    xml = z.read('word/document.xml').decode('utf-8')
    z.close()
    # Replace paragraph and line break tags with newlines
    text = xml.replace('</w:p>', '\n')
    text = xml.replace('<w:br/>', '\n')
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[ \t]+', ' ', text)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"Extracted {out_path}")

extract_docx(r'c:\Users\hamza\Downloads\rag project\plan.docx', r'c:\Users\hamza\Downloads\rag project\plan_out.txt')
extract_docx(r'c:\Users\hamza\Downloads\rag project\actual problem.docx', r'c:\Users\hamza\Downloads\rag project\problem_out.txt')
print("ALL DONE")
