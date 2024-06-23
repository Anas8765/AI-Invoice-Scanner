from flask import Flask, request, jsonify
from transformers import pipeline
from pdf2image import convert_from_bytes
from PIL import Image
import os
import tempfile

app = Flask(__name__)

# Initialize the pipeline for document-question-answering
pipe = pipeline("document-question-answering", model="impira/layoutlm-invoices")

def process_image(img):
    predefined_questions = [
        "Name of Company:",
        "Invoice Number:",
        "Invoice Date:",
        "Contact Number:",
        "Name of person the product is billed to:",
        "Billing Address Location:",
        "Name of person the product is shipped to:",
        "Shipping Address Location:",
        #"Due Date:",
        #"Products/Service:",
        "Total Amount:",
        "Tax included:",
        "Payment Method:"
    ]

    answers = {}
    for user_question in predefined_questions:
        answer = pipe(image=img, question=user_question)
        extracted_answer = answer[0]['answer'] if len(answer) > 0 else "N/A"
        answers[user_question] = extracted_answer
    return answers

@app.route('/process_file', methods=['POST'])
def process_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    uploaded_file = request.files['file']
    
    if uploaded_file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    _, file_extension = os.path.splitext(uploaded_file.filename)
    allowed_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.pdf'}
    
    if file_extension.lower() not in allowed_extensions:
        return jsonify({'error': 'Unsupported file format. Please upload a PDF or an image.'})
    
    try:
        if file_extension.lower() in {'.png', '.jpg', '.jpeg', '.gif', '.bmp'}:
            img = Image.open(uploaded_file)
            answers = process_image(img)
        else:  # Process PDF
            with tempfile.NamedTemporaryFile(delete=False) as temp_pdf:
                temp_pdf.write(uploaded_file.read())
                images = convert_from_bytes(open(temp_pdf.name, 'rb').read())
                temp_pdf.close()
                os.unlink(temp_pdf.name)

            answers = {}
            for idx, img in enumerate(images):
                answers[f'Page {idx + 1}'] = process_image(img)
        
        return jsonify({'answers': answers})
    except Exception as e:
        return jsonify({'error': f'Error processing file: {str(e)}'})

if __name__ == '__main__':
    app.run(debug=True)