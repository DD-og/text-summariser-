# text-summariser-

# LLAMA Multi-Model Document Summarizer

This Streamlit application provides a multi-model document summarization tool using various LLAMA models through the Groq API.

## Setup

1. Clone the repository:
   ```
   git clone <repository-url>
   cd <repository-name>
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   ```

3. Activate the virtual environment:
   - On Windows:
     ```
     venv\Scripts\activate
     ```
   - On macOS and Linux:
     ```
     source venv/bin/activate
     ```

4. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

5. Create a `config.json` file in the project root with your Groq API key:
   ```json
   {
     "GROQ_API_KEY": "your-api-key-here"
   }
   ```

## Running the Application

1. Ensure your virtual environment is activated.

2. Run the Streamlit app:
   ```
   streamlit run dd.py
   ```

3. Open your web browser and navigate to the URL provided by Streamlit (usually http://localhost:8501).

## Features

- Text input or file upload (PDF/DOCX) for summarization
- Multiple LLAMA model options for summarization
- Adjustable summary length (word count or percentage)
- Interactive summary view with original text comparison
- Download summaries in PDF, DOCX, or HTML formats
- Summary history and side-by-side model comparison

## Requirements

See `requirements.txt` for a list of required Python packages.

## License

[Specify your license here]

