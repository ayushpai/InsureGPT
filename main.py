from fpdf import FPDF
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import OpenAIEmbeddings
from openai import OpenAI
import os
import base64
from langchain_community.vectorstores.singlestoredb import SingleStoreDB

OpenAI.api_key = os.environ["OPENAI_API_KEY"]
os.environ["SINGLESTOREDB_URL"] = "<SINGLESTOREDB_URL>"


# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


client = OpenAI()


def analyze_image(base64_image, prompt):
    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        },
                    },
                ],
            }
        ],
        max_tokens=300,
    )
    response_text = response.choices[0].message.content
    return response_text


def analyze_document(base64_image):
    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "system",
                "content": "You are a scribe. Convert this image text to formatted text output. Output nothing else.",
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        },
                    },
                ],
            }
        ],
        max_tokens=300,
    )
    response_text = response.choices[0].message.content
    return response_text


def analyze_accident(accident_report, car_repair_details, insurance_claim):
    response = client.chat.completions.create(
        model="gpt-4-0125-preview",
        messages=[
                     {
                         "role": "system",
                         "content": """You are a car insurance agent. Your job is to analyze the following documents
                                    including a insurance claim document, a detailed description of the damage, 
                                    and a CSV of the car parts and prices. Your goal is to use this information and
                                    infer the type of damage and make a payout estimate based on the CSV.""",
                     },
                 ]
                 + [{"role": "assistant", "content": accident_report}]
                 + [{"role": "assistant", "content": car_repair_details}]
                 + [{"role": "assistant", "content": insurance_claim}]

    )

    response_text = response.choices[0].message.content
    return response_text

def create_pdf(text, filename="Accident_Analysis_Report.pdf"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, text)
    pdf.output(filename)


def main():
    car_accident_image = encode_image("car-accident.jpg")
    accident_report = analyze_image(car_accident_image, "Visually describe this car accident. Explain the damage on "
                                                        "the car. Return a detailed accident report")


    print(accident_report)

    car_repair_details = analyze_document(encode_image("car_repair_estimate.png"))
    print(car_repair_details)

    insurance_claim = analyze_document(encode_image("insurance_claim.png"))
    print(insurance_claim)

    final_accident_analysis = analyze_accident(accident_report, car_repair_details, insurance_claim)
    print(final_accident_analysis)
    # create_pdf(final_accident_analysis) # if you want to create a PDF out of the final accident analysis

    # store documents in vectorized format for RAG, quick search (ANN), etc.
    loader = TextLoader("insurance_claim.txt")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()

    db = SingleStoreDB.from_documents(
        docs,
        embeddings,
        table_name="insurance_claim",  # use table with a custom name
    )

if __name__ == "__main__":
    main()
