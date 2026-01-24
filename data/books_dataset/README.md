📥 Books Dataset Setup

This directory contains the raw books dataset sourced from Kaggle.

Due to its large size (~1.16 GB), the dataset is not included in the GitHub repository and must be downloaded manually.

📚 Dataset Source

Platform: Kaggle

Dataset: Books Dataset

Approximate Size: ~1.16 GB

License: As specified on Kaggle

Please ensure you comply with the dataset’s license before use.

⬇️ How to Download the Dataset
Step 1: Kaggle Account

Create an account at https://www.kaggle.com/
 if you don’t already have one.

Step 2: Download

Open the dataset page on Kaggle

Click Download

Extract the downloaded archive

Step 3: Place Files in the Correct Location

After extraction, ensure the dataset files are placed exactly here:

data/books_dataset/


The directory structure should look like this:

data/books_dataset/
├── authors.csv
├── categories.csv
├── dataset.csv
├── formats.csv
├── places.csv
└── README.md


⚠️ Do not rename these files unless you also update the ingestion scripts.

📄 File Descriptions

dataset.csv
Core book-level data (titles, descriptions, language, publication info)

authors.csv
Author metadata linked to books

categories.csv
Raw category / genre labels provided by the source (may be noisy)

formats.csv
Information about book formats (hardcover, paperback, digital, etc.)

places.csv
Location-related metadata (publication or contextual data)

These files are treated as raw inputs and are not modified directly.

🚫 Git Ignore Notice

This directory is intentionally ignored by Git to avoid committing large files:

data/books_dataset/


This is standard practice for data-heavy projects.

🔁 How This Dataset Is Used

Files in this directory are consumed by scripts in:

src/ingestion/


The data is cleaned, deduplicated, and filtered before further processing

LLM-based enrichment (genres, descriptions) happens after this stage

❗ Troubleshooting

Files not detected:
Confirm all CSVs are inside data/books_dataset/

Pipeline errors:
Ensure filenames match exactly as shown above

Permission issues:
Check that files are readable by your scripts

📌 Why the Dataset Is Not Included in the Repo

File size exceeds GitHub limits

Licensing considerations

Keeps the repository lightweight and easy to clone

This follows standard industry and open-source practices.

📬 Notes for Reviewers

A small, cleaned sample dataset is provided elsewhere in the repository to demonstrate schema and processing without requiring the full dataset.