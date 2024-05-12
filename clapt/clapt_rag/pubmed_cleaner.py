# Path to the original dataset
input_file_path = "/data/clapt/eval/pubmed/corpus_pubtator.txt"
# Path for the cleaned dataset
output_file_path = "/data/clapt/eval/pubmed/filtered_corpus.txt"


# Function to check if a line contains title or abstract
def is_title_or_abstract(line):
    return "|t|" in line or "|a|" in line


# Function to clean the text (optional, adjust according to your needs)
def clean_text(text):
    # Example: Replace unwanted characters, strip leading/trailing spaces
    cleaned_text = (
        text.replace("|t|", ": Title: ").replace("|a|", ": Abstract: ").strip()
    )
    return cleaned_text


# Processing the file
with open(input_file_path, "r", encoding="utf-8") as file, open(
    output_file_path, "w", encoding="utf-8"
) as output_file:
    for line in file:
        if is_title_or_abstract(line):
            # Clean the text before writing to output
            cleaned_line = clean_text(line)
            output_file.write(cleaned_line + "\n")

print(f"Filtered data written to {output_file_path}")
