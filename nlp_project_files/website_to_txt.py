"""
filename: website_to_txt.py

converts online articles into text files
"""

import requests
from bs4 import BeautifulSoup

def save_website_text(url, output_file):
    """
    Fetches and saves the textual content of a website (paragraphs) to a file.

    Parameters:
        url (str): The URL of the website to fetch content from.
        output_file (str): The name of the file to save the extracted text. Default is 'website_text.txt'.

    Returns:
        None
    """
    try:
        # Fetch the webpage content
        response = requests.get(url)
        response.raise_for_status()  # Check for request errors

        # Parse the HTML content
        soup = BeautifulSoup(response.text, "html.parser")

        # Extract all text from <p> tags (paragraphs)
        paragraphs = soup.find_all('p')
        text_content = "\n".join(paragraph.get_text(strip=True) for paragraph in paragraphs)

        # Save the extracted text to a file
        with open(output_file, "w", encoding="utf-8") as file:
            file.write(text_content)

        print(f"Website text saved successfully to '{output_file}'!")
    except requests.exceptions.RequestException as e:
        print(f"An error occurred while fetching the website: {e}")

# Example usage:
save_website_text("https://www.cnbc.com/2024/09/06/climate-crisis-world-logs-its-hottest-summer-on-record.html",
                  "center_cnbc.txt")

