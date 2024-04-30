from bs4 import BeautifulSoup
import requests
import warnings
import os
import re

class WikipediaScraper:
    def __init__(self):
        self.filepath = None

    def scrap_pages(self, n_pages = 1000, filepath = 'data/data.txt'):

        """
        Scrapes random Wikipedia pages and saves the content to a text file.

        Parameters:
        - n_pages (int): The number of Wikipedia pages to scrape. Default is 1000.
        - filepath (str): Path where the file is to be saved
        """
        if not filepath.endswith('.txt'):
            warnings.warn("Filename does not ends with '.txt'. We suggest you saving it with a '.txt' extension", category=Warning)

        self.__check_path(filepath)

        print('Scrapping started...')
        data = []
        for i in range(n_pages):
            text = self.scrap_one_page()
            data.append(text)
            if (i+1) % (n_pages // 10) == 0:
                print(f'Scraped Pages: {i+1}')
        data = '\n'.join(data)

        data = re.sub(r'\s+', ' ', data)
        data = re.sub(r'\n+', '\n', data)

        self.filepath = filepath
        self.save(filepath, data)

    def get_file_path(self):
        return self.filepath

    def scrap_one_page(self):

        """
        Scrapes content from a random Wikipedia page.

        Returns:
        - text (str): The scraped text content from the page.
        """

        page_url = 'https://en.wikipedia.org/wiki/Special:Random'

        try:
            response = requests.get(page_url)
            if response.status_code == 200:
                content = response.content
                soup = BeautifulSoup(content, 'html.parser')
                main_content = soup.find('div', {
                    'id' : 'mw-content-text'
                })
                if main_content:
                    paragraphs = main_content.find_all('p')
                    paragraphs_list = [p.get_text() for p in paragraphs]
                    text = ''.join(paragraphs_list)
                    text = text.strip()
                    return text

        except Exception as e:
            print(f'Error occured: {e}')
        
        return ''

    def save(self, filepath, text):

        """
        Saves text content to a text file.

        Parameters:
        - filepath (str): The name of the text file to save.
        - text (str): The text content to save.
        """

        print(f'Saving file containing {len(text)} characters')
        try:
            with open(f'{filepath}', 'w', encoding='utf-8') as f:
                f.write(text)
            
        except UnicodeEncodeError as e:
            print(f"Error occurred while encoding: {e}")
            cleaned_text = ''.join([char for char in text if ord(char) < 128])
            with open('file.txt', 'w', encoding='utf-8') as file:
                file.write(cleaned_text)


    def __check_path(self, filepath):
        print(os.getcwd())
        dir = os.path.dirname(filepath)
        print(dir)
        if not os.path.exists(dir) and dir != '':
            print(f"Directory doesnot exists. Creating {dir} path...")
            os.makedirs(dir)
        else:
            print('Path exists...')


if __name__ == '__main__':
    scraper = WikipediaScraper()
    print('Scrapping pages...')
    scraper.scrap_pages(n_pages=50)
    print('Finished Scrapping...')

        