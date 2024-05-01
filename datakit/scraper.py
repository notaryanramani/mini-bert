from bs4 import BeautifulSoup
import requests
import warnings
import os
import re
from utils import CustomLogger
from tqdm import tqdm

class WikipediaScraper:
    def __init__(self, logfile='logs/scraper_logs.log'):
        self.filepath = None
        self.logger = CustomLogger(__name__, logfile)

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

        print('Scraping Started...')
        self.logger.info(f'Scraping {n_pages} pages, filepath: {filepath}')
        data = []
        l = tqdm(range(n_pages), ncols=100)
        for i in l:
            text = self.scrap_one_page()
            text = re.sub(r'\s+', ' ', text)
            data.append(text)
            l.set_postfix(text_len = len(text))
            if (i+1) % (n_pages // 10) == 0:
                self.logger.info(f'Scraped Pages: {i+1}')
        data = '<|endoftext|>'.join(data)
        data = re.sub(r'\[\d+\]', '', data)
        
        self.filepath = filepath
        self.save(filepath, data)
        print('Scraping Completed. Check logs for more details.')

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

        self.logger.info(f'Saving file containing {len(text)} characters at {filepath}')
        try:
            with open(f'{filepath}', 'w', encoding='utf-8') as f:
                f.write(text)
            
        except UnicodeEncodeError as e:
            print(f"Error occurred while encoding: {e}")
            cleaned_text = ''.join([char for char in text if ord(char) < 128])
            with open('file.txt', 'w', encoding='utf-8') as file:
                file.write(cleaned_text)


    def __check_path(self, filepath):
        dir = os.path.dirname(filepath)
        if not os.path.exists(dir) and dir != '':
            self.logger.info(f"Directory doesnot exists. Creating {dir} path...")
            os.makedirs(dir)
        else:
            self.logger.info('Input path exists. Starting scraping...')


if __name__ == '__main__':
    scraper = WikipediaScraper()
    scraper.scrap_pages(n_pages=50)

        