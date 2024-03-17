from bs4 import BeautifulSoup
import requests
import warnings

class WikipediaScraper:
    def __init__(self):
        pass

    def scrap_pages(self, n_pages = 1000, filename = 'data.txt'):

        """
        Scrapes random Wikipedia pages and saves the content to a text file.

        Parameters:
        - n_pages (int): The number of Wikipedia pages to scrape. Default is 1000.
        """
        if not filename.endswith('.txt'):
            warnings.warn("Filename does not ends with '.txt'. We suggest you saving it with a '.txt' extension", category=Warning)

        data = []
        for i in range(n_pages):
            text = self.scrap_one_page()
            data.append(text)
            if (i+1) % (n_pages // 10) == 0:
                print(f'Scraped Page {i+1}')
        data = '\n'.join(data)
        self.save(filename, data)

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

    def save(self, filename, text):

        """
        Saves text content to a text file.

        Parameters:
        - filename (str): The name of the text file to save.
        - text (str): The text content to save.
        """

        print(f'Saving file containing {len(text)} characters')
        with open(f'datakit/data/{filename}', 'w') as f:
            f.write(text)


if __name__ == '__main__':
    scraper = WikipediaScraper()
    print('Scrapping pages...')
    scraper.scrap_pages()
    print('Finished Scrapping...')

        