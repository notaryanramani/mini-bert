# Mini-BERT: A Miniaturized Implementation of BERT (Bidirectional Encoder Representations from Transformers)

Mini-BERT is a simplified implementation of BERT (Bidirectional Encoder Representations from Transformers) based on the original [BERT paper](https://arxiv.org/pdf/1810.04805.pdf) from 2019. It provides a lightweight and easy-to-understand version of BERT for educational and experimental purposes.

## Features

- **Simplified Architecture**: Mini-BERT follows the core principles of BERT while minimizing complexity.
- **Easy to Understand**: Designed to be accessible for educational purposes, with clear code and documentation.
- **Minimal Dependencies**: Requires only basic libraries such as PyTorch and NumPy.
- **Flexible**: Easy to extend and customize for different experiments and applications.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/notaryanramani/mini-bert.git
   ```

2. Install the dependencies for MacOS.
    ```
    pip install -r requirements.txt
    ```

    *Note* - Please refer [PyTorch documentation](https://pytorch.org/get-started/locally/) to install PyTorch for you operating system

## WikipediaScraper 

WikipediaScraper is a Python class designed to scrape random pages from Wikipedia and save the scraped text content to a text file.

### Usage

1. **Example Snippet**: Import the class, instantiate an object and call the `scrap_pages()` method.
    ```python
    from wikipedia_scraper import WikipediaScraper
    scraper = WikipediaScraper()
    scraper.scrap_pages(n_pages=100)
    ```

2. **View Saved Data**: View the saved text data in the specified text file.
    ```bash
    cat datakit/data/data.txt
    ```

## Contribution

Contributions are welcome! If you find any bugs or have suggestions for improvement, please open an issue or submit a pull request.



