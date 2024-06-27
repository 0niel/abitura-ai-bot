from typing import Callable, List, Optional
from urllib.parse import urljoin

import html2text
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager


class SeleniumRecursiveLoader:
    """
    A class to recursively load web pages using Selenium and extract their content.

    Attributes:
        urls (List[str]): Initial list of URLs to start loading.
        base_url (str): The base URL for normalizing relative links.
        page_ready_check (Optional[Callable[[webdriver.Chrome], bool]]): An optional function to check if the page is ready.
        page_extractor (Callable[[str], str]): A function to extract content from the page's raw HTML.
        exclude_urls (List[str]): List of URLs to exclude from loading.
        headless (bool): Flag to run the browser in headless mode.
    """

    def __init__(
        self,
        urls: List[str],
        base_url: str,
        page_extractor: Callable[[str], str],
        page_ready_check: Optional[Callable[[webdriver.Chrome], bool]] = None,
        exclude_urls: List[str] = [],
        headless: bool = True,
        max_depth: int = 2,
        user_agent: str = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.103 Safari/537.36",
    ):
        """
        Initializes the SeleniumRecursiveLoader with the provided parameters.

        Args:
            urls (List[str]): Initial list of URLs to start loading.
            base_url (str): The base URL for normalizing relative links.
            page_extractor (Callable[[str], str]): A function to extract content from the page's raw HTML.
            page_ready_check (Optional[Callable[[webdriver.Chrome], bool]], optional): An optional function to check if the page is ready. Defaults to None.
            exclude_urls (List[str], optional): List of URLs to exclude from loading. Defaults to [].
            headless (bool, optional): Flag to run the browser in headless mode. Defaults to True.
            max_depth (int, optional): Maximum depth for recursive loading. Defaults to 2.
            user_agent (str, optional): Custom user agent string for the browser. Defaults to a standard user agent.
        """
        self._urls = urls
        self._base_url = base_url.rstrip("/")
        self._page_ready_check = page_ready_check or (lambda d: True)
        self._page_extractor = page_extractor
        self._exclude_urls = [url.rstrip("/") for url in exclude_urls]
        self._headless = headless
        self._max_depth = max_depth
        self._user_agent = user_agent
        self._driver = None
        self._visited_urls = set()

    def _init_driver(self) -> webdriver.Chrome:
        """Initializes the Selenium Chrome driver with specified options."""
        options = webdriver.ChromeOptions()
        if self._headless:
            options.add_argument("--headless")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument(f"user-agent={self._user_agent}")
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option("useAutomationExtension", False)
        driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options)
        return driver

    def load(self) -> List[Document]:
        """
        Starts the loading process for all initial URLs and returns the extracted documents.

        Returns:
            List[Document]: A list of Document objects with extracted content.
        """
        self._driver = self._init_driver()
        documents = []
        for url in self._urls:
            documents.extend(self._load_url_recursive(url))
        self.close()
        return documents

    def _load_url_recursive(self, url: str, depth: int = 0) -> List[Document]:
        """
        Recursively loads a URL and extracts its content and linked pages.

        Args:
            url (str): The URL to load.
            depth (int, optional): Current recursion depth. Defaults to 0.

        Returns:
            List[Document]: A list of Document objects with extracted content.
        """
        WebDriverWait(self._driver, 10).until(lambda d: not self._page_ready_check(d))
        if (
            depth > self._max_depth
            or not url.startswith(self._base_url)
            or any(url.startswith(exclude) for exclude in self._exclude_urls)
            or url in self._visited_urls
        ):
            return []

        self._visited_urls.add(url)

        self._driver.get(url)
        WebDriverWait(self._driver, 10).until(lambda d: not self._page_ready_check(d))
        raw_html = self._driver.page_source
        content = self._page_extractor(raw_html)
        documents = [Document(page_content=content, metadata={"source": url})]

        soup = BeautifulSoup(raw_html, "html.parser")
        links = set(self._normalize_url(a["href"]) for a in soup.find_all("a", href=True))
        for link in links:
            if link:
                documents.extend(self._load_url_recursive(link, depth + 1))

        return documents

    def _normalize_url(self, url: str) -> str:
        """
        Normalizes a URL to be absolute based on the base URL.

        Args:
            url (str): The URL to normalize.

        Returns:
            str: The normalized URL.
        """
        if url.startswith("/"):
            return urljoin(self._base_url, url)
        if url.startswith(self._base_url):
            return url.rstrip("/")
        return ""

    def close(self):
        """Closes the Selenium WebDriver."""
        self._driver.quit()


def default_page_ready_check(driver: webdriver.Chrome) -> bool:
    """
    Checks if the page is ready based on specific text elements.

    Args:
        driver (webdriver.Chrome): The Selenium WebDriver instance.

    Returns:
        bool: True if the page is ready, False otherwise.
    """
    try:
        loading_text = driver.find_element(By.XPATH, "//*[contains(text(), 'Пожалуйста, подождите! Идет загрузка...')]")
        ddos_guard_text = "Проверка браузера перед переходом" in driver.page_source
        return loading_text is not None or ddos_guard_text is not None
    except Exception:
        return False


def default_page_extractor(html: str) -> str:
    """
    Extracts and converts the HTML content to text.

    Args:
        html (str): Raw HTML content.

    Returns:
        str: Extracted text content.
    """
    soup = BeautifulSoup(html, "html.parser")
    return html2text.html2text(soup.prettify()).strip()
