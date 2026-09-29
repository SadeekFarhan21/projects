from typing import List, Dict, Any, Optional
import requests
from bs4 import BeautifulSoup
import time
import re


class GoogleScholarService:
    """
    Service to search Google Scholar and find paper PDFs
    Note: Google Scholar doesn't have an official API, so we use web scraping
    with rate limiting to be respectful
    """
    
    def __init__(self):
        self.base_url = "https://scholar.google.com"
        self.search_url = f"{self.base_url}/scholar"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
        }
        self.rate_limit_delay = 2  # seconds between requests
    
    def search(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Search Google Scholar for papers
        
        Args:
            query: Search query
            max_results: Maximum number of results (default 10)
            
        Returns:
            List of paper dictionaries with title, authors, year, url, pdf_url
        """
        papers = []
        
        params = {
            'q': query,
            'hl': 'en',
            'as_sdt': '0,5',  # Include patents and citations
        }
        
        try:
            print(f"🔍 Searching Google Scholar for: {query}")
            response = requests.get(
                self.search_url,
                params=params,
                headers=self.headers,
                timeout=30
            )
            
            if response.status_code == 200:
                papers = self._parse_search_results(response.text, max_results)
                print(f"✓ Found {len(papers)} papers on Google Scholar")
            else:
                print(f"⚠️ Google Scholar returned status {response.status_code}")
                
            # Rate limiting
            time.sleep(self.rate_limit_delay)
            
        except Exception as e:
            print(f"Error searching Google Scholar: {e}")
        
        return papers
    
    def _parse_search_results(self, html: str, max_results: int) -> List[Dict[str, Any]]:
        """Parse Google Scholar search results HTML"""
        papers = []
        
        try:
            soup = BeautifulSoup(html, 'html.parser')
            results = soup.find_all('div', class_='gs_r gs_or gs_scl')
            
            for idx, result in enumerate(results[:max_results]):
                try:
                    paper = self._extract_paper_info(result)
                    if paper:
                        papers.append(paper)
                except Exception as e:
                    print(f"Error parsing result {idx}: {e}")
                    continue
                    
        except Exception as e:
            print(f"Error parsing Google Scholar results: {e}")
        
        return papers
    
    def _extract_paper_info(self, result_div) -> Optional[Dict[str, Any]]:
        """Extract paper information from a search result div"""
        try:
            # Title and main link
            title_tag = result_div.find('h3', class_='gs_rt')
            if not title_tag:
                return None
            
            title_link = title_tag.find('a')
            title = title_link.get_text() if title_link else title_tag.get_text()
            url = title_link.get('href') if title_link else None
            
            # Clean title (remove [PDF], [HTML], etc.)
            title = re.sub(r'\[PDF\]|\[HTML\]|\[BOOK\]', '', title).strip()
            
            # Authors and publication info
            authors_tag = result_div.find('div', class_='gs_a')
            authors_text = authors_tag.get_text() if authors_tag else ""
            
            # Parse authors and year from the authors_text
            # Format is usually: "Author1, Author2 - Source, Year - Publisher"
            authors = []
            year = None
            
            if authors_text:
                parts = authors_text.split(' - ')
                if parts:
                    # First part is authors
                    author_part = parts[0].strip()
                    authors = [a.strip() for a in author_part.split(',')[:5]]  # Limit to 5
                    
                    # Try to extract year (4 consecutive digits)
                    year_match = re.search(r'\b(19|20)\d{2}\b', authors_text)
                    if year_match:
                        year = int(year_match.group(0))
            
            # Abstract/snippet
            abstract_tag = result_div.find('div', class_='gs_rs')
            abstract = abstract_tag.get_text() if abstract_tag else ""
            
            # PDF link - look for [PDF] link or sidebar PDF link
            pdf_url = None
            
            # Method 1: Look for [PDF] badge with link
            pdf_badge = result_div.find('span', class_='gs_ctg2', string=re.compile(r'\[PDF\]'))
            if pdf_badge:
                pdf_link = pdf_badge.find_parent('a')
                if pdf_link:
                    pdf_url = pdf_link.get('href')
            
            # Method 2: Look for sidebar PDF link
            if not pdf_url:
                gs_ggs = result_div.find('div', class_='gs_ggs gs_fl')
                if gs_ggs:
                    pdf_link = gs_ggs.find('a')
                    if pdf_link:
                        pdf_url = pdf_link.get('href')
            
            # Method 3: Check if main link is a PDF
            if not pdf_url and url and url.lower().endswith('.pdf'):
                pdf_url = url
            
            return {
                'title': title,
                'authors': authors,
                'year': year,
                'abstract': abstract,
                'url': url,
                'pdf_url': pdf_url,
                'source': 'google_scholar'
            }
            
        except Exception as e:
            print(f"Error extracting paper info: {e}")
            return None
    
    def find_pdf_for_paper(self, title: str, authors: List[str] = None) -> Optional[str]:
        """
        Given a paper title (and optionally authors), try to find a PDF link
        
        Args:
            title: Paper title
            authors: Optional list of author names
            
        Returns:
            PDF URL if found, None otherwise
        """
        # Construct search query
        query = f'"{title}"'
        if authors and len(authors) > 0:
            query += f' {authors[0]}'  # Add first author
        
        papers = self.search(query, max_results=3)
        
        # Look for exact title match with PDF
        title_lower = title.lower()
        for paper in papers:
            paper_title_lower = paper.get('title', '').lower()
            
            # Check if titles match (allowing for minor differences)
            if self._titles_match(title_lower, paper_title_lower):
                if paper.get('pdf_url'):
                    return paper['pdf_url']
        
        return None
    
    def _titles_match(self, title1: str, title2: str) -> bool:
        """Check if two titles match (allowing for minor differences)"""
        # Remove punctuation and extra spaces
        clean1 = re.sub(r'[^\w\s]', '', title1.lower()).strip()
        clean2 = re.sub(r'[^\w\s]', '', title2.lower()).strip()
        
        # Check for substantial overlap
        words1 = set(clean1.split())
        words2 = set(clean2.split())
        
        if not words1 or not words2:
            return False
        
        # At least 70% of words should match
        overlap = len(words1.intersection(words2))
        min_words = min(len(words1), len(words2))
        
        return overlap / min_words >= 0.7
    
    def discover_and_fetch(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Combined search operation - find papers and their PDFs
        
        Args:
            query: Search query
            max_results: Maximum number of results
            
        Returns:
            List of papers with metadata and PDF links where available
        """
        return self.search(query, max_results)


# Singleton instance
_google_scholar_service = None

def get_google_scholar_service() -> GoogleScholarService:
    """Get or create GoogleScholarService singleton"""
    global _google_scholar_service
    if _google_scholar_service is None:
        _google_scholar_service = GoogleScholarService()
    return _google_scholar_service
