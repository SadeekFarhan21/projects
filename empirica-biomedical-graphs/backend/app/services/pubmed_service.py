from typing import List, Dict, Any
import requests
import xml.etree.ElementTree as ET


class PubMedService:
    """
    Service to search and fetch papers from PubMed via E-utilities API
    """
    
    def __init__(self):
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        # You should set this to a real email for NCBI compliance
        self.email = "empirica@example.com"
    
    def search(self, query: str, max_results: int = 10) -> List[str]:
        """
        Search PubMed and return PMIDs
        """
        url = f"{self.base_url}/esearch.fcgi"
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retmode": "json",
            "email": self.email,
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            return data.get("esearchresult", {}).get("idlist", [])
        except Exception as e:
            print(f"PubMed search error: {e}")
            return []
    
    def fetch_abstracts(self, pmids: List[str]) -> List[Dict[str, Any]]:
        """
        Fetch article details including abstracts for given PMIDs
        """
        if not pmids:
            return []
        
        url = f"{self.base_url}/efetch.fcgi"
        params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
            "rettype": "abstract",
            "email": self.email,
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            return self._parse_pubmed_xml(response.text)
        except Exception as e:
            print(f"PubMed fetch error: {e}")
            return []
    
    def _parse_pubmed_xml(self, xml_text: str) -> List[Dict[str, Any]]:
        """Parse PubMed XML response"""
        papers = []
        
        try:
            root = ET.fromstring(xml_text)
            
            for article in root.findall(".//PubmedArticle"):
                paper = self._extract_article_data(article)
                if paper:
                    papers.append(paper)
        except Exception as e:
            print(f"XML parsing error: {e}")
        
        return papers
    
    def _extract_article_data(self, article_elem) -> Dict[str, Any]:
        """Extract structured data from a PubmedArticle element"""
        try:
            # PMID
            pmid_elem = article_elem.find(".//PMID")
            pmid = pmid_elem.text if pmid_elem is not None else ""
            
            # Article info
            article_node = article_elem.find(".//Article")
            if article_node is None:
                return None
            
            # Title
            title_elem = article_node.find(".//ArticleTitle")
            title = title_elem.text if title_elem is not None else "No Title"
            
            # Abstract
            abstract_parts = []
            for abs_text in article_node.findall(".//AbstractText"):
                text = abs_text.text or ""
                abstract_parts.append(text)
            abstract = " ".join(abstract_parts) if abstract_parts else "No abstract available"
            
            # Authors
            authors = []
            for author in article_node.findall(".//Author"):
                last = author.find("LastName")
                first = author.find("ForeName")
                if last is not None:
                    name = last.text or ""
                    if first is not None:
                        name = f"{first.text} {name}"
                    authors.append(name)
            
            # Journal
            journal_elem = article_node.find(".//Journal/Title")
            journal = journal_elem.text if journal_elem is not None else ""
            
            # Year
            year_elem = article_node.find(".//PubDate/Year")
            year = int(year_elem.text) if year_elem is not None and year_elem.text else None
            
            # DOI - Try to extract DOI for PDF access
            doi = None
            for article_id in article_elem.findall(".//ArticleId"):
                if article_id.get("IdType") == "doi":
                    doi = article_id.text
                    break
            
            # PMC ID - For free full-text access
            pmc_id = None
            for article_id in article_elem.findall(".//ArticleId"):
                if article_id.get("IdType") == "pmc":
                    pmc_id = article_id.text
                    break
            
            return {
                "pmid": pmid,
                "id": pmid,
                "title": title,
                "abstract": abstract,
                "authors": authors[:5],  # Limit to first 5
                "journal": journal,
                "year": year,
                "doi": doi,
                "pmc_id": pmc_id,
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                "pdf_url": f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmc_id}/pdf/" if pmc_id else None
            }
        except Exception as e:
            print(f"Error extracting article: {e}")
            return None
    
    def discover_and_fetch(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Combined search and fetch operation
        """
        pmids = self.search(query, max_results)
        if not pmids:
            return []
        return self.fetch_abstracts(pmids)

