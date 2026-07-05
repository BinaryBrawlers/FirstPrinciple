#!/usr/bin/env python3
"""
Academic Dependency Graph Generator
Generates concept dependency graphs for academic topics using OpenAlex and LLM.
"""
import os
import json
import asyncio
import aiohttp
import requests
from dataclasses import dataclass
from typing import List, Optional
from dotenv import load_dotenv
from mistralai import Mistral

# =============================================================================
# CONFIGURATION
# =============================================================================

TOP_PAPERS = 70
REFERENCE_DEPTH = 5
TOP_REFERENCES_PER_PAPER = 7
MAX_PAPERS = 300
MAX_PAPERS_PER_DEPTH = 60
FINAL_CONCEPTS = 12

OPENALEX_BASE_URL = "https://api.openalex.org"

# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class Paper:
    id: str
    title: str
    year: Optional[int]
    citation_count: int
    abstract: Optional[str]
    references: List[str]
    
    def __hash__(self):
        return hash(self.id)
    
    def __eq__(self, other):
        if isinstance(other, Paper):
            return self.id == other.id
        return False

@dataclass
class DependencyGraph:
    nodes: List[str]
    edges: List[List[str]]

# =============================================================================
# OPENALEX CLIENT
# =============================================================================

def search_papers(topic: str, top: int = TOP_PAPERS) -> List[Paper]:
    """Search OpenAlex for top papers on a topic."""
    url = f"{OPENALEX_BASE_URL}/works"
    params = {
        "search": topic,
        "sort": "cited_by_count:desc",
        "per_page": top,
        "filter": "type:article"
    }
    
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()
    
    papers = []
    for item in data.get("results", []):
        paper = Paper(
            id=item.get("id", ""),
            title=item.get("title", ""),
            year=item.get("publication_year"),
            citation_count=item.get("cited_by_count", 0),
            abstract=item.get("abstract"),
            references=[ref.replace("https://openalex.org/", "") 
                       for ref in item.get("referenced_works", [])]
        )
        papers.append(paper)
    
    return papers

async def fetch_paper_by_id_async(session: aiohttp.ClientSession, paper_id: str) -> Paper:
    """Async fetch a single paper by OpenAlex ID."""
    url = f"{OPENALEX_BASE_URL}/works/{paper_id}"
    
    async with session.get(url) as response:
        response.raise_for_status()
        item = await response.json()
        
        return Paper(
            id=item.get("id", ""),
            title=item.get("title", ""),
            year=item.get("publication_year"),
            citation_count=item.get("cited_by_count", 0),
            abstract=item.get("abstract"),
            references=[ref.replace("https://openalex.org/", "") 
                       for ref in item.get("referenced_works", [])]
        )

async def fetch_batch_async(paper_ids: List[str]) -> List[Paper]:
    """Fetch multiple papers concurrently."""
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_paper_by_id_async(session, paper_id) for paper_id in paper_ids]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions
        papers = [r for r in results if isinstance(r, Paper)]
        return papers

def expand_references(papers: List[Paper], depth: int = REFERENCE_DEPTH, 
                     top_refs_per_paper: int = TOP_REFERENCES_PER_PAPER, 
                     max_papers: int = MAX_PAPERS,
                     max_papers_per_depth: int = MAX_PAPERS_PER_DEPTH) -> List[Paper]:
    """Expand paper references to specified depth using async fetching."""
    all_papers = {p.id: p for p in papers}
    to_expand = list(papers)
    
    for d in range(depth):
        if len(all_papers) >= max_papers:
            print(f"  ✓ Reached max papers limit ({max_papers})")
            break
            
        print(f"  Depth {d+1}: Processing {len(to_expand)} papers...")
        
        # Collect all reference IDs to fetch
        refs_to_fetch = []
        for paper in to_expand:
            for ref_id in paper.references[:top_refs_per_paper]:
                if ref_id not in all_papers and len(refs_to_fetch) < max_papers_per_depth:
                    if ref_id not in refs_to_fetch:
                        refs_to_fetch.append(ref_id)
                        
        if not refs_to_fetch:
            print(f"  ✓ No more references to expand")
            break
        
        print(f"    → Fetching {len(refs_to_fetch)} papers in parallel...")
        
        # Fetch all in parallel
        fetched_papers = asyncio.run(fetch_batch_async(refs_to_fetch))
        
        # Add to collection
        next_batch = []
        for paper in fetched_papers:
            if paper.id not in all_papers and len(all_papers) < max_papers:
                all_papers[paper.id] = paper
                next_batch.append(paper)
        
        skipped_count = len(refs_to_fetch) - len(fetched_papers)
        print(f"  ✓ Depth {d+1} complete: {len(fetched_papers)} new papers, {skipped_count} skipped, {len(all_papers)} total")
        
        to_expand = next_batch
    
    return list(all_papers.values())

# =============================================================================
# GRAPH BUILDER
# =============================================================================

def rank_papers(papers: List[Paper]) -> List[Paper]:
    """Rank papers by citation count."""
    return sorted(papers, key=lambda p: p.citation_count, reverse=True)

def build_context(topic: str, papers: List[Paper]) -> str:
    """Build comprehensive context from papers."""
    top_papers = papers[:TOP_PAPERS]
    
    context_parts = [
        f"Below are the most influential papers ordered by citation count.\n"
    ]
    
    # Add paper details
    for i, paper in enumerate(top_papers, 1):
        context_parts.append(f"\nPaper {i}:")
        context_parts.append(f"Title: {paper.title}")
        context_parts.append(f"Year: {paper.year}")
        if paper.abstract:
            # Truncate long abstracts
            abstract = paper.abstract[:500] + "..." if len(paper.abstract) > 500 else paper.abstract
            context_parts.append(f"Abstract: {abstract}")
        context_parts.append(f"Citations: {paper.citation_count}")
        context_parts.append(f"References: {len(paper.references)}")
    
    return "\n".join(context_parts)

# =============================================================================
# LLM CLIENT
# =============================================================================

def get_mistral_client():
    """Initialize Mistral client."""
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("MISTRAL_API_KEY not found in environment")
    return Mistral(api_key=api_key)

def build_dependency_graph(context: str, topic: str, final_concepts: int = FINAL_CONCEPTS) -> DependencyGraph:
    """Single LLM call to build the dependency graph from all evidence."""
    
    client = get_mistral_client()
    
    prompt = f"""Topic: {topic}

{context}

Task:

Using ONLY the supplied evidence:

1. Identify the {final_concepts} most important concepts required to understand the topic.
2. Ignore implementation details and minor concepts.
3. Focus only on historically significant concepts.
4. Arrange them as prerequisite relationships.
5. Return ONLY JSON.

Format:

{{
  "nodes": [
    "Biological Neuron",
    "Artificial Neuron",
    ...
  ],
  "edges": [
    ["Biological Neuron", "Artificial Neuron"],
    ["Artificial Neuron", "Perceptron"],
    ...
  ]
}}"""
    
    response = client.chat.complete(
        model="mistral-large-latest",
        messages=[
            {"role": "system", "content": "You are a scientific concept mapper. Return only valid JSON."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )
    
    result = response.choices[0].message.content.strip()
    
    # Parse JSON
    if result.startswith("```json"):
        result = result[7:]
    if result.startswith("```"):
        result = result[3:]
    if result.endswith("```"):
        result = result[:-3]
    result = result.strip()
    
    data = json.loads(result)
    
    return DependencyGraph(
        nodes=data["nodes"],
        edges=data["edges"]
    )

# =============================================================================
# MAIN PIPELINE
# =============================================================================

def generate_dependency_graph(topic: str) -> dict:
    """
    Generate a dependency graph for a given topic.

    Args:
        topic: The academic topic to generate a dependency graph for.

    Returns:
        A dict with keys "nodes" (list of concept strings) and
        "edges" (list of [prerequisite, concept] pairs).

    Raises:
        ValueError: If MISTRAL_API_KEY is not set in the environment.
    """
    load_dotenv()

    print(f"🔍 Searching OpenAlex for: {topic}")
    initial_papers = search_papers(topic, top=TOP_PAPERS)
    print(f"✓ Found {len(initial_papers)} initial papers")
    
    print(f"\n📚 Expanding references (depth={REFERENCE_DEPTH})...")
    all_papers = expand_references(
        initial_papers,
        depth=REFERENCE_DEPTH,
        top_refs_per_paper=TOP_REFERENCES_PER_PAPER,
        max_papers=MAX_PAPERS,
        max_papers_per_depth=MAX_PAPERS_PER_DEPTH
    )
    print(f"✓ Expanded to {len(all_papers)} total papers")
    
    print(f"\n📊 Ranking papers by citations...")
    ranked_papers = rank_papers(all_papers)
    print(f"✓ Top paper: {ranked_papers[0].title} ({ranked_papers[0].citation_count} citations)")
    
    print(f"\n🧠 Building context for LLM...")
    context = build_context(topic, ranked_papers)
    print(f"✓ Context: {len(context)} characters")
    
    print(f"\n🤖 Calling LLM to generate dependency graph...")
    graph = build_dependency_graph(context, topic, FINAL_CONCEPTS)
    print(f"✓ Graph: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
    
    return {
        "nodes": graph.nodes,
        "edges": graph.edges
    }
