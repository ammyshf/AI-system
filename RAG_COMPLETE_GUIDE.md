# Building RAG Systems: From Beginner to Production

> A complete guide to understanding, building, and deploying Retrieval Augmented Generation pipelines for real-world applications.

---

## Table of Contents

1. [What is RAG and Why Does It Exist?](#1-what-is-rag-and-why-does-it-exist)
2. [The RAG Mental Model](#2-the-rag-mental-model)
3. [The Complete RAG Pipeline](#3-the-complete-rag-pipeline)
4. [Chunking — Breaking Documents Into Pieces](#4-chunking--breaking-documents-into-pieces)
5. [Embeddings — The Heart of RAG](#5-embeddings--the-heart-of-rag)
6. [Vector Databases — Where Embeddings Live](#6-vector-databases--where-embeddings-live)
7. [Retrieval Strategies — Finding the Right Documents](#7-retrieval-strategies--finding-the-right-documents)
8. [Augmentation — Prompt Engineering for RAG](#8-augmentation--prompt-engineering-for-rag)
9. [Generation — Controlling the LLM Output](#9-generation--controlling-the-llm-output)
10. [Building Your First RAG System (Code)](#10-building-your-first-rag-system-code)
11. [Production Patterns — From Prototype to Real System](#11-production-patterns--from-prototype-to-real-system)
12. [Advanced RAG Architectures](#12-advanced-rag-architectures)
13. [Evaluation — Measuring RAG Quality](#13-evaluation--measuring-rag-quality)
14. [Common Problems and Solutions](#14-common-problems-and-solutions)
15. [RAG vs Fine-Tuning — When to Use Which](#15-rag-vs-fine-tuning--when-to-use-which)
16. [Production Checklist](#16-production-checklist)
17. [Interview Quick Reference](#17-interview-quick-reference)

---

## 1. What is RAG and Why Does It Exist?

### The Problem

Large Language Models (LLMs) like GPT-4, Gemini, and Claude are incredibly smart. They've read most of the public internet during training. But they have three critical blindspots:

```
┌─────────────────────────────────────────────────────┐
│                    LLM's Brain                       │
│                                                      │
│   ✅ Knows: Wikipedia, textbooks, public websites    │
│   ✅ Knows: General medicine, law, coding, etc.      │
│                                                      │
│   ❌ Doesn't know: Your company's private data       │
│   ❌ Doesn't know: Events after training cutoff      │
│   ❌ Sometimes: Confidently makes things up          │
│                                                      │
└─────────────────────────────────────────────────────┘
```

**Real example:**

```
You:  "What medications is Patient #4521 taking?"
LLM:  "I have no idea who Patient #4521 is."

You:  "What's our company's refund policy?"
LLM:  "I don't have access to your company's policies."

You:  "What happened in the markets yesterday?"
LLM:  "My training data only goes up to [cutoff date]."
```

### The Solution: RAG

**RAG = Retrieval Augmented Generation**

Instead of asking the LLM to answer from memory, you:
1. **Search** your own data first
2. **Give** the relevant results to the LLM
3. **Let** the LLM answer based on that data

```
WITHOUT RAG:
┌──────────┐     "What's Patient X's     ┌─────────┐
│   You    │ ──── medication?" ─────────→ │   LLM   │ → "I don't know"
└──────────┘                              └─────────┘

WITH RAG:
┌──────────┐   ① Search your DB          ┌──────────┐
│   You    │ ───────────────────────────→ │ Database │
└──────────┘                              └────┬─────┘
                                               │
                 ② Found: "Metformin 500mg"    │
                                               ▼
┌──────────┐   ③ "Patient X takes          ┌─────────┐
│   You    │ ◄── Metformin 500mg for..." ──│   LLM   │ ← Sees the data
└──────────┘                               └─────────┘
```

**That's it. RAG = Search first, then ask LLM with the search results.**

The term comes from a 2020 research paper by Facebook AI. It stands for:
- **R**etrieval — find relevant information
- **A**ugmented — add it to the prompt
- **G**eneration — LLM generates the answer

---

## 2. The RAG Mental Model

### The Open-Book Exam Analogy

This is the simplest way to understand RAG:

```
╔══════════════════════════════════════════════════════════╗
║                                                          ║
║   CLOSED-BOOK EXAM (= LLM without RAG)                  ║
║                                                          ║
║   Teacher: "What's the GDP of France in 2025?"           ║
║   Student: "Umm... maybe $3 trillion?" (GUESSING!)      ║
║                                                          ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║   OPEN-BOOK EXAM (= LLM with RAG)                       ║
║                                                          ║
║   Teacher: "What's the GDP of France in 2025?"           ║
║   Student: *flips to page 47 of reference book*          ║
║            "According to page 47, it's $3.2 trillion"    ║
║            (ACCURATE! WITH SOURCE!)                      ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
```

The AI doesn't need to memorize your data. It just needs to **look it up** before answering.

### Another Analogy: Google Search + Smart Friend

You already use a version of RAG in daily life:

```
You want to know: "What time does the bank close today?"

Step 1: RETRIEVE — You Google it, find the bank's website
Step 2: AUGMENT  — You read the relevant hours section
Step 3: GENERATE — You tell your friend: "It closes at 5 PM"

RAG does exactly this, but automatically with AI.
```

---

## 3. The Complete RAG Pipeline

Every RAG system has two phases:

```
═══════════════════════════════════════════════════════════════
  OFFLINE PHASE (done once, or when documents change)
═══════════════════════════════════════════════════════════════

  📄 Documents → [CHUNK] → [EMBED] → [STORE in Vector DB]

  This prepares your "library" for searching.

═══════════════════════════════════════════════════════════════
  ONLINE PHASE (every time a user asks a question)
═══════════════════════════════════════════════════════════════

  User Question
       │
       ├──→ [EMBED the question] → convert to same number format
       │              │
       │              ▼
       │    [SEARCH Vector DB] → find closest document chunks
       │              │
       │              ▼
       │    Retrieved chunks (top 3-5 most relevant)
       │              │
       ├──────────────┘
       │
       ▼
  [AUGMENT] = Question + Retrieved chunks combined into prompt
       │
       ▼
  [GENERATE] = LLM reads everything, writes answer
       │
       ▼
  Final Answer to User
```

### Walkthrough: Bank Customer Support Bot

A customer asks: **"What's the penalty for early loan closure?"**

```
                    "What's the penalty for
                     early loan closure?"
                            │
                            │
                ┌───────────▼────────────┐
                │                        │
                │   ① R — RETRIEVE       │
                │                        │
                │   Search through all   │
                │   bank policy docs     │
                │   to find relevant     │
                │   paragraphs           │
                │                        │
                │   Found:               │
                │   📄 "Loan closure     │
                │   before 12 months     │
                │   incurs 3% penalty    │
                │   on outstanding       │
                │   principal..."        │
                │                        │
                └───────────┬────────────┘
                            │
                ┌───────────▼────────────┐
                │                        │
                │   ② A — AUGMENT        │
                │                        │
                │   Combine the found    │
                │   document WITH the    │
                │   original question    │
                │   into ONE prompt      │
                │                        │
                │   "Here is the bank's  │
                │    policy: [...]       │
                │    Now answer:          │
                │    What's the penalty  │
                │    for early closure?" │
                │                        │
                └───────────┬────────────┘
                            │
                ┌───────────▼────────────┐
                │                        │
                │   ③ G — GENERATE       │
                │                        │
                │   LLM reads the        │
                │   context + question   │
                │   and writes a         │
                │   natural answer       │
                │                        │
                │   "Early loan closure  │
                │   within 12 months     │
                │   has a 3% penalty     │
                │   on the outstanding   │
                │   principal amount."   │
                │                        │
                └────────────────────────┘
```

---

## 4. Chunking — Breaking Documents Into Pieces

### Why Chunk?

You can't embed a 200-page PDF as one vector. The meaning would be too diluted — like asking "what's this book about?" and getting "everything" as the answer. Smaller, focused chunks match specific questions better.

```
    📄 200-page Loan Policy PDF

    Pages 1-5:   KYC requirements
    Pages 6-15:  Anti-money laundering rules
    Pages 16-30: Loan interest rate guidelines
    Pages 31-40: Digital payment regulations
    Pages 41-50: Data privacy compliance

    ❌ One vector for ALL 200 pages:
       [0.5, 0.5, 0.5, ...] ← Average of everything. Useless!

    ✅ One vector PER focused paragraph:
       "KYC docs needed" → [0.9, 0.1, 0.8, ...] ← Specific. Useful!
```

### The 4 Chunking Strategies

#### Strategy 1: Fixed-Size Chunking (Simplest)

Split every N characters, regardless of content:

```
Original text:
"KYC requires identity proof such as Aadhaar or PAN card.
 Address proof like utility bills is also mandatory. The
 bank must verify documents within 14 days of account
 opening. For high-risk customers, enhanced due diligence
 is required including source of funds verification..."

Split at every 150 characters:

┌─────────────────────────────────────────────┐
│ Chunk 1: "KYC requires identity proof such  │
│ as Aadhaar or PAN card. Address proof like  │
│ utility bills is also mandatory. The ba..." │ ← SENTENCE CUT!
├─────────────────────────────────────────────┤
│ Chunk 2: "...nk must verify documents       │
│ within 14 days of account opening. For      │
│ high-risk customers, enhanced due dili..."  │ ← CUT AGAIN!
└─────────────────────────────────────────────┘

⚠️ Problem: Sentences get chopped in half
```

**Fix: Add overlap** — each chunk starts a few sentences before the previous one ended:

```
    Chunk 1:  [========================]
    Chunk 2:       [========================]     ← overlaps!
    Chunk 3:            [========================]

    The overlapping region ensures no sentence is completely
    lost at boundaries. Typical overlap: 10-20% of chunk size.
```

```python
# Fixed-size chunking with overlap
from langchain.text_splitter import CharacterTextSplitter

splitter = CharacterTextSplitter(
    chunk_size=500,       # 500 characters per chunk
    chunk_overlap=50,     # 50 characters overlap
    separator=" "         # try to split at spaces
)
chunks = splitter.split_text(document_text)
```

#### Strategy 2: Recursive Character Splitting (Most Common in Production)

Try to split at natural boundaries first, then fall back to smaller boundaries:

```
Priority order:
  1st try: Split at paragraph breaks ("\n\n")
  2nd try: Split at sentence endings (". ")
  3rd try: Split at word boundaries (" ")
  4th try: Split at character level (last resort)

Input:
"## KYC Requirements

Identity proof such as Aadhaar or PAN card is mandatory.
Address proof like utility bills must be submitted.

## Verification Timeline

The bank must verify all documents within 14 days.
For high-risk customers, enhanced due diligence applies."


Output:
┌──────────────────────────────────────────┐
│ Chunk 1: "KYC Requirements:             │
│ Identity proof such as Aadhaar or PAN    │  Split at paragraph
│ card is mandatory. Address proof like    │  break ("\n\n") ✅
│ utility bills must be submitted."        │
├──────────────────────────────────────────┤
│ Chunk 2: "Verification Timeline:         │
│ The bank must verify all documents       │  Clean semantic
│ within 14 days. For high-risk customers, │  boundary ✅
│ enhanced due diligence applies."         │
└──────────────────────────────────────────┘

No broken sentences! Each chunk is a complete thought.
```

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""]  # priority order
)
chunks = splitter.split_text(document_text)
```

#### Strategy 3: Semantic Chunking (Smartest)

Use embeddings themselves to detect where the topic changes:

```
Sentence 1: "KYC requires identity proof"         → [0.9, 0.1, 0.2]
Sentence 2: "Address proof is also needed"         → [0.85, 0.15, 0.25]
Sentence 3: "Documents verified within 14 days"    → [0.8, 0.2, 0.3]
Sentence 4: "Loan interest rates set by RBI"       → [0.1, 0.9, 0.7]
Sentence 5: "Fixed rate loans have constant EMI"   → [0.15, 0.85, 0.65]

Compare consecutive sentence similarity:
  Sent 1↔2: 0.95 (same topic → keep together)
  Sent 2↔3: 0.88 (same topic → keep together)
  Sent 3↔4: 0.25 (TOPIC CHANGE → SPLIT HERE!)
  Sent 4↔5: 0.92 (same topic → keep together)

Result:
  Chunk 1: Sentences 1+2+3 (all about KYC verification)
  Chunk 2: Sentences 4+5 (all about loan interest rates)
```

```python
from langchain_experimental.text_splitter import SemanticChunker
from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-exp-03-07")
chunker = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="percentile",  # split at big similarity drops
    breakpoint_threshold_amount=75
)
chunks = chunker.split_text(document_text)
```

#### Strategy 4: Document-Structure Chunking

Use the document's own structure (headings, HTML tags, markdown):

```
## KYC Requirements          ← Chunk boundary
Content about KYC...

## Anti-Money Laundering     ← Chunk boundary
Content about AML...

## Loan Guidelines           ← Chunk boundary
Content about loans...
```

```python
from langchain.text_splitter import MarkdownHeaderTextSplitter

headers = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]
splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers)
chunks = splitter.split_text(markdown_text)
# Each chunk retains its header as metadata!
```

### Chunking Decision Guide

```
┌──────────────────┬──────────┬───────────┬──────────────────────┐
│ Strategy         │ Speed    │ Quality   │ Use When             │
├──────────────────┼──────────┼───────────┼──────────────────────┤
│ Fixed-size       │ Fastest  │ ⭐⭐      │ Quick prototype, MVP │
│ Recursive        │ Fast     │ ⭐⭐⭐    │ Most production apps │
│ Semantic         │ Slow     │ ⭐⭐⭐⭐  │ High-quality needs   │
│ Document-struct  │ Fast     │ ⭐⭐⭐⭐  │ Structured docs      │
└──────────────────┴──────────┴───────────┴──────────────────────┘

Recommended chunk sizes:
  - Small (200-500 tokens): Precise retrieval, FAQ-style
  - Medium (500-1000 tokens): General purpose (START HERE)
  - Large (1000-2000 tokens): Complex topics needing context

Overlap: 10-20% of chunk size
```

---

## 5. Embeddings — The Heart of RAG

### What is an Embedding?

An embedding converts text into a list of numbers (a "vector") that captures the **meaning** of that text.

Think of it as GPS coordinates, but for meaning instead of location:

```
    REAL WORLD (GPS):                  MEANING WORLD (Embeddings):

    Mumbai  = [19.0, 72.8]            "loan penalty"      = [0.8, 0.9, 0.2]
    Pune    = [18.5, 73.8]            "prepayment charges" = [0.75, 0.85, 0.25]
    Tokyo   = [35.6, 139.6]           "savings account"    = [0.1, 0.2, 0.9]

    Mumbai & Pune = CLOSE              loan penalty & prepayment = CLOSE ✅
    Mumbai & Tokyo = FAR               loan penalty & savings = FAR ❌
```

**Key insight:** "Loan penalty" and "prepayment charges" use completely different words but mean the same thing. Embeddings capture this — they're close together in the meaning space.

### How Does the Model Learn This?

The embedding model is trained on millions of text pairs:

```
TRAINING DATA:
┌──────────────────────────────┬───────────────────────────────┐
│ Text A                       │ Text B (similar meaning)      │
├──────────────────────────────┼───────────────────────────────┤
│ "wire transfer fee"          │ "cost of sending money abroad"│
│ "open a bank account"        │ "start a new account"         │
│ "loan EMI calculation"       │ "monthly installment formula" │
│ "credit score"               │ "CIBIL rating"                │
└──────────────────────────────┴───────────────────────────────┘

The model learns:
  "wire transfer fee" and "cost of sending money abroad"
  → should map to SIMILAR vectors (close on the map)

  "wire transfer fee" and "loan EMI calculation"
  → should map to DIFFERENT vectors (far on the map)
```

### What Do the Numbers Represent?

Each dimension captures an aspect of meaning (learned automatically, not hand-coded):

```
Simplified (real vectors have 768-3072 dimensions):

                          [financial, personal, digital, regulatory]

"KYC verification"      → [0.85,      0.70,     0.30,    0.90]
                            ↑ high      ↑ yes      ↑ low    ↑ very high
                            (finance)  (identity) (paper)  (compliance)

"mobile banking app"    → [0.80,      0.40,     0.95,    0.50]
                            ↑ high     ↑ low      ↑ very   ↑ moderate
                            (finance)             high(digital!)

"pasta recipe"          → [0.05,      0.30,     0.20,    0.02]
                            ↑ not                          ↑ not
                            financial                      regulatory
```

### Cosine Similarity — Measuring Closeness

How do we measure if two vectors are "close"? The standard method is **cosine similarity** — it measures the angle between two arrows:

```
                Small angle = Similar meaning

                            ▲ "loan penalty"
                           /
                          /  } 10° angle → similarity ≈ 0.98
                         /
                        ▲ "prepayment charges"


                Large angle = Different meaning

                            ▲ "loan penalty"
                           /
                          /
                         /      } 80° angle → similarity ≈ 0.17
                        /
                       /________▶ "cooking recipe"
```

```
Score 1.0  = Identical meaning (0° angle)
Score 0.95 = Very similar (common threshold for "match")
Score 0.5  = Somewhat related
Score 0.0  = Completely unrelated (90° angle)
```

The formula (for reference, libraries handle this):

```python
import numpy as np

def cosine_similarity(vec_a, vec_b):
    dot_product = np.dot(vec_a, vec_b)
    magnitude_a = np.linalg.norm(vec_a)
    magnitude_b = np.linalg.norm(vec_b)
    return dot_product / (magnitude_a * magnitude_b)

# Example
query    = [0.8, 0.9, 0.2]
doc_good = [0.75, 0.85, 0.25]
doc_bad  = [0.1, 0.2, 0.9]

print(cosine_similarity(query, doc_good))  # 0.997 → Very similar!
print(cosine_similarity(query, doc_bad))   # 0.456 → Not similar
```

### Popular Embedding Models

```
┌────────────────────────────┬───────────┬────────┬────────────────────┐
│ Model                      │ Dimensions│ Cost   │ Notes              │
├────────────────────────────┼───────────┼────────┼────────────────────┤
│ OpenAI text-embedding-     │ 1536      │ Paid   │ Most popular,      │
│ 3-small                    │           │        │ good balance       │
├────────────────────────────┼───────────┼────────┼────────────────────┤
│ OpenAI text-embedding-     │ 3072      │ Paid   │ Best quality from  │
│ 3-large                    │           │        │ OpenAI             │
├────────────────────────────┼───────────┼────────┼────────────────────┤
│ Google Gemini Embedding    │ 768-3072  │ Free/  │ Configurable dims, │
│                            │ (config.) │ Paid   │ very competitive   │
├────────────────────────────┼───────────┼────────┼────────────────────┤
│ Cohere embed-v3            │ 1024      │ Paid   │ Built for search,  │
│                            │           │        │ multilingual       │
├────────────────────────────┼───────────┼────────┼────────────────────┤
│ sentence-transformers      │ 384-768   │ Free   │ Runs locally,      │
│ (all-MiniLM-L6-v2)        │           │        │ great for MVP      │
├────────────────────────────┼───────────┼────────┼────────────────────┤
│ BGE / E5 (open-source)     │ 768-1024  │ Free   │ Top open-source    │
│                            │           │        │ performance        │
└────────────────────────────┴───────────┴────────┴────────────────────┘

More dimensions = more nuance = better quality but more storage/slower
```

### Critical Rules for Embeddings

```
RULE 1: Same model for documents AND queries
  ✅ Docs with Gemini + Queries with Gemini
  ❌ Docs with OpenAI + Queries with Gemini
  Why? Each model has its own "coordinate system"

RULE 2: Re-embed when you change models
  If you switch from OpenAI to Gemini, you must re-embed
  ALL documents. Old vectors are incompatible.

RULE 3: Batch embedding is cheaper and faster
  ❌ Embed 1000 documents one by one (1000 API calls)
  ✅ Embed 1000 documents in batches of 100 (10 API calls)
```

---

## 6. Vector Databases — Where Embeddings Live

### Why Not a Regular Database?

```
REGULAR SQL DATABASE:
    SELECT * FROM docs WHERE content = 'loan penalty'
    → Only finds EXACT text match ❌
    → Won't find "prepayment charges" even though same meaning

VECTOR DATABASE:
    SELECT * FROM docs ORDER BY embedding <=> query_embedding LIMIT 5
    → Finds documents with SIMILAR MEANING ✅
    → "prepayment charges" matches "loan penalty" because vectors are close
```

### How Vector Search Works

**Naive approach (brute force):** Compare query with every single vector.

```
Query vector: [0.8, 0.9, 0.2]

Compare with Doc 1:    [0.75, 0.85, 0.25] → similarity = 0.98
Compare with Doc 2:    [0.10, 0.20, 0.90] → similarity = 0.15
Compare with Doc 3:    [0.50, 0.60, 0.30] → similarity = 0.72
...
Compare with Doc 999,999: ...
Compare with Doc 1,000,000: ...  😰 Takes forever!
```

**Smart approach (indexing):** Organize vectors so you only search a subset.

#### HNSW Index (Hierarchical Navigable Small World)

The most popular vector index. Think of it like a map with zoom levels:

```
Level 3 (top — few nodes, big jumps):

    [Region A: Finance] -------- [Region B: Medical] -------- [Region C: Tech]

Level 2 (more detail):

    [Loans]---[Accounts]---[Cards]    [Drugs]---[Diagnosis]    [AI]---[Web]

Level 1 (most detail):

    [Home loan] [Personal] [Auto]  [Savings] [Current] [FD] ...

Search path for "loan penalty":

    Level 3: Start → Region A (finance) ← closest at this zoom!
    Level 2: Region A → Loans ← closest!
    Level 1: Loans → [Home loan penalty doc] ← FOUND!

    Checked ~10 vectors instead of 1,000,000 ⚡

Trade-off: Might miss the absolute best match (approximate search)
           but finds a very good match extremely fast.
```

#### IVF Index (Inverted File Index)

```
Step 1: Cluster all vectors into groups (like folders)

    Cluster A: [all finance vectors]     — 50,000 vectors
    Cluster B: [all medical vectors]     — 50,000 vectors
    Cluster C: [all tech vectors]        — 50,000 vectors

Step 2: For a query, find the closest cluster(s) first
Step 3: Only search within those clusters

    Query about "loan penalty" → Cluster A → search 50,000 (not 150,000)
```

### Vector Database Comparison

```
┌───────────────┬───────────┬───────────┬──────────────────────────────┐
│ Database      │ Type      │ Cost      │ Best For                     │
├───────────────┼───────────┼───────────┼──────────────────────────────┤
│ pgvector      │ PostgreSQL│ Free      │ You already use PostgreSQL.  │
│               │ extension │           │ No new infra needed.         │
│               │           │           │ Great up to ~5M vectors.     │
├───────────────┼───────────┼───────────┼──────────────────────────────┤
│ Pinecone      │ Managed   │ Paid      │ Zero ops, auto-scales,       │
│               │ cloud     │ (free     │ production-ready from day 1. │
│               │           │  tier)    │ Best DX (developer exp).     │
├───────────────┼───────────┼───────────┼──────────────────────────────┤
│ ChromaDB      │ Embedded/ │ Free      │ Prototyping, small datasets. │
│               │ local     │           │ Simple Python API.           │
│               │           │           │ Runs in-process.             │
├───────────────┼───────────┼───────────┼──────────────────────────────┤
│ Weaviate      │ Self-host │ Free/Paid │ Built-in ML models,          │
│               │ or cloud  │           │ hybrid search, GraphQL API.  │
├───────────────┼───────────┼───────────┼──────────────────────────────┤
│ Qdrant        │ Self-host │ Free/Paid │ Rust-based, very fast,       │
│               │ or cloud  │           │ rich filtering support.      │
├───────────────┼───────────┼───────────┼──────────────────────────────┤
│ FAISS         │ Library   │ Free      │ Research, pure speed.        │
│ (Facebook)    │ (in-RAM)  │           │ No persistence by default.   │
│               │           │           │ You manage storage.          │
└───────────────┴───────────┴───────────┴──────────────────────────────┘

RECOMMENDATION:
  - Prototype:  ChromaDB (zero setup)
  - Production: pgvector (if you have PostgreSQL) or Pinecone (managed)
  - Research:   FAISS (raw speed)
```

### pgvector in Practice

```sql
-- Enable the extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create table with vector column
CREATE TABLE document_chunks (
    id SERIAL PRIMARY KEY,
    content TEXT,
    source_file TEXT,
    embedding VECTOR(768)          -- 768-dimensional vector
);

-- Create HNSW index for fast cosine search
CREATE INDEX ON document_chunks
USING hnsw (embedding vector_cosine_ops);

-- Insert a chunk with its embedding
INSERT INTO document_chunks (content, source_file, embedding)
VALUES (
    'Loan prepayment penalty is 3% within 12 months',
    'loan_policy_v3.pdf',
    '[0.8, 0.9, 0.2, ...]'        -- 768 numbers from embedding model
);

-- Search: Find most similar chunks to a question
SELECT
    content,
    source_file,
    1 - (embedding <=> $1) AS similarity    -- <=> is cosine distance
FROM document_chunks
WHERE 1 - (embedding <=> $1) > 0.80        -- minimum similarity threshold
ORDER BY embedding <=> $1                   -- closest first
LIMIT 5;                                    -- return top 5 matches
```

Key pgvector operators:
```
<=>  Cosine distance     (most common for text similarity)
<->  L2/Euclidean distance
<#>  Inner product (negative)
```

---

## 7. Retrieval Strategies — Finding the Right Documents

Retrieval quality determines RAG quality. There are four main approaches:

### Strategy A: Dense Retrieval (Embedding-Based)

What we've been discussing — everything is vectors.

```
Question: "What are the charges for NEFT transfer?"
                    │
                    ▼ Embed
            [0.7, 0.3, 0.8, ...]
                    │
                    ▼ Vector similarity search
    ┌──────────────────────────────────┐
    │ "NEFT transaction fee: ₹2.50    │ similarity: 0.93 ✅
    │  for up to ₹10,000..."          │
    ├──────────────────────────────────┤
    │ "RTGS charges: ₹20 for amounts  │ similarity: 0.85
    │  above ₹2 lakhs..."             │ (related, not exact)
    └──────────────────────────────────┘

✅ Understands meaning: "charges" = "fee" = "cost"
❌ Can miss exact terms, codes, account numbers
```

### Strategy B: Sparse Retrieval (Keyword-Based — BM25)

Traditional text search, matching specific words:

```
Question: "What is RBI circular RBI/2024-25/27?"
                    │
                    ▼ Keyword extraction
            ["RBI", "circular", "RBI/2024-25/27"]
                    │
                    ▼ BM25 scoring
    ┌──────────────────────────────────┐
    │ "As per RBI circular             │ score: 12.5 ✅
    │  RBI/2024-25/27, banks must..."  │ (exact match on code!)
    └──────────────────────────────────┘

✅ Perfect for exact codes, IDs, names, specific numbers
❌ "charges" won't match "fee" — no meaning understanding
```

**BM25 explained simply:** It scores documents by:
- How often the search terms appear in the document (more = better)
- How rare those terms are across ALL documents (rarer = more important)
- How long the document is (shorter docs with the term = more relevant)

### Strategy C: Hybrid Retrieval (Best of Both Worlds)

Combine dense + sparse, then merge the results:

```
Question: "What are NEFT charges as per RBI/2024-25/27?"
          ├── meaning part ──┤    ├── exact code ───────┤

    Dense search (embeddings):          Sparse search (BM25):
      "NEFT fee is ₹2.50..."  (8)       "RBI/2024-25/27 states..." (9)
      "RTGS charges: ₹20..."  (6)       "NEFT fee is ₹2.50..."    (7)
      "Fund transfer limits.." (5)       "RBI circular summary..."  (4)

    ┌─────────────────────────────────────────────────┐
    │  Reciprocal Rank Fusion (RRF) — merge formula:  │
    │                                                  │
    │  For each document, combine its rank from both:  │
    │  RRF_score = 1/(k + rank_dense) + 1/(k + rank_sparse)  │
    │                                                  │
    │  "NEFT fee is ₹2.50..." → rank 1 in dense,      │
    │                           rank 2 in sparse       │
    │                        → HIGH combined score ✅  │
    │                                                  │
    │  Final ranking:                                  │
    │    #1: "NEFT fee is ₹2.50..."      (both lists!) │
    │    #2: "RBI/2024-25/27 states..."  (exact code)  │
    │    #3: "RTGS charges: ₹20..."      (related)     │
    └─────────────────────────────────────────────────┘
```

### Strategy D: Re-Ranking (Second Pass Filtering)

Initial retrieval is fast but rough. A re-ranker polishes the results:

```
Step 1: Retrieve top 20 candidates (fast)

    1. "NEFT fee schedule..."              score: 0.93
    2. "NEFT stands for National..."       score: 0.91  ← not useful!
    3. "Transfer charges policy..."        score: 0.89
    4. "NEFT was introduced in 2005..."    score: 0.88  ← history, not fees!
    ...20 results

Step 2: Re-rank with cross-encoder (accurate)

    A cross-encoder sees the (question, document) PAIR together
    and directly judges relevance:

    ("NEFT charges?", "NEFT fee schedule...")        → 0.97 ✅
    ("NEFT charges?", "NEFT stands for National...") → 0.15 ❌ Junk!
    ("NEFT charges?", "Transfer charges policy...")   → 0.91 ✅
    ("NEFT charges?", "NEFT was introduced in 2005") → 0.08 ❌ Junk!

Step 3: Return top 3-5 after re-ranking

    1. "NEFT fee schedule..."         ✅
    2. "Transfer charges policy..."   ✅ (was #3, promoted!)
    3. "RBI guidelines on NEFT..."    ✅ (was #7, promoted!)

    Irrelevant results filtered out! Much better context for the LLM.
```

**Why not use the cross-encoder for everything?**
- Cross-encoder compares query with EACH document individually → slow
- Bi-encoder (regular embedding) pre-computes document vectors → fast search
- Best approach: Bi-encoder for fast initial retrieval → Cross-encoder for re-ranking top results

```python
# Hybrid retrieval with re-ranking in pseudocode
from sentence_transformers import CrossEncoder

# Step 1: Fast retrieval (get 20 candidates)
dense_results = vector_db.similarity_search(query, k=20)
sparse_results = bm25_search(query, k=20)

# Step 2: Merge with RRF
merged = reciprocal_rank_fusion(dense_results, sparse_results)

# Step 3: Re-rank top 20 with cross-encoder
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
pairs = [(query, doc.content) for doc in merged[:20]]
scores = reranker.predict(pairs)

# Step 4: Sort by re-ranker score, return top 5
reranked = sorted(zip(scores, merged), reverse=True)
final_results = [doc for score, doc in reranked[:5]]
```

### Retrieval Strategy Decision Guide

```
┌──────────────┬──────────────────────┬──────────────────────────────┐
│ Strategy     │ How It Works         │ When to Use                  │
├──────────────┼──────────────────────┼──────────────────────────────┤
│ Dense only   │ Embedding vectors    │ General concept questions.   │
│              │                      │ "How does KYC work?"         │
├──────────────┼──────────────────────┼──────────────────────────────┤
│ Sparse only  │ BM25 keywords        │ Exact IDs, codes, names.     │
│              │                      │ "Find policy #RBI-2024-27"   │
├──────────────┼──────────────────────┼──────────────────────────────┤
│ Hybrid       │ Dense + Sparse       │ Production systems.          │
│              │ merged with RRF      │ Best overall quality.        │
├──────────────┼──────────────────────┼──────────────────────────────┤
│ + Re-ranking │ Cross-encoder        │ When precision matters       │
│              │ second pass          │ (medical, legal, finance).   │
└──────────────┴──────────────────────┴──────────────────────────────┘
```

---

## 8. Augmentation — Prompt Engineering for RAG

The augmentation step is where you assemble the final prompt. How you structure it dramatically affects answer quality.

### Bad vs Good Prompts

```python
# ❌ BAD: Just dump everything together
prompt = f"{retrieved_text} {question}"
# LLM gets confused, might mix up sources, might hallucinate

# ✅ GOOD: Structured with clear instructions
prompt = f"""
You are a banking support assistant. Answer the customer's
question using ONLY the provided context.

If the answer is not in the context, say:
"I don't have this information. Please contact support."

CONTEXT:
---
[Source: Fee Schedule v3.2, Page 12]
{retrieved_chunk_1}

[Source: Digital Payments Policy, Section 4.1]
{retrieved_chunk_2}
---

QUESTION: {user_question}

RULES:
- Cite the source document for each fact
- If the context has conflicting information, mention both
- Use bullet points for clarity
- Do NOT use any knowledge outside the provided context
"""
```

### The "Lost in the Middle" Problem

Research shows that LLMs pay more attention to the beginning and end of the context, and tend to **forget information in the middle**:

```
Attention level when reading 7 chunks:

    HIGH ████████                      ← Chunk 1 (reads carefully)
         ██████                        ← Chunk 2
         ████                          ← Chunk 3
    LOW  ██                            ← Chunk 4 (LOST! Ignored!)
         ████                          ← Chunk 5
         ██████                        ← Chunk 6
    HIGH ████████                      ← Chunk 7 (reads carefully)
```

**Solutions:**
1. Put the most relevant chunk **first** or **last** in the context
2. Use **fewer, higher-quality chunks** (3-5 is better than 10-15)
3. Re-rank to ensure only truly relevant chunks are included

### Metadata Augmentation

Don't just pass raw text — add metadata that helps the LLM:

```python
# Instead of just the text content, include metadata:
context = f"""
[Document: {chunk.source_file}]
[Section: {chunk.section_heading}]
[Last Updated: {chunk.updated_date}]
[Relevance Score: {chunk.similarity_score:.2f}]

{chunk.content}
"""

# This helps the LLM:
# - Cite specific sources
# - Prefer recent documents over old ones
# - Judge confidence based on relevance score
```

---

## 9. Generation — Controlling the LLM Output

### Temperature Setting

```
Temperature controls randomness in the LLM's output:

    Temperature 0.0 (DETERMINISTIC):
    → Same answer every time
    → "NEFT charges are ₹2.50 for transactions up to ₹10,000."
    → Best for RAG! Factual, consistent. ✅

    Temperature 0.7 (CREATIVE):
    → Different answer each time
    → "NEFT transfers typically cost around ₹2-3, though this
        can vary based on your bank and transfer amount."
    → Adds unreliable fluff. Bad for RAG. ❌

    FOR RAG: Always use temperature 0.0 to 0.2
```

### Grounding Techniques (Preventing Hallucination)

```
TECHNIQUE 1: Explicit Constraint
    "Answer ONLY using the provided context.
     Never add information from your training data."

TECHNIQUE 2: Citation Requirement
    "For each statement, cite the source in [brackets].
     Example: The fee is ₹2.50 [Fee Schedule, Page 12]"

    → If the LLM can't cite a source, it's probably hallucinating.

TECHNIQUE 3: Confidence Fallback
    "If you cannot find the answer in the context, respond:
     'I don't have this information in my current documents.
      Please contact our support team at 1800-XXX-XXXX.'"

TECHNIQUE 4: Structured Output
    "Respond in this JSON format:
     {
       'answer': '...',
       'sources': ['doc1.pdf, page 5', 'doc2.pdf, section 3'],
       'confidence': 'high/medium/low'
     }"
```

### Model Selection for Generation

In production, you often use **different models for different steps**:

```
┌────────────────────┬─────────────────────┬──────────────────┐
│ Step               │ Model Choice        │ Why              │
├────────────────────┼─────────────────────┼──────────────────┤
│ Query routing      │ Small/Fast model    │ Simple yes/no    │
│ (classify intent)  │ (Gemini Flash)      │ decision         │
├────────────────────┼─────────────────────┼──────────────────┤
│ Re-ranking         │ Cross-encoder       │ Specialized for  │
│                    │ (dedicated model)   │ relevance scoring│
├────────────────────┼─────────────────────┼──────────────────┤
│ Final answer       │ Large/Smart model   │ Quality matters  │
│ generation         │ (GPT-4, Gemini Pro) │ for user-facing  │
└────────────────────┴─────────────────────┴──────────────────┘

This "cascade" approach balances cost and quality:
  - Cheap models for simple decisions (routing, classification)
  - Expensive models only for the final answer
```

---

## 10. Building Your First RAG System (Code)

### Level 1: Minimal RAG (30 lines, understand the concept)

```python
"""
Minimal RAG: No frameworks, just raw Python.
Purpose: Understand the concept clearly.
"""
import numpy as np
from openai import OpenAI

client = OpenAI()

# === OFFLINE PHASE: Prepare your documents ===

documents = [
    "NEFT transaction fee: ₹2.50 for amounts up to ₹10,000. "
    "₹5 for ₹10,001 to ₹1 lakh. Free on Sundays.",

    "RTGS minimum amount is ₹2 lakh. Charges: ₹20 for ₹2-5 lakh, "
    "₹40 for above ₹5 lakh. Available 24x7.",

    "IMPS allows instant transfer up to ₹5 lakh. "
    "Charges: ₹5 for up to ₹1 lakh, ₹15 above ₹1 lakh.",

    "Savings account minimum balance: ₹10,000 for metro branches, "
    "₹5,000 for rural. Penalty: ₹500 per quarter if not maintained.",
]

# Embed all documents (one-time)
def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

doc_embeddings = [get_embedding(doc) for doc in documents]


# === ONLINE PHASE: Answer a question ===

def answer_question(question):
    # Step 1: RETRIEVE — find relevant documents
    question_embedding = get_embedding(question)

    similarities = []
    for i, doc_emb in enumerate(doc_embeddings):
        sim = np.dot(question_embedding, doc_emb) / (
            np.linalg.norm(question_embedding) * np.linalg.norm(doc_emb)
        )
        similarities.append((sim, documents[i]))

    # Sort by similarity, get top 2
    similarities.sort(reverse=True)
    top_docs = [doc for _, doc in similarities[:2]]

    # Step 2: AUGMENT — build the prompt
    context = "\n\n".join(top_docs)
    prompt = f"""Answer based ONLY on the following context.
If the answer is not in the context, say "I don't know."

Context:
{context}

Question: {question}"""

    # Step 3: GENERATE — get LLM answer
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )
    return response.choices[0].message.content


# Test it!
print(answer_question("What are NEFT charges?"))
# → "NEFT charges are ₹2.50 for amounts up to ₹10,000,
#    ₹5 for ₹10,001 to ₹1 lakh, and free on Sundays."

print(answer_question("What is the minimum balance for savings?"))
# → "The minimum balance is ₹10,000 for metro branches
#    and ₹5,000 for rural branches..."

print(answer_question("What is the weather today?"))
# → "I don't know. This information is not in the provided context."
```

### Level 2: Production-Ready RAG with LangChain + pgvector

```python
"""
Production RAG with:
- LangChain for orchestration
- pgvector for vector storage
- Google Gemini for embeddings + generation
- Proper document loading and chunking
"""
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import PGVector
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


# ============================================================
# OFFLINE PHASE: Load, Chunk, Embed, Store
# ============================================================

# 1. Load documents from PDF
loader = PyPDFLoader("banking_policies.pdf")
raw_documents = loader.load()
print(f"Loaded {len(raw_documents)} pages")

# 2. Chunk the documents
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,            # ~500 characters per chunk
    chunk_overlap=50,          # 50 chars overlap between chunks
    separators=["\n\n", "\n", ". ", " ", ""]
)
chunks = splitter.split_documents(raw_documents)
print(f"Created {len(chunks)} chunks")

# 3. Create embedding model
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-exp-03-07",
    task_type="RETRIEVAL_DOCUMENT"  # optimized for document storage
)

# 4. Store in pgvector (embeds automatically)
CONNECTION_STRING = "postgresql://user:password@localhost:5432/ragdb"

vectorstore = PGVector.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="banking_policies",
    connection_string=CONNECTION_STRING,
)
print("Documents embedded and stored!")


# ============================================================
# ONLINE PHASE: Retrieve + Generate
# ============================================================

# 5. Create retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 5,                         # return top 5 chunks
        # "score_threshold": 0.80,       # optional: minimum similarity
    }
)

# 6. Create the prompt template
prompt_template = PromptTemplate(
    template="""You are a banking support assistant. Answer the
customer's question using ONLY the provided context.

If the answer is not in the context, say:
"I don't have this information. Please contact our support team."

Context:
{context}

Question: {question}

Answer (cite sources where possible):""",
    input_variables=["context", "question"]
)

# 7. Create the RAG chain
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.0  # deterministic for factual answers
)

rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",           # "stuff" = put all chunks in one prompt
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt_template},
    return_source_documents=True  # include sources in response
)

# 8. Ask questions!
result = rag_chain.invoke({"query": "What are NEFT charges?"})

print("Answer:", result["result"])
print("\nSources:")
for doc in result["source_documents"]:
    print(f"  - Page {doc.metadata.get('page', '?')}: {doc.page_content[:100]}...")
```

### Level 3: FastAPI RAG Service

```python
"""
RAG as a web service with FastAPI.
Production-ready API endpoint.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager

# --- Models ---
class QuestionRequest(BaseModel):
    question: str
    top_k: int = 5

class AnswerResponse(BaseModel):
    answer: str
    sources: list[dict]
    confidence: str

# --- Globals (initialized at startup) ---
vectorstore = None
llm = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize RAG components at startup."""
    global vectorstore, llm

    from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
    from langchain_community.vectorstores import PGVector

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-exp-03-07"
    )

    vectorstore = PGVector(
        embedding_function=embeddings,
        collection_name="banking_policies",
        connection_string="postgresql://user:pass@localhost:5432/ragdb",
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.0
    )

    print("RAG system initialized!")
    yield
    print("Shutting down RAG system.")

app = FastAPI(title="RAG API", lifespan=lifespan)


@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """RAG endpoint: retrieve relevant docs and generate answer."""

    # Step 1: RETRIEVE
    docs_with_scores = vectorstore.similarity_search_with_score(
        request.question, k=request.top_k
    )

    if not docs_with_scores:
        raise HTTPException(status_code=404, detail="No relevant documents found")

    # Step 2: AUGMENT
    context_parts = []
    sources = []
    for doc, score in docs_with_scores:
        context_parts.append(f"[Score: {1-score:.2f}] {doc.page_content}")
        sources.append({
            "content": doc.page_content[:200],
            "source": doc.metadata.get("source", "unknown"),
            "page": doc.metadata.get("page", "unknown"),
            "similarity": round(1 - score, 3)
        })

    context = "\n\n".join(context_parts)

    prompt = f"""You are a banking support assistant.
Answer using ONLY the provided context.
If unsure, say "I don't have this information."

Context:
{context}

Question: {request.question}

Answer:"""

    # Step 3: GENERATE
    response = llm.invoke(prompt)

    # Determine confidence based on top similarity score
    top_score = 1 - docs_with_scores[0][1]
    confidence = "high" if top_score > 0.85 else "medium" if top_score > 0.70 else "low"

    return AnswerResponse(
        answer=response.content,
        sources=sources,
        confidence=confidence
    )


@app.post("/ingest")
async def ingest_document(file_path: str):
    """Ingest a new document into the RAG system."""
    from langchain_community.document_loaders import PyPDFLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    # Load and chunk
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50
    )
    chunks = splitter.split_documents(docs)

    # Add to vectorstore
    vectorstore.add_documents(chunks)

    return {"status": "success", "chunks_added": len(chunks)}
```

---

## 11. Production Patterns — From Prototype to Real System

### Pattern 1: Multi-Level Caching

The most impactful optimization. Don't re-compute what you've already answered:

```
User asks: "What are NEFT charges?"
                    │
                    ▼
    ┌───────────────────────────────┐
    │  Level 1: EXACT CACHE (Hash) │  ← ~0ms, $0.00
    │                               │
    │  hash("what are neft charges")│
    │  = "a1b2c3..."               │
    │  Found in cache? → Return!   │
    │  Not found? → Level 2        │
    └───────────────┬───────────────┘
                    │ miss
                    ▼
    ┌───────────────────────────────┐
    │  Level 2: SEMANTIC CACHE     │  ← ~50ms, $0.001
    │  (Embedding Similarity)       │
    │                               │
    │  Similar question asked       │
    │  before? (≥ 0.95 similarity)  │
    │  "NEFT transfer costs?" → YES │
    │  Return cached answer!        │
    │  Not found? → Level 3        │
    └───────────────┬───────────────┘
                    │ miss
                    ▼
    ┌───────────────────────────────┐
    │  Level 3: FULL RAG PIPELINE  │  ← ~2s, $0.01
    │                               │
    │  Embed → Search → Augment →  │
    │  Generate                     │
    │                               │
    │  Save answer to cache for    │
    │  future questions!            │
    └───────────────────────────────┘
```

```python
import hashlib

def generate_query_hash(question: str) -> str:
    """Create a hash of the normalized question."""
    normalized = question.lower().strip()
    return hashlib.sha256(normalized.encode()).hexdigest()

async def ask_with_cache(question: str):
    # Level 1: Exact hash lookup
    query_hash = generate_query_hash(question)
    cached = await redis.get(f"rag:exact:{query_hash}")
    if cached:
        return json.loads(cached)  # Instant!

    # Level 2: Semantic similarity cache
    question_embedding = embedder.embed_text(question)
    similar_cached = await vector_cache.search(
        question_embedding, threshold=0.95
    )
    if similar_cached:
        return similar_cached  # Very fast!

    # Level 3: Full RAG pipeline
    answer = await full_rag_pipeline(question)

    # Save to both caches
    await redis.set(f"rag:exact:{query_hash}", json.dumps(answer), ex=3600)
    await vector_cache.store(question_embedding, answer)

    return answer
```

### Pattern 2: Query Routing

Not every question needs the same pipeline. Route to the best handler:

```
User question
      │
      ▼
┌──────────────┐
│ Query Router │ ← Small/fast LLM classifies the question
│ (LLM or      │
│  classifier) │
└──────┬───────┘
       │
       ├── "factual lookup" ──→ Standard RAG pipeline
       │                        (retrieve → augment → generate)
       │
       ├── "comparison" ──→ Multi-query RAG
       │                    (retrieve for EACH item, then compare)
       │
       ├── "aggregation" ──→ SQL/structured query
       │   "total loans"     (don't need RAG, query database directly)
       │
       └── "chitchat" ──→ Direct LLM response
           "hello!"       (no retrieval needed)
```

```python
async def route_query(question: str) -> str:
    """Use a fast LLM to classify the question type."""
    routing_prompt = f"""Classify this question into one category:
    - FACTUAL: needs to look up specific information
    - COMPARISON: compares two or more things
    - AGGREGATION: needs counting, totals, statistics
    - CHITCHAT: greeting, thanks, off-topic

    Question: {question}
    Category:"""

    response = await fast_llm.invoke(routing_prompt)  # Gemini Flash
    category = response.content.strip().upper()

    if category == "FACTUAL":
        return await standard_rag(question)
    elif category == "COMPARISON":
        return await multi_query_rag(question)
    elif category == "AGGREGATION":
        return await sql_query(question)
    else:
        return await direct_llm(question)
```

### Pattern 3: Multi-Query RAG

One question might need information from multiple angles:

```
Question: "Compare NEFT and RTGS for large transfers"

Single query RAG (basic):
    Search: "compare NEFT RTGS large transfers" → might miss details

Multi-query RAG (better):
    LLM generates multiple search queries:
      Query 1: "NEFT charges and limits"
      Query 2: "RTGS charges and limits"
      Query 3: "NEFT vs RTGS processing time"

    Search each → Merge all results → Deduplicate → Generate answer

    Result: More comprehensive answer with details about both!
```

```python
async def multi_query_rag(question: str):
    # Step 1: Generate multiple search queries
    expansion_prompt = f"""Generate 3 different search queries that
would help answer this question from different angles:

Question: {question}

Output as a numbered list:"""

    queries_response = await llm.invoke(expansion_prompt)
    queries = parse_numbered_list(queries_response.content)

    # Step 2: Search for each query
    all_docs = []
    for query in queries:
        docs = vectorstore.similarity_search(query, k=3)
        all_docs.extend(docs)

    # Step 3: Deduplicate by content
    unique_docs = deduplicate(all_docs)

    # Step 4: Generate answer with all retrieved context
    return await generate_answer(question, unique_docs)
```

### Pattern 4: Metadata Filtering

Don't just search by similarity — filter by metadata first:

```
Scenario: Bank has policies from 2020, 2022, and 2024.
Question about current policy should ONLY search 2024 docs.

    Without filtering:
    Results: [2020 doc (0.95), 2024 doc (0.93), 2022 doc (0.91)]
    → Might answer with OUTDATED 2020 info! ❌

    With filtering:
    Filter: year >= 2024
    Results: [2024 doc (0.93)]
    → Always answers with current info! ✅
```

```python
# pgvector with metadata filtering
results = vectorstore.similarity_search(
    query="loan penalty policy",
    k=5,
    filter={
        "year": {"$gte": 2024},           # only recent docs
        "department": "retail_banking",     # only relevant dept
        "status": "active"                 # only active policies
    }
)
```

### Pattern 5: Streaming Responses

Don't make users wait for the full answer — stream it word by word:

```python
from fastapi.responses import StreamingResponse

@app.post("/ask/stream")
async def ask_stream(request: QuestionRequest):
    """Stream the RAG response token by token."""

    # Retrieve context (non-streaming, fast)
    docs = vectorstore.similarity_search(request.question, k=5)
    context = "\n\n".join([d.page_content for d in docs])

    prompt = f"Context:\n{context}\n\nQuestion: {request.question}\nAnswer:"

    async def generate():
        # Stream the LLM response
        async for chunk in llm.astream(prompt):
            if chunk.content:
                yield chunk.content

    return StreamingResponse(generate(), media_type="text/plain")
```

---

## 12. Advanced RAG Architectures

### Architecture 1: RAG Fusion

Combine multiple retrieval strategies and let an LLM pick the best results:

```
Question: "What's the maximum UPI transaction limit?"
                    │
         ┌──────────┼──────────┐
         ▼          ▼          ▼
    Dense Search  BM25 Search  Generated
    (embeddings)  (keywords)   Sub-queries
         │          │          │
         ▼          ▼          ▼
    [Results A]  [Results B]  [Results C]
         │          │          │
         └──────────┼──────────┘
                    ▼
        Reciprocal Rank Fusion (merge)
                    │
                    ▼
            Top 5 best results
                    │
                    ▼
              LLM generates answer
```

### Architecture 2: Self-RAG (Self-Reflective RAG)

The LLM checks its own work:

```
Question → Retrieve docs → Generate answer
                                │
                                ▼
                    ┌─────────────────────┐
                    │ Self-Check:         │
                    │                     │
                    │ 1. Is retrieval     │
                    │    needed? (yes/no) │
                    │                     │
                    │ 2. Are retrieved    │
                    │    docs relevant?   │
                    │    (if no, retry)   │
                    │                     │
                    │ 3. Is my answer     │
                    │    supported by     │
                    │    the docs?        │
                    │    (if no, revise)  │
                    │                     │
                    │ 4. Is my answer     │
                    │    useful?          │
                    │    (if no, regen)   │
                    └─────────────────────┘
```

### Architecture 3: CRAG (Corrective RAG)

If retrieved documents aren't good enough, fall back to web search:

```
Question → Retrieve from your DB
                │
                ▼
    ┌──────────────────────┐
    │ Relevance Check:     │
    │ Are results good?    │
    ├──────────────────────┤
    │                      │
    │ CORRECT (>0.85):     │──→ Use retrieved docs → Generate
    │ Results are great!   │
    │                      │
    │ AMBIGUOUS (0.5-0.85):│──→ Use retrieved docs + web search
    │ Partially relevant   │    → Generate from both
    │                      │
    │ INCORRECT (<0.5):    │──→ Ignore DB results
    │ Nothing useful       │    → Web search only → Generate
    │                      │
    └──────────────────────┘
```

### Architecture 4: Agentic RAG

An AI agent decides which tools to use, including RAG as one tool:

```
Question: "Compare our NEFT fees with ICICI's and recommend changes"
                    │
                    ▼
            ┌──────────────┐
            │   AI Agent   │
            │  (decides    │
            │   actions)   │
            └──────┬───────┘
                   │
    Step 1: "I need our fee data"
            → Tool: RAG search internal docs
            → Found: "Our NEFT fee: ₹2.50"
                   │
    Step 2: "I need ICICI's fee data"
            → Tool: Web search
            → Found: "ICICI NEFT fee: ₹2.00"
                   │
    Step 3: "I need transaction volume data"
            → Tool: SQL database query
            → Found: "1.2M NEFT transactions/month"
                   │
    Step 4: "Now I can analyze and recommend"
            → Tool: LLM analysis
            → "Our NEFT fee is ₹0.50 higher than ICICI.
               With 1.2M monthly transactions, reducing
               to ₹2.00 would cost ₹600K/month in revenue
               but could increase transaction volume by..."
```

```python
# Agentic RAG with LangChain
from langchain.agents import create_tool_calling_agent
from langchain.tools import Tool

# Define tools the agent can use
tools = [
    Tool(
        name="search_internal_docs",
        description="Search internal banking policy documents",
        func=lambda q: vectorstore.similarity_search(q, k=3)
    ),
    Tool(
        name="search_web",
        description="Search the internet for public information",
        func=lambda q: web_search(q)
    ),
    Tool(
        name="query_database",
        description="Run SQL queries on the transaction database",
        func=lambda q: run_sql(q)
    ),
    Tool(
        name="calculate",
        description="Perform mathematical calculations",
        func=lambda expr: eval(expr)  # use a safe math parser in production!
    ),
]

agent = create_tool_calling_agent(llm, tools, prompt)
result = agent.invoke({"input": question})
```

### Architecture Comparison

```
┌──────────────┬─────────────────────┬────────────────────────────┐
│ Architecture │ Complexity          │ Use When                   │
├──────────────┼─────────────────────┼────────────────────────────┤
│ Basic RAG    │ Simple              │ FAQ, simple lookups        │
│ RAG Fusion   │ Medium              │ Need comprehensive results │
│ Self-RAG     │ Medium-High         │ Need verified, reliable    │
│              │                     │ answers (medical, legal)   │
│ CRAG         │ Medium              │ Internal + external data   │
│ Agentic RAG  │ High                │ Complex multi-step tasks   │
└──────────────┴─────────────────────┴────────────────────────────┘
```

---

## 13. Evaluation — Measuring RAG Quality

### The RAGAS Framework

RAGAS (Retrieval Augmented Generation Assessment) has 4 key metrics:

```
Given:
  Question:  "What are NEFT charges for a ₹5000 transfer?"
  Retrieved: "NEFT fee: ₹2.50 for up to ₹10,000"
  Answer:    "NEFT charges for ₹5000 are ₹2.50"
  Truth:     "₹2.50"
```

#### Metric 1: Faithfulness

"Is the answer **supported** by the retrieved context?"

```
Answer says: "₹2.50"
Context says: "₹2.50 for up to ₹10,000"
₹5000 < ₹10,000 → ₹2.50 is correct ✅
Score: 1.0 (fully supported)

BAD: Answer says "₹5" but context says "₹2.50"
Score: 0.0 (HALLUCINATED!)

This catches when the LLM makes things up.
```

#### Metric 2: Answer Relevancy

"Does the answer actually **address** the question?"

```
Question: "What are NEFT charges?"
Answer: "NEFT charges are ₹2.50" ✅ Directly addresses the question
Score: 0.95

BAD: Answer talks about RTGS instead of NEFT
Score: 0.2 (wrong topic!)
```

#### Metric 3: Context Precision

"Are the **retrieved documents** actually relevant?"

```
Retrieved 3 chunks:
  Chunk 1: "NEFT fee schedule..."    ← relevant ✅
  Chunk 2: "RTGS charges..."         ← not relevant ❌
  Chunk 3: "NEFT history..."         ← not relevant ❌

Score: 0.33 (only 1 of 3 was useful)

This tells you: your retrieval needs improvement.
```

#### Metric 4: Context Recall

"Did we retrieve **ALL** the info needed to answer?"

```
Ground truth needs: "₹2.50 for up to ₹10,000"
Retrieved context has this info? ✅ YES
Score: 1.0

BAD: Ground truth also says "free on weekends"
but we didn't retrieve that chunk.
Score: 0.5 (missed half the answer!)

This tells you: your chunking or top-k might be too restrictive.
```

### Evaluation Summary

```
                 RETRIEVAL QUALITY          GENERATION QUALITY
                 ┌─────────────────┐       ┌──────────────────┐
                 │ Context          │       │ Faithfulness     │
                 │ Precision +      │       │ + Answer         │
                 │ Recall           │       │ Relevancy        │
                 └─────────────────┘       └──────────────────┘
                       │                          │
                       │                          │
            If bad, fix:                If bad, fix:
            - Chunking strategy         - Prompt template
            - Embedding model           - Temperature setting
            - Retrieval strategy        - Model choice
            - top_k value               - Grounding instructions
```

```python
# Evaluating with RAGAS
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
from datasets import Dataset

# Prepare evaluation data
eval_data = {
    "question": ["What are NEFT charges?"],
    "answer": ["NEFT charges are ₹2.50 for up to ₹10,000"],
    "contexts": [["NEFT fee: ₹2.50 for amounts up to ₹10,000"]],
    "ground_truth": ["₹2.50 for up to ₹10,000"]
}

dataset = Dataset.from_dict(eval_data)

results = evaluate(
    dataset=dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall]
)
print(results)
# {'faithfulness': 1.0, 'answer_relevancy': 0.95,
#  'context_precision': 1.0, 'context_recall': 1.0}
```

---

## 14. Common Problems and Solutions

### Problem 1: Hallucination (LLM Makes Stuff Up)

```
Symptom: Answer contains facts not in the retrieved context.

Causes:
  - Prompt doesn't restrict LLM to context only
  - Retrieved docs are vaguely related, LLM fills gaps
  - Temperature too high

Solutions:
  ┌────────────────────────────────────────────────────────┐
  │ 1. Add strict instruction:                             │
  │    "Answer ONLY from the provided context.             │
  │     Do NOT use your training knowledge."               │
  │                                                        │
  │ 2. Set temperature = 0.0                               │
  │                                                        │
  │ 3. Require citations:                                  │
  │    "Cite the source for every fact."                   │
  │    (If LLM can't cite → it's making it up)            │
  │                                                        │
  │ 4. High similarity threshold (≥ 0.85):                │
  │    Only return docs that truly match the question.     │
  │                                                        │
  │ 5. Add a "fallback" response:                         │
  │    If no good docs found, say "I don't know" instead   │
  │    of guessing.                                        │
  └────────────────────────────────────────────────────────┘
```

### Problem 2: Wrong Documents Retrieved

```
Symptom: LLM gives wrong answer because it received wrong context.

Causes:
  - Chunks too large (meaning diluted)
  - Chunks too small (missing context)
  - Embedding model not good enough
  - No metadata filtering

Solutions:
  ┌────────────────────────────────────────────────────────┐
  │ 1. Experiment with chunk sizes (try 300, 500, 1000)   │
  │                                                        │
  │ 2. Use hybrid search (dense + BM25)                   │
  │                                                        │
  │ 3. Add re-ranking step (cross-encoder)                │
  │                                                        │
  │ 4. Add metadata filters:                              │
  │    - Department, date, document type                   │
  │    - Filter BEFORE vector search                       │
  │                                                        │
  │ 5. Try a better embedding model                       │
  │    (e.g., text-embedding-3-large vs 3-small)          │
  └────────────────────────────────────────────────────────┘
```

### Problem 3: "Lost in the Middle"

```
Symptom: LLM ignores information in the middle of the context.

Solutions:
  ┌────────────────────────────────────────────────────────┐
  │ 1. Reduce number of chunks (3-5 instead of 10-15)    │
  │                                                        │
  │ 2. Put most relevant chunk FIRST in context           │
  │                                                        │
  │ 3. Use re-ranking to ensure only good chunks are sent │
  │                                                        │
  │ 4. Use map-reduce: process each chunk separately,     │
  │    then combine answers                                │
  └────────────────────────────────────────────────────────┘
```

### Problem 4: Too Slow

```
Symptom: Response takes > 5 seconds.

Bottleneck analysis:
  Embedding the question: ~100ms
  Vector search:          ~50ms
  LLM generation:         ~2-5s    ← usually the bottleneck
  Total:                  ~2.5-5.5s

Solutions:
  ┌────────────────────────────────────────────────────────┐
  │ 1. Cache frequent queries (hash + semantic cache)     │
  │                                                        │
  │ 2. Use HNSW index (approximate but fast search)       │
  │                                                        │
  │ 3. Stream the response (user sees partial answer fast) │
  │                                                        │
  │ 4. Use a faster LLM for simple queries (Flash models) │
  │                                                        │
  │ 5. Pre-compute answers for known frequent questions   │
  └────────────────────────────────────────────────────────┘
```

### Problem 5: Too Expensive

```
Cost breakdown per query:
  Embedding:  $0.0001 (cheap)
  Vector DB:  $0.0001 (cheap)
  LLM call:   $0.01-0.10 (EXPENSIVE — this is 99% of cost)

Solutions:
  ┌────────────────────────────────────────────────────────┐
  │ 1. Cache aggressively (80%+ of queries are repeated)  │
  │                                                        │
  │ 2. Use cheap models for routing/classification         │
  │    (Flash for routing, Pro only for final answer)      │
  │                                                        │
  │ 3. Reduce context size (fewer/shorter chunks)          │
  │                                                        │
  │ 4. Batch embedding calls (not one-by-one)              │
  │                                                        │
  │ 5. Use open-source models for non-critical steps       │
  └────────────────────────────────────────────────────────┘
```

### Complete Troubleshooting Flowchart

```
RAG giving bad answers?
        │
        ├── Is the RETRIEVED context relevant?
        │       │
        │       ├── NO → Fix RETRIEVAL
        │       │         • Better chunking
        │       │         • Hybrid search
        │       │         • Re-ranking
        │       │         • Better embedding model
        │       │
        │       └── YES → Fix GENERATION
        │                 • Better prompt template
        │                 • Lower temperature
        │                 • Stricter instructions
        │                 • Better LLM model
        │
        └── Is the answer too slow / expensive?
                │
                ├── SLOW → Add caching + streaming
                │
                └── EXPENSIVE → Add caching + cheaper models for routing
```

---

## 15. RAG vs Fine-Tuning — When to Use Which

```
OPTION A: RAG
    Keep the LLM as-is, search company docs before each question.
    Like: Giving an employee a REFERENCE MANUAL.

OPTION B: Fine-Tuning
    Retrain the LLM on your company data.
    Like: TRAINING the employee for 6 months.
```

```
┌───────────────────┬─────────────────────┬──────────────────────┐
│                   │ RAG                 │ Fine-Tuning          │
├───────────────────┼─────────────────────┼──────────────────────┤
│ Setup time        │ Days                │ Weeks                │
│ Setup cost        │ Low                 │ High (GPU compute)   │
├───────────────────┼─────────────────────┼──────────────────────┤
│ Data changes      │ Update DB, instant  │ Retrain model        │
│                   │                     │ (hours to days)      │
├───────────────────┼─────────────────────┼──────────────────────┤
│ Best for          │ Facts, lookups,     │ Tone, style, format, │
│                   │ specific data       │ domain behavior      │
├───────────────────┼─────────────────────┼──────────────────────┤
│ Hallucination     │ Lower (has source   │ Higher (from memory) │
│                   │ documents as proof) │                      │
├───────────────────┼─────────────────────┼──────────────────────┤
│ Traceability      │ Can cite exact      │ Cannot explain where │
│                   │ source document     │ answer came from     │
├───────────────────┼─────────────────────┼──────────────────────┤
│ Data privacy      │ Data stays in DB,   │ Data baked into      │
│                   │ never in model      │ model weights        │
├───────────────────┼─────────────────────┼──────────────────────┤
│ Cost per query    │ Embedding + DB +    │ Just LLM call        │
│                   │ LLM = moderate      │ (but hosting costly) │
├───────────────────┼─────────────────────┼──────────────────────┤
│ Maintenance       │ Keep docs updated   │ Periodic retraining  │
└───────────────────┴─────────────────────┴──────────────────────┘
```

### Decision Framework

```
Choose RAG when:
  ✅ Data changes frequently (regulations, policies, prices)
  ✅ You need to cite sources (compliance, legal, medical)
  ✅ Data is private/sensitive (stays in your DB)
  ✅ You need answers about specific documents/records
  ✅ Quick to deploy, easy to maintain

Choose Fine-Tuning when:
  ✅ You want the model to "think" in your domain's style
  ✅ You need consistent formatting/structure in outputs
  ✅ The knowledge is stable and doesn't change often
  ✅ You need the model to understand domain jargon deeply

Choose BOTH when:
  ✅ Fine-tune for domain style + RAG for current facts
  ✅ Example: Fine-tune to write like a banker,
     RAG to look up current rates
```

### Fintech Interview Answer

> "For a Fintech product, I'd choose RAG because: (1) Financial regulations change frequently — RAG lets us update documents without retraining; (2) Compliance requires citing sources — RAG naturally provides this; (3) Customer data must stay private — with RAG it stays in our database, never enters model weights; (4) We can use RAG + fine-tuning together: fine-tune for our communication style, RAG for current data lookup."

---

## 16. Production Checklist

### Before Going Live

```
INFRASTRUCTURE:
  □ Vector database deployed and indexed (pgvector/Pinecone)
  □ HNSW or IVF index created on vector columns
  □ Connection pooling configured for database
  □ Redis/cache layer for frequent queries
  □ API rate limiting on LLM calls
  □ Fallback model if primary LLM is down

QUALITY:
  □ Chunk size optimized (tested 300/500/1000)
  □ Overlap configured (10-20% of chunk size)
  □ Similarity threshold tuned (typically 0.75-0.90)
  □ Top-k value tuned (typically 3-5)
  □ Prompt template tested with edge cases
  □ Hallucination guardrails in prompt
  □ "I don't know" fallback for low-confidence answers

EVALUATION:
  □ RAGAS evaluation dataset created (50-100 Q&A pairs)
  □ Faithfulness > 0.85
  □ Answer relevancy > 0.80
  □ Context precision > 0.75
  □ Context recall > 0.75
  □ Automated evaluation pipeline (run on every change)

MONITORING:
  □ Log every query + retrieved docs + answer
  □ Track latency per step (embed, search, generate)
  □ Track cost per query
  □ Alert on low similarity scores (no good docs found)
  □ Track user feedback (thumbs up/down)
  □ Dashboard for retrieval quality over time

SECURITY:
  □ No PII leaking across user boundaries
  □ Access control on document collections
  □ Input sanitization (prompt injection prevention)
  □ API authentication on RAG endpoints
  □ Audit trail for compliance

DOCUMENT MANAGEMENT:
  □ Pipeline to re-embed when documents change
  □ Version tracking for document collections
  □ Stale document detection and removal
  □ New document ingestion API
```

### Production Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     PRODUCTION RAG SYSTEM                        │
│                                                                  │
│  ┌──────────┐    ┌───────────┐    ┌──────────────┐             │
│  │  User /  │───→│   API     │───→│   Query      │             │
│  │  Client  │    │  Gateway  │    │   Router     │             │
│  └──────────┘    │  (auth,   │    │  (classify   │             │
│       ▲          │   rate    │    │   intent)    │             │
│       │          │   limit)  │    └──────┬───────┘             │
│       │          └───────────┘           │                      │
│       │                          ┌───────┼───────┐              │
│       │                          │       │       │              │
│       │                          ▼       ▼       ▼              │
│       │                    ┌────────┐ ┌──────┐ ┌────────┐      │
│       │                    │  RAG   │ │ SQL  │ │ Direct │      │
│       │                    │Pipeline│ │Query │ │  LLM   │      │
│       │                    └───┬────┘ └──────┘ └────────┘      │
│       │                        │                                │
│       │              ┌─────────┼─────────┐                     │
│       │              ▼         ▼         ▼                     │
│       │         ┌────────┐ ┌───────┐ ┌────────┐               │
│       │         │ Cache  │ │Vector │ │  LLM   │               │
│       │         │(Redis) │ │  DB   │ │(Gemini/│               │
│       │         │        │ │(pgvec)│ │ GPT-4) │               │
│       │         └────────┘ └───────┘ └───┬────┘               │
│       │                                   │                     │
│       └───────────────────────────────────┘                     │
│                                                                  │
│  ┌──────────────────────────────────────────────────────┐       │
│  │  OFFLINE PIPELINE (runs when documents change)        │       │
│  │                                                       │       │
│  │  New docs → Load → Chunk → Embed → Store in VectorDB │       │
│  └──────────────────────────────────────────────────────┘       │
│                                                                  │
│  ┌──────────────────────────────────────────────────────┐       │
│  │  MONITORING                                           │       │
│  │                                                       │       │
│  │  Logs → Latency tracking → Cost tracking →            │       │
│  │  Quality metrics → Alerts → Dashboard                 │       │
│  └──────────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────────┘
```

---

## 17. Interview Quick Reference

### "Explain RAG in one sentence"
> RAG retrieves relevant documents from a database, adds them to the LLM's prompt, and lets the LLM generate answers grounded in real data instead of relying on memory.

### "Walk me through a RAG pipeline"
> Offline: documents are chunked, embedded into vectors, and stored in a vector database. Online: the user's question is embedded, similar chunks are retrieved via cosine similarity, added to a prompt template, and an LLM generates the final answer citing those sources.

### "How do embeddings work?"
> Embedding models convert text into fixed-length numerical vectors where semantic similarity is preserved — similar meanings produce vectors that are close together in the vector space, measured by cosine similarity.

### "Dense vs Sparse retrieval?"
> Dense uses embedding vectors and understands meaning ("car" matches "automobile"). Sparse uses keyword matching like BM25 and excels at exact terms, codes, and IDs. Hybrid combines both for best results.

### "How do you evaluate RAG?"
> Using the RAGAS framework: Faithfulness (is the answer supported by context?), Answer Relevancy (does it address the question?), Context Precision (are retrieved docs relevant?), and Context Recall (did we find all needed info?).

### "How do you prevent hallucination?"
> Five techniques: strict prompt instructions ("only use context"), low temperature (0.0), citation requirements, high similarity thresholds, and explicit fallback responses ("I don't know") when confidence is low.

### "RAG vs Fine-tuning?"
> RAG is best for dynamic facts, source citation, and data privacy — data stays in your DB. Fine-tuning is best for teaching style, tone, and stable domain knowledge. In Fintech, RAG is preferred because regulations change frequently and compliance requires traceability.

### "What vector databases have you used?"
> pgvector (PostgreSQL extension) for production when already using PostgreSQL — supports HNSW indexing, cosine/L2/inner product distances, and metadata filtering. ChromaDB for prototyping. Pinecone for managed production deployments.

### "What chunking strategy would you use?"
> Recursive character splitting for most use cases — it tries paragraph boundaries first, then sentences, then words. Chunk size of 500-1000 tokens with 10-20% overlap. For structured documents like HTML or Markdown, document-structure chunking preserves natural sections.

### "How would you make RAG production-ready?"
> Multi-level caching (hash cache → semantic cache → full pipeline), query routing (simple vs complex questions), hybrid retrieval with re-ranking, streaming responses, comprehensive logging, RAGAS evaluation pipeline, and monitoring dashboards for latency, cost, and quality metrics.

---

*This guide covers RAG from first principles to production deployment. The key insight remains simple: don't ask the AI to remember everything — let it look things up, just like you would.*
