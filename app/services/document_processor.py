import os
import pandas as pd
from typing import List, Dict, Any, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import logging
import pypdf

from ..config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentProcessor:
    def __init__(self):
        self.embedding_model_name = settings.EMBEDDING_MODEL_NAME
        self.chunk_size = settings.CHUNK_SIZE
        self.chunk_overlap = settings.CHUNK_OVERLAP
        self.doc_prefix = settings.DEFAULT_DOC_PREFIX
        self.comment_prefix = settings.DEFAULT_COMMENT_PREFIX

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
        )

        logger.info(f"Loading embedding model: {self.embedding_model_name}")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        logger.info("Embedding model loaded successfully")

    def load_document_text(self, file_path: str) -> Optional[str]:
        if os.path.exists(file_path):
            path = file_path
        elif os.path.exists(os.path.join("/app", file_path)):
            path = os.path.join("/app", file_path)
        else:
            logger.error(f"File not found: {file_path}")
            return None

        logger.info(f"Loading document from: {path}")
        ext = os.path.splitext(path)[1].lower()

        try:
            if ext == ".txt":
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
            elif ext == ".pdf":
                reader = pypdf.PdfReader(path)
                content = ""
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        content += page_text + "\n\n"
            else:
                logger.error(f"Unsupported file format: {ext}")
                return None

            logger.info(f"Document loaded: {len(content)} chars")
            return content

        except Exception as e:
            logger.error(f"Error loading document: {e}")
            return None

    def load_comments(self, file_path: str) -> Optional[List[Dict[str, Any]]]:
        if os.path.exists(file_path):
            path = file_path
        elif os.path.exists(os.path.join("/app", file_path)):
            path = os.path.join("/app", file_path)
        else:
            logger.error(f"File not found: {file_path}")
            return None

        logger.info(f"Loading comments from: {path}")

        try:
            comments = []
            with open(path, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    text = line.strip()
                    if text:
                        comments.append({"comment_id": f"C{i+1}", "comment_text": text})

            logger.info(f"Loaded {len(comments)} comments")
            return comments

        except Exception as e:
            logger.error(f"Error loading comments: {e}")
            return None

    def process_document(self, file_path: str) -> Optional[Dict[str, Any]]:
        logger.info(f"Processing document: {file_path}")

        text = self.load_document_text(file_path)
        if not text:
            return None

        chunks = self.text_splitter.split_text(text)
        if not chunks:
            logger.warning("No chunks generated")
            return {"original_text": text, "chunks": []}

        try:
            embeddings = self.embedding_model.encode(
                chunks, prompt=self.doc_prefix, show_progress_bar=False
            )

            processed_chunks = []
            for i, (chunk_text, emb) in enumerate(zip(chunks, embeddings)):
                processed_chunks.append(
                    {"text": chunk_text, "embedding": emb.tolist(), "chunk_index": i}
                )

            logger.info(f"Document processed: {len(chunks)} chunks")
            return {"original_text": text, "chunks": processed_chunks}

        except Exception as e:
            logger.error(f"Error embedding document chunks: {e}")
            return None

    def process_comments(self, file_path: str) -> Optional[List[Dict[str, Any]]]:
        logger.info(f"Processing comments: {file_path}")

        comments = self.load_comments(file_path)
        if not comments:
            return [] if comments == [] else None

        texts = [c["comment_text"] for c in comments]

        try:
            embeddings = self.embedding_model.encode(
                texts, prompt=self.comment_prefix, show_progress_bar=False
            )

            for comment, embedding in zip(comments, embeddings):
                comment["embedding"] = embedding.tolist()

            logger.info(f"Comments processed: {len(comments)}")
            return comments

        except Exception as e:
            logger.error(f"Error embedding comments: {e}")
            return None
