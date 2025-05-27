import json
import logging
from typing import Dict, Optional
from ..config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CommentAnalyzer:
    def __init__(self, vector_store, llm_client, top_k=10):
        self.vector_store = vector_store
        self.llm_client = llm_client
        self.top_k = top_k

    def analyze_comment(self, comment: Dict, doc_base_name: str) -> Dict:
        v1_collection = f"{settings.QDRANT_COLLECTION_V1_PREFIX}{doc_base_name}"
        v2_collection = f"{settings.QDRANT_COLLECTION_V2_PREFIX}{doc_base_name}"

        logger.info(
            f"Analyzing comment {comment['comment_id']}: '{comment['comment_text'][:50]}...'"
        )

        v1_results = self.vector_store.search(
            v1_collection, comment["embedding"], self.top_k
        )
        v2_results = self.vector_store.search(
            v2_collection, comment["embedding"], self.top_k
        )

        v1_text = "\n\n".join(
            [f"Chunk {i+1}:\n{hit.payload['text']}" for i, hit in enumerate(v1_results)]
        )
        v2_text = "\n\n".join(
            [f"Chunk {i+1}:\n{hit.payload['text']}" for i, hit in enumerate(v2_results)]
        )

        prompt = f"""
You are an expert document analyst. Determine if a comment made on an earlier document version was addressed in a newer version.

Comment:
{comment['comment_text']}

Relevant text from VERSION 1 of the document:
{v1_text}

Relevant text from VERSION 2 of the document:
{v2_text}

Based ONLY on the provided texts, determine if the comment was addressed in version 2.
Your response MUST be ONLY a single, valid JSON object. Do NOT include ```json markers, introductory sentences, or any text outside the JSON object itself.
Strictly adhere to JSON syntax: use double quotes ("") for all keys and all string values. Do not use single quotes for keys.

The JSON object MUST contain these fields, with string values in Russian:
- "explanation": Detailed explanation of whether and how the comment was addressed.
- "evidence_v1": Relevant evidence text from version 1.
- "evidence_v2": Relevant evidence text from version 2 showing if/how the comment was addressed.
- "suggestion": Suggestion for further improvements if needed (empty string if none).
- "status": Exactly one of: "учтен", "не учтен", "частично учтен".

Example of the required EXACT output format:
{{
  "explanation": "Комментарий был полностью учтен путем добавления нового раздела 3.2.",
  "evidence_v1": "Раздел 3 отсутствует.",
  "evidence_v2": "Раздел 3.2: Описание новой функции.",
  "suggestion": "",
  "status": "учтен"
}}

Now, generate the JSON object for the provided comment and texts. Respond ONLY with the JSON object.
"""

        llm_response = self.llm_client.get_completion(prompt)
        if not llm_response:
            logger.error(
                f"LLM returned empty response for comment {comment['comment_id']}"
            )
            return self._create_error_result(comment, None)

        try:
            json_text = llm_response.strip()
            first_brace = json_text.find("{")
            last_brace = json_text.rfind("}")

            if first_brace != -1 and last_brace != -1:
                json_text = json_text[first_brace : last_brace + 1]

            result = json.loads(json_text)

            return {
                "comment_id": comment["comment_id"],
                "comment_text": comment["comment_text"],
                "explanation": result.get("explanation", "No explanation provided"),
                "evidence_v1": result.get("evidence_v1", "No evidence provided"),
                "evidence_v2": result.get("evidence_v2", "No evidence provided"),
                "suggestion": result.get("suggestion", ""),
                "status": result.get("status", "не учтен"),
            }

        except Exception as e:
            logger.error(
                f"Error processing LLM response for comment {comment['comment_id']}: {e}"
            )
            logger.error(f"Original response: '{llm_response}'")
            return self._create_error_result(comment, e, json_text)

    def _create_error_result(
        self, comment: Dict, error: Optional[Exception], json_text: str = ""
    ) -> Dict:
        error_message = (
            f"Error analyzing comment: {str(error)}"
            if error
            else "Error analyzing comment (Unknown)"
        )
        return {
            "comment_id": comment["comment_id"],
            "comment_text": comment["comment_text"],
            "explanation": error_message,
            "evidence_v1": json_text,
            "evidence_v2": "",
            "suggestion": "",
            "status": "error",
        }
