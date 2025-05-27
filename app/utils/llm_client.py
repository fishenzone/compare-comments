import logging
from openai import OpenAI
import time
from ..config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(self):
        self.vllm_base_url = f"http://{settings.VLLM_HOST}:{settings.VLLM_PORT}/v1"
        logger.info(f"Initializing LLM client with URL: {self.vllm_base_url}")
        self.client = OpenAI(base_url=self.vllm_base_url, api_key=settings.VLLM_API_KEY)

    def get_completion(self, prompt, max_retries=3):
        for attempt in range(max_retries + 1):
            try:
                logger.info(
                    f"Sending prompt to LLM (attempt {attempt + 1}/{max_retries + 1})"
                )
                response = self.client.chat.completions.create(
                    model=settings.VLLM_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=2000,
                )

                if response.choices and response.choices[0].message:
                    return response.choices[0].message.content
                else:
                    logger.warning("Empty response from LLM")

            except Exception as e:
                logger.error(f"LLM request failed: {e}")

            if attempt < max_retries:
                sleep_time = 2**attempt
                logger.info(f"Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)

        logger.error("All LLM attempts failed")
        return None
