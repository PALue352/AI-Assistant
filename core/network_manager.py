# network_manager.py (v1.0.2)
import logging
import aiohttp
import asyncio

logger = logging.getLogger(__name__)

class NetworkManager:
    def __init__(self):
        logger.info("NetworkManager initializing...")
        self.session = None  # Initialize session to None
        logger.info("NetworkManager initialized.")

    async def initialize(self): #Added for consistency
        pass

    async def get(self, url, max_retries=3, timeout=10):
        """Make an asynchronous GET request with retries and timeout.

        Args:
            url (str): URL to request.
            max_retries (int): Maximum number of retries.
            timeout (int): Timeout in seconds.

        Returns:
            str: Response text, or None if the request failed.
        """
        if self.session is None:
          self.session = aiohttp.ClientSession()
        for attempt in range(max_retries):
            try:
                async with self.session.get(url, timeout=timeout) as response:
                    response.raise_for_status()  # Raise an exception for bad status codes
                    return await response.text()
            except aiohttp.ClientError as e:
                logger.warning(f"Attempt {attempt + 1} failed for {url}: {e}")
                if attempt == max_retries - 1:
                    logger.error(f"Max retries reached for {url}. Giving up.")
                    return None
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
            except asyncio.TimeoutError:
                logger.warning(f"Timeout (attempt {attempt+1}) for {url}")
                if attempt == max_retries - 1:
                  logger.error(f"Max retries reached for {url}. Giving up.")
                  return None
                await asyncio.sleep(2 ** attempt)  # Exponential backoff

    async def close(self):
        """Close the aiohttp session."""
        if self.session:
            await self.session.close()
            logger.info("NetworkManager session closed.")