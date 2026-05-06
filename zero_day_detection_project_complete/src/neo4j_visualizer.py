from neo4j import GraphDatabase
import logging
from typing import Optional, Dict, Any
import time
from contextlib import contextmanager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Neo4jHandler:
    def __init__(self, uri: str = "bolt://localhost:7687",
                 user: str = "neo4j",
                 password: str = "password",
                 max_retries: int = 3,
                 retry_delay: int = 5):
        self.uri = uri
        self.user = user
        self.password = password
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.driver = None
        self._connect()

    def _connect(self) -> None:
        """Establish connection to Neo4j database with retry logic."""
        for attempt in range(self.max_retries):
            try:
                self.driver = GraphDatabase.driver(
                    self.uri,
                    auth=(self.user, self.password)
                )
                # Test connection
                with self.driver.session() as session:
                    session.run("RETURN 1")
                logger.info("Successfully connected to Neo4j database")
                return
            except Exception as e:
                if attempt < self.max_retries - 1:
                    logger.warning(f"Connection attempt {attempt + 1} failed: {str(e)}")
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"Failed to connect to Neo4j after {self.max_retries} attempts")
                    raise

    @contextmanager
    def get_session(self):
        """Context manager for Neo4j sessions."""
        if not self.driver:
            self._connect()
        
        session = None
        try:
            session = self.driver.session()
            yield session
        except Exception as e:
            logger.error(f"Session error: {str(e)}")
            raise
        finally:
            if session:
                session.close()

    def push_anomaly(self, src_ip: str, dst_ip: str, 
                     additional_data: Optional[Dict[str, Any]] = None) -> None:
        """
        Push anomaly data to Neo4j database.
        
        Args:
            src_ip: Source IP address
            dst_ip: Destination IP address
            additional_data: Optional dictionary of additional data to store
        """
        try:
            with self.get_session() as session:
                # Create or update IP nodes and relationship
                query = """
                MERGE (s:IP {addr: $src})
                MERGE (d:IP {addr: $dst})
                MERGE (s)-[r:ATTACKED]->(d)
                SET r.timestamp = datetime()
                """
                
                if additional_data:
                    # Add additional properties to the relationship
                    props = ", ".join([f"r.{k} = ${k}" for k in additional_data.keys()])
                    query += f" SET {props}"
                
                params = {"src": src_ip, "dst": dst_ip}
                if additional_data:
                    params.update(additional_data)
                
                session.run(query, params)
                logger.info(f"Successfully pushed anomaly data for {src_ip} -> {dst_ip}")
        
        except Exception as e:
            logger.error(f"Error pushing anomaly data: {str(e)}")
            raise

    def close(self) -> None:
        """Close the Neo4j driver connection."""
        if self.driver:
            try:
                self.driver.close()
                logger.info("Neo4j connection closed")
            except Exception as e:
                logger.error(f"Error closing Neo4j connection: {str(e)}")
                raise

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
