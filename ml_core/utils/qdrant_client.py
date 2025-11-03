"""
Qdrant Vector Database Client
Stores interaction embeddings for similarity-based preference matching

Uses cosine similarity to find similar past decisions
"""

import os
import numpy as np
import hashlib
import json
from typing import Dict, List, Optional
from datetime import datetime

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance,
        VectorParams,
        PointStruct,
        Filter,
        FieldCondition,
        MatchValue
    )
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    print("⚠ qdrant-client not installed. Run: pip install qdrant-client")


class InteractionVectorStore:
    """
    Qdrant client for storing and searching interaction vectors

    Features:
    - Store interaction embeddings (64-dim vectors)
    - Find similar past interactions
    - Filter by user_id
    - Query interaction counts
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        collection_name: str = "user_interactions",
        vector_size: int = 64
    ):
        """
        Initialize Qdrant client

        Args:
            host: Qdrant server host (default: localhost)
            port: Qdrant server port (default: 6333)
            collection_name: Name of collection to use
            vector_size: Dimension of embedding vectors (default: 64)
        """
        if not QDRANT_AVAILABLE:
            raise ImportError(
                "qdrant-client not installed. Install with: pip install qdrant-client"
            )

        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.vector_size = vector_size

        # Initialize client
        self.client = QdrantClient(host=host, port=port)

        # Create collection if doesn't exist
        self._ensure_collection_exists()

        print(f"✓ Qdrant client initialized: {host}:{port}/{collection_name}")

    def _ensure_collection_exists(self):
        """Create collection if it doesn't exist"""
        try:
            collections = self.client.get_collections().collections
            exists = any(c.name == self.collection_name for c in collections)

            if not exists:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE
                    )
                )
                print(f"✓ Created Qdrant collection: {self.collection_name}")
        except Exception as e:
            print(f"⚠ Qdrant collection creation failed: {e}")
            raise

    def embed_interaction(self, interaction: Dict) -> np.ndarray:
        """
        Create vector embedding from interaction

        Extracts features from the selected recommendation and creates
        a normalized 64-dimensional vector.

        Args:
            interaction: User interaction dict containing:
                - selected_recommendation_index: int
                - recommendations: List[Dict]

        Returns:
            64-dimensional numpy array (normalized)
        """
        # Extract selected recommendation
        selected_idx = interaction.get('selected_recommendation_index', 0)
        recs = interaction.get('recommendations', [])

        if selected_idx >= len(recs) or len(recs) == 0:
            # Default embedding (zeros)
            return np.zeros(self.vector_size)

        selected = recs[selected_idx]
        summary = selected.get('summary', {})

        # Feature extraction (normalize to 0-1 range)
        features = [
            # Cost features (normalized, cap at 100k)
            min(summary.get('total_cost', 0) / 100000, 1.0),

            # Coverage features
            min(summary.get('hospitals_helped', 0) / 20, 1.0),  # Cap at 20 hospitals
            min(summary.get('total_transfers', 0) / 50, 1.0),  # Cap at 50 transfers
            min(summary.get('total_quantity_transferred', 0) / 1000, 1.0),  # Cap at 1000 units

            # Performance features
            summary.get('shortage_reduction_percent', 0) / 100,

            # Strategy type (one-hot encoding)
            1.0 if 'cost' in selected.get('strategy_name', '').lower() else 0.0,
            1.0 if 'coverage' in selected.get('strategy_name', '').lower() else 0.0,
            1.0 if 'balanced' in selected.get('strategy_name', '').lower() else 0.0,
            1.0 if 'urgent' in selected.get('strategy_name', '').lower() else 0.0,

            # Efficiency metrics
            # Shortage reduction per dollar (scaled)
            (summary.get('shortage_reduction_percent', 0) / max(summary.get('total_cost', 1), 1)) * 1000,

            # Coverage per transfer
            summary.get('hospitals_helped', 0) / max(summary.get('total_transfers', 1), 1),

            # Score features
            selected.get('cost_score', 0.5),
            selected.get('speed_score', 0.5),
            selected.get('coverage_score', 0.5),
            selected.get('overall_score', 0.5),
        ]

        # Pad to vector_size
        embedding = np.zeros(self.vector_size)
        embedding[:min(len(features), self.vector_size)] = features[:self.vector_size]

        # Add small random noise to prevent exact duplicates
        embedding += np.random.normal(0, 0.005, self.vector_size)

        # Normalize to unit length
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    def store_interaction(
        self,
        user_id: str,
        interaction: Dict,
        interaction_id: Optional[str] = None
    ) -> str:
        """
        Store interaction in Qdrant

        Args:
            user_id: User identifier
            interaction: Interaction data
            interaction_id: Optional custom ID (auto-generated if not provided)

        Returns:
            ID of stored point
        """
        # Generate embedding
        embedding = self.embed_interaction(interaction)

        # Generate ID if not provided
        if interaction_id is None:
            # Hash of user_id + timestamp
            timestamp = interaction.get('timestamp', datetime.now().isoformat())
            hash_input = f"{user_id}_{timestamp}".encode()
            interaction_id = hashlib.md5(hash_input).hexdigest()

        # Extract selected recommendation for payload
        selected_idx = interaction.get('selected_recommendation_index', 0)
        recs = interaction.get('recommendations', [])

        if selected_idx < len(recs):
            selected = recs[selected_idx]
            summary = selected.get('summary', {})

            payload = {
                'user_id': user_id,
                'timestamp': interaction.get('timestamp', datetime.now().isoformat()),
                'selected_index': selected_idx,
                'strategy_name': selected.get('strategy_name', ''),
                'total_cost': float(summary.get('total_cost', 0)),
                'hospitals_helped': int(summary.get('hospitals_helped', 0)),
                'total_transfers': int(summary.get('total_transfers', 0)),
                'shortage_reduction': float(summary.get('shortage_reduction_percent', 0))
            }
        else:
            # Minimal payload if no valid selection
            payload = {
                'user_id': user_id,
                'timestamp': interaction.get('timestamp', datetime.now().isoformat()),
                'selected_index': selected_idx,
                'strategy_name': 'unknown'
            }

        # Upload to Qdrant
        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    PointStruct(
                        id=interaction_id,
                        vector=embedding.tolist(),
                        payload=payload
                    )
                ]
            )
            return interaction_id
        except Exception as e:
            print(f"⚠ Qdrant storage failed: {e}")
            raise

    def find_similar_interactions(
        self,
        interaction: Dict,
        user_id: Optional[str] = None,
        limit: int = 5
    ) -> List[Dict]:
        """
        Find similar past interactions using cosine similarity

        Args:
            interaction: Current interaction to match
            user_id: Optional filter by user (None = search all users)
            limit: Max results to return

        Returns:
            List of similar interactions with scores:
            [
                {
                    'id': str,
                    'score': float (0-1, higher = more similar),
                    'payload': Dict (metadata)
                },
                ...
            ]
        """
        # Generate embedding for query
        query_vector = self.embed_interaction(interaction)

        # Build filter
        query_filter = None
        if user_id:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="user_id",
                        match=MatchValue(value=user_id)
                    )
                ]
            )

        # Search
        try:
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector.tolist(),
                limit=limit,
                query_filter=query_filter
            )

            # Format results
            similar = []
            for result in results:
                similar.append({
                    'id': result.id,
                    'score': result.score,
                    'payload': result.payload
                })

            return similar

        except Exception as e:
            print(f"⚠ Qdrant search failed: {e}")
            return []

    def get_user_interaction_count(self, user_id: str) -> int:
        """
        Get total interactions stored for user

        Args:
            user_id: User identifier

        Returns:
            Count of interactions
        """
        try:
            result = self.client.count(
                collection_name=self.collection_name,
                count_filter=Filter(
                    must=[
                        FieldCondition(
                            key="user_id",
                            match=MatchValue(value=user_id)
                        )
                    ]
                )
            )
            return result.count
        except Exception as e:
            print(f"⚠ Qdrant count failed: {e}")
            return 0

    def delete_user_interactions(self, user_id: str):
        """
        Delete all interactions for a user

        Args:
            user_id: User identifier
        """
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="user_id",
                            match=MatchValue(value=user_id)
                        )
                    ]
                )
            )
            print(f"✓ Deleted all interactions for user: {user_id}")
        except Exception as e:
            print(f"⚠ Qdrant delete failed: {e}")
            raise

    def get_collection_stats(self) -> Dict:
        """
        Get statistics about the collection

        Returns:
            Dict with collection stats
        """
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                'total_points': info.points_count,
                'vector_size': info.config.params.vectors.size,
                'distance': info.config.params.vectors.distance.name,
                'status': info.status.name
            }
        except Exception as e:
            print(f"⚠ Failed to get collection stats: {e}")
            return {}
