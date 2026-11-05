"""Experimental feature-only reranker for the saved FAISS retriever."""
import hashlib
import json
from pathlib import Path
import numpy as np


def _sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


class FeatureRanker:
    def __init__(self, model, mean, std, popularity):
        self.model=model
        self.mean=np.asarray(mean,dtype=np.float32)
        self.std=np.asarray(std,dtype=np.float32)
        ids=np.asarray(popularity,dtype=np.int64)
        if (self.mean.shape!=(2,) or self.std.shape!=(2,) or not np.isfinite(self.mean).all()
                or not np.isfinite(self.std).all() or np.any(self.std<=0)):
            raise ValueError('Invalid feature normalization')
        if ids.ndim!=1 or not len(ids) or np.any(ids<=0) or len(np.unique(ids))!=len(ids):
            raise ValueError('Invalid popularity vocabulary')
        self.ranks={int(item):float(i)/len(ids) for i,item in enumerate(ids)}

    @classmethod
    def load(cls,directory,embeddings):
        directory=Path(directory);embeddings=Path(embeddings)
        manifest=json.loads((directory/'artifact_manifest.json').read_text(encoding='utf-8'))
        for name in ('ranker.keras','normalization.npz','metrics.json'):
            if manifest['files_sha256'].get(name)!=_sha256(directory/name):
                raise ValueError(f'Feature-ranker artifact hash mismatch: {name}')
        if manifest['embeddings_sha256']!=_sha256(embeddings):
            raise ValueError('Feature ranker does not match retriever embeddings')
        metrics=json.loads((directory/'metrics.json').read_text(encoding='utf-8'))
        if metrics['embeddings_sha256']!=manifest['embeddings_sha256']:
            raise ValueError('Feature ranker training metadata mismatch')
        with np.load(directory/'normalization.npz',allow_pickle=False) as z:mean,std=z['mean'],z['std']
        with np.load(embeddings,allow_pickle=False) as z:popularity=z['popularity']
        import tensorflow as tf
        import scripts.dcn_ranker
        model=tf.keras.models.load_model(directory/'ranker.keras',compile=False)
        return cls(model,mean,std,popularity)

    def rank(self,item_ids,retrieval_scores):
        ids=np.asarray(item_ids,dtype=np.int64);scores=np.asarray(retrieval_scores,dtype=np.float32)
        if (ids.ndim!=1 or scores.shape!=ids.shape or np.any(ids<=0)
                or len(np.unique(ids))!=len(ids) or not np.isfinite(scores).all()):
            raise ValueError('Candidates and scores must be aligned, finite and unique')
        if any(int(i) not in self.ranks for i in ids):
            raise ValueError('A candidate is absent from the frozen popularity catalog')
        if not len(ids):return []
        features=np.column_stack((scores,[self.ranks[int(i)] for i in ids])).astype(np.float32)
        predicted=np.asarray(self.model((features-self.mean)/self.std,training=False)).reshape(-1)
        if len(predicted)!=len(ids) or not np.isfinite(predicted).all():raise ValueError('Invalid ranker predictions')
        order=np.lexsort((ids,-predicted))
        return ids[order].tolist()
