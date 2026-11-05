"""Measure observed-label support in time-safe FAISS candidate lists."""
from collections import defaultdict
import argparse, hashlib, json
from pathlib import Path
import faiss
import numpy as np
import pandas as pd
from scripts.audit_faiss_ranker_feasibility import TRAIN_END, MIDPOINT, DEV_END, EXPECTED_RATINGS_SHA256, sha256
from scripts.two_tower_inference import TwoTowerRetriever


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ratings',required=True,type=Path)
    p.add_argument('--retriever',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    args=p.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    actual=sha256(args.ratings)
    if actual != EXPECTED_RATINGS_SHA256: raise ValueError(f'Unexpected ratings SHA-256: {actual}')
    faiss.omp_set_num_threads(2)
    retriever=TwoTowerRetriever.load(args.retriever)
    chosen=np.sort(np.random.default_rng(42).choice(retriever.user_ids,10000,replace=False))
    active=set()
    for chunk in pd.read_csv(args.ratings,usecols=['userId','timestamp'],chunksize=500000):
        relevant=chunk.loc[(chunk.timestamp>=TRAIN_END)&(chunk.timestamp<DEV_END)]
        active.update(map(int,relevant.loc[relevant.userId.isin(chosen),'userId'].unique()))
    prior=defaultdict(set); early=defaultdict(dict); late=defaultdict(dict)
    for chunk in pd.read_csv(args.ratings,usecols=['userId','movieId','rating','timestamp'],chunksize=500000):
        part=chunk.loc[(chunk.timestamp<DEV_END)&chunk.userId.isin(active)]
        for row in part.itertuples(index=False):
            user,item,score,ts=int(row.userId),int(row.movieId),float(row.rating),int(row.timestamp)
            if ts<TRAIN_END:
                if item in prior[user]: raise ValueError('Duplicate user/item in prior window')
                prior[user].add(item)
            elif ts<MIDPOINT:
                if item in early[user] or item in prior[user]: raise ValueError('Duplicate user/item before midpoint')
                early[user][item]=score
            else:
                if item in late[user]: raise ValueError('Duplicate user/item in later window')
                late[user][item]=score
    counts=[]
    for user in sorted(active):
        early_candidates=set(retriever.retrieve(user,prior[user],100)['item_ids'])
        later_history=prior[user]|set(early[user])
        late_candidates=set(retriever.retrieve(user,later_history,100)['item_ids'])
        early_labels=early_candidates & early[user].keys()
        late_labels=late_candidates & late[user].keys()
        counts.append((len(early_labels),sum(early[user][i]>=4 for i in early_labels),
                       len(late_labels),sum(late[user][i]>=4 for i in late_labels)))
    a=np.asarray(counts,dtype=np.int64)
    if a.shape != (len(active),4): raise ValueError('No active sampled users')
    result={'protocol':'docs/faiss-ranker-feasibility-protocol.md',
        'ratings_sha256':actual,'embeddings_sha256':sha256(args.retriever/'embeddings.npz'),
        'index_sha256':sha256(args.retriever/'items.faiss'), 'sample_size':10000,'seed':42,
        'active_users_either_half':len(active),'candidate_limit':100,
        'early_observed_candidate_labels':int(a[:,0].sum()),'early_positive_candidate_labels':int(a[:,1].sum()),
        'late_observed_candidate_labels':int(a[:,2].sum()),'late_positive_candidate_labels':int(a[:,3].sum()),
        'early_users_with_candidate_labels':int((a[:,0]>0).sum()),
        'late_users_with_candidate_labels':int((a[:,2]>0).sum()),
        'users_with_candidate_labels_both_halves':int(((a[:,0]>0)&(a[:,2]>0)).sum()),
        'users_with_positive_candidate_labels_both_halves':int(((a[:,1]>0)&(a[:,3]>0)).sum()),
        'limitations':['Reused development window, not independent validation.',
                       'Observed ratings on retrieved items only; unknown items are not negatives.',
                       'No model training or recommendation quality scoring.']}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as f: json.dump(result,f,indent=2); f.write('\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
