"""Export bounded, chronological FAISS candidates and observed rating labels."""
from collections import defaultdict
import argparse, gzip, json
from pathlib import Path
import faiss
import numpy as np
import pandas as pd
from scripts.audit_faiss_ranker_feasibility import TRAIN_END, MIDPOINT, DEV_END, EXPECTED_RATINGS_SHA256, sha256
from scripts.two_tower_inference import TwoTowerRetriever


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ratings',type=Path,required=True)
    parser.add_argument('--retriever',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    if sha256(args.ratings)!=EXPECTED_RATINGS_SHA256: raise ValueError('Ratings checksum mismatch')
    faiss.omp_set_num_threads(2)
    retriever=TwoTowerRetriever.load(args.retriever)
    active=set()
    for chunk in pd.read_csv(args.ratings,usecols=['userId','timestamp'],chunksize=500000):
        part=chunk.loc[(chunk.timestamp>=TRAIN_END)&(chunk.timestamp<DEV_END)]
        active.update(map(int,part.loc[part.userId.isin(retriever.user_ids),'userId'].unique()))
    prior=defaultdict(set); early=defaultdict(dict); late=defaultdict(dict)
    for chunk in pd.read_csv(args.ratings,usecols=['userId','movieId','rating','timestamp'],chunksize=500000):
        part=chunk.loc[(chunk.timestamp<DEV_END)&chunk.userId.isin(active)]
        for row in part.itertuples(index=False):
            u,i,r,t=int(row.userId),int(row.movieId),float(row.rating),int(row.timestamp)
            if t<TRAIN_END:
                if i in prior[u]: raise ValueError('Duplicate prior rating')
                prior[u].add(i)
            elif t<MIDPOINT:
                if i in early[u]: raise ValueError('Duplicate early rating')
                early[u][i]=r
            else:
                if i in late[u]: raise ValueError('Duplicate later rating')
                late[u][i]=r
    args.output.mkdir(parents=True,exist_ok=False)
    counts={k:0 for k in ('early_users','early_labels','early_positives','late_users','late_labels','late_positives')}
    with gzip.open(args.output/'candidates.jsonl.gz','wt',encoding='utf-8',compresslevel=6) as out:
        for u in sorted(active):
            if set(early[u]) & prior[u] or set(late[u]) & (prior[u]|set(early[u])):
                raise ValueError('Repeated user/item across windows')
            for phase,ratings,history in (('early',early[u],prior[u]),('late',late[u],prior[u]|set(early[u]))):
                if not ratings: continue
                result=retriever.retrieve(u,history,100)
                if result['reason'] or len(result['item_ids'])!=len(result['scores']):
                    raise ValueError('Unexpected retriever result')
                candidate_ids=result['item_ids']; candidate_set=set(candidate_ids)
                observed={str(i):r for i,r in ratings.items() if i in candidate_set}
                relevant=sorted(i for i,r in ratings.items() if r>=4 and i not in history)
                if candidate_set & history: raise ValueError('Seen item in candidates')
                row={'phase':phase,'user_id':u,'candidate_item_ids':candidate_ids,
                     'retrieval_scores':result['scores'],'observed_ratings':observed,'relevant_item_ids':relevant}
                out.write(json.dumps(row,separators=(',',':'))+'\n')
                counts[phase+'_users']+=1;counts[phase+'_labels']+=len(observed)
                counts[phase+'_positives']+=len(relevant)
    artifact=args.output/'candidates.jsonl.gz'
    summary={'protocol':'docs/faiss-aligned-ranker-protocol.md','ratings_sha256':EXPECTED_RATINGS_SHA256,
             'embeddings_sha256':sha256(args.retriever/'embeddings.npz'),'index_sha256':sha256(args.retriever/'items.faiss'),
             'candidate_file_sha256':sha256(artifact),'candidate_file_bytes':artifact.stat().st_size,
             'retriever_users':len(retriever.user_ids),'active_users':len(active),'counts':counts,
             'train_end':TRAIN_END,'midpoint':MIDPOINT,'dev_end':DEV_END,
             'limitations':['Reused development period, not independent validation.','Unobserved candidate outcomes are not negatives.','The final period was not retained or used.']}
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
