"""Fit and evaluate a fixed, feature-only DCN on frozen FAISS candidates."""
import argparse,gzip,hashlib,json,os,time
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import numpy as np
import tensorflow as tf
from scripts.dcn_ranker import CrossLayer


def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''): h.update(block)
 return h.hexdigest()


def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--candidates',required=True,type=Path);p.add_argument('--embeddings',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
 a=p.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 summary=json.loads((a.candidates/'summary.json').read_text(encoding='utf-8'));cfile=a.candidates/'candidates.jsonl.gz'
 if sha(cfile)!=summary['candidate_file_sha256'] or sha(a.embeddings)!=summary['embeddings_sha256']:raise ValueError('Input artifact hash mismatch')
 with np.load(a.embeddings,allow_pickle=False) as z: popularity=z['popularity'].astype(np.int64); indexed=z['item_ids'].astype(np.int64)
 if len(np.unique(popularity))!=len(popularity):raise ValueError('Duplicate popularity ID')
 ranks={int(i):float(j)/len(popularity) for j,i in enumerate(popularity)};indexset=set(map(int,indexed))
 early=[];late=[]
 with gzip.open(cfile,'rt',encoding='utf-8') as f:
  for line in f:
   row=json.loads(line);(early if row['phase']=='early' else late if row['phase']=='late' else (_ for _ in ()).throw(ValueError('Invalid phase'))).append(row)
 def features(row):
  ids=list(map(int,row['candidate_item_ids']));scores=np.asarray(row['retrieval_scores'],np.float32)
  if len(ids)!=len(scores) or len(ids)!=len(set(ids)) or not np.isfinite(scores).all():raise ValueError('Invalid candidates')
  return ids,np.column_stack((scores,np.asarray([ranks[i] for i in ids],np.float32))).astype(np.float32)
 train_x=[];train_y=[]
 for row in early:
  ids,x=features(row);pos={item:i for i,item in enumerate(ids)}
  for item,value in row['observed_ratings'].items():
   if int(item) not in pos:raise ValueError('Training label outside candidates')
   train_x.append(x[pos[int(item)]]);train_y.append(float(value))
 x=np.asarray(train_x,np.float32);y=np.asarray(train_y,np.float32)
 if len(y)!=summary['counts']['early_labels'] or not len(y):raise ValueError('Training support mismatch')
 mean=x.mean(axis=0);std=x.std(axis=0);std[std<1e-6]=1.0;x=(x-mean)/std
 tf.keras.utils.set_random_seed(42)
 try:tf.config.threading.set_intra_op_parallelism_threads(4);tf.config.threading.set_inter_op_parallelism_threads(2)
 except RuntimeError:pass
 inp=tf.keras.Input(shape=(2,),name='features');cross=inp
 for i in range(2):cross=CrossLayer(name=f'cross_{i}')([inp,cross])
 deep=tf.keras.layers.Dense(16,activation='relu')(inp);deep=tf.keras.layers.Dense(8,activation='relu')(deep)
 out=tf.keras.layers.Dense(1)(tf.keras.layers.Concatenate()([cross,deep]));model=tf.keras.Model(inp,out)
 model.compile(optimizer=tf.keras.optimizers.Adam(0.001),loss='mse');start=time.monotonic()
 hist=model.fit(x,y,batch_size=256,epochs=4,verbose=0,shuffle=True)
 per=[];candidate_hits=all_relevant=all_indexed=0;zero=0
 for row in late:
  ids,raw=features(row);relevant=set(map(int,row['relevant_item_ids']))
  if not relevant:zero+=1;continue
  pred=np.asarray(model((raw-mean)/std,training=False)).reshape(-1)
  if not np.isfinite(pred).all():raise ValueError('Non-finite ranker scores')
  methods={'faiss':ids,'dcn':[ids[i] for i in np.lexsort((np.asarray(ids),-pred))],
           'popularity':[ids[i] for i in np.lexsort((np.asarray(ids),raw[:,1]))]}
  metrics={}
  for name,ordered in methods.items():
   hits=[i for i,item in enumerate(ordered[:10]) if item in relevant]
   ideal=sum(1/np.log2(i+2) for i in range(min(10,len(relevant))))
   metrics[name]={'recall10':len(hits)/len(relevant),'ndcg10':sum(1/np.log2(i+2) for i in hits)/ideal}
  all_relevant+=len(relevant);all_indexed+=len(relevant&indexset);candidate_hits+=len(relevant&set(ids));per.append(metrics)
 if not per:raise ValueError('No eligible evaluation users')
 vals={name:np.asarray([[r[name]['recall10'],r[name]['ndcg10']] for r in per]) for name in ('faiss','dcn','popularity')}
 rng=np.random.default_rng(42);draws=rng.integers(0,len(per),size=(2000,len(per)),dtype=np.int32)
 results={name:{'recall@10':float(v[:,0].mean()),'ndcg@10':float(v[:,1].mean())} for name,v in vals.items()}
 boot={}
 for base in ('faiss','popularity'):
  for j,name in enumerate(('recall@10','ndcg@10')):
   diff=(vals['dcn'][:,j]-vals[base][:,j])[draws].mean(axis=1)
   boot['dcn_minus_'+base+'_'+name]={'median':float(np.median(diff)),'ci95':[float(np.quantile(diff,.025)),float(np.quantile(diff,.975))]}
 a.output.mkdir(parents=True,exist_ok=False);model.save(a.output/'ranker.keras')
 np.savez(a.output/'normalization.npz',mean=mean,std=std)
 report={'protocol':'docs/faiss-aligned-ranker-protocol.md','candidate_sha256':sha(cfile),'embeddings_sha256':sha(a.embeddings),
  'training_observed_labels':len(y),'epochs':4,'loss_by_epoch':[float(v) for v in hist.history['loss']],
  'late_users_total':len(late),'late_users_without_positive':zero,'metric_users':len(per),'late_positive_targets':all_relevant,
  'positive_target_index_coverage':all_indexed/all_relevant,'positive_target_candidate_recall@100_micro':candidate_hits/all_relevant,
  'metrics':results,'paired_bootstrap_2000':boot,'elapsed_seconds':time.monotonic()-start,
  'limitations':['Reused development interval; no independent validation.','Observed rating labels are exposure biased, not clicks.','Feature-only DCN; no learned user/item embeddings.','No final-period outcomes used.']}
 (a.output/'metrics.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
 print(json.dumps(report,indent=2))

if __name__=='__main__':main()
