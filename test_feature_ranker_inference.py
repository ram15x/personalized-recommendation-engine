import numpy as np
import pytest
from fastapi import HTTPException
from scripts.feature_ranker_inference import FeatureRanker
from scripts.recommendation_api import TwoTowerRequest,create_app


def test_feature_ranker_uses_score_and_frozen_popularity():
    def model(x,training=False):return -x[:,1:2]
    ranker=FeatureRanker(model,[0,0],[1,1],[20,10,30])
    assert ranker.rank([10,20],[0.8,0.3])==[20,10]
    with pytest.raises(ValueError,match='absent'):ranker.rank([99],[0.4])
    with pytest.raises(ValueError,match='aligned'):ranker.rank([10,10],[0.2,0.3])


def test_api_reranks_candidates_and_keeps_original_retrieval_scores(tmp_path):
    pop=tmp_path/'pop.csv';movies=tmp_path/'movies.csv'
    pop.write_text('movieId,positive_count\n10,5\n20,3\n',encoding='utf-8')
    movies.write_text('movieId,title,genres\n10,A,Drama\n20,B,Comedy\n',encoding='utf-8')
    class Retriever:
        def retrieve(self,user_id,seen,limit):return {'item_ids':[10,20],'scores':[0.8,0.3],'reason':None}
    ranker=FeatureRanker(lambda x,training=False:-x[:,1:2],[0,0],[1,1],[20,10])
    app=create_app(pop,movies,experimental_retriever=Retriever(),experimental_feature_ranker=ranker)
    route=next(r for r in app.routes if getattr(r,'path',None)=='/experimental/two-tower')
    response=route.endpoint(TwoTowerRequest(user_id=1,limit=2,experimental_feature_dcn=True))
    assert response=={'strategy':'experimental_two_tower_feature_dcn','reason':None,'item_ids':[20,10],'scores':[0.3,0.8]}
    display=next(r for r in app.routes if getattr(r,'path',None)=='/experimental/two-tower/display')
    shown=display.endpoint(TwoTowerRequest(user_id=1,limit=2,experimental_feature_dcn=True))
    assert shown['movies']==[{'movie_id':20,'title':'B'},{'movie_id':10,'title':'A'}]
    with pytest.raises(HTTPException):route.endpoint(TwoTowerRequest(user_id=1,experimental_dcn=True,experimental_feature_dcn=True))
