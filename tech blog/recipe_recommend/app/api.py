from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from .recommender import RecipeRecommender
import pandas as pd
import psycopg2
import uvicorn

app = FastAPI()

def load_data():
    conn = psycopg2.connect(
        dbname="mydb",
        user="postgres",
        password="pass",
        # host="my_postgres", # To build docker image, it would be changed
        host = "18.144.40.123",
        port="5432"
    )

    df = pd.read_sql(
        """SELECT * 
        FROM recommendations""", conn)
    conn.close()
    print("✅ Recipe Recommender Data Loaded")
    return df

df = load_data()

def request_API(**kwargs):
    agg = df.groupby(['recipe_name'])[['agg_rating']].mean().sort_values(['agg_rating'], ascending=False)[0:15]
    popular_recipe = {'first': list(agg.index)}

    user_name = kwargs['user_name']
    search_keyword = kwargs['keyword']
    history_recipe = kwargs['history_recipe']

    df['str_collection'] = df['keyword_collection'].apply(lambda x: ', '.join(x) if isinstance(x, list) else '')

    if (user_name not in list(df.user_name)) & (search_keyword == ''):
        recommend_result = popular_recipe
    if (user_name not in list(df.user_name)) & (search_keyword != ''):
        tmp = df[df['str_collection'].str.contains(search_keyword, case=False, regex=True)]
        if len(tmp) == 0:
            recommend_result = popular_recipe
        else:
            agg = tmp.groupby(['recipe_name'])[['agg_rating']].mean(
            ).sort_values(['agg_rating'], ascending=False)[0:15]
            recommend_result = {"first": list(agg.index)}
    if user_name in list(df.user_name):
        recommend_result = df.loc[(df.user_name == user_name) & (df.recipe_name == history_recipe), ][['recommend_result']].values[0][0]
    return recommend_result


class RecommendationInput(BaseModel):
    user_name: str
    keyword: str
    history_recipe: str


@app.post("/recommend")
def recommend_recipe(input_data: RecommendationInput):
    try:
        result = request_API(
            **input_data.dict())  # ✅ dict() 변환 후 전달
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# FastAPI 실행 (직접 실행할 경우만)
if __name__ == "__main__":
    uvicorn.run(app, host="18.144.40.123", port=7000)
