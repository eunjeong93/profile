# create class
import pandas as pd
import psycopg2
import re
import numpy as np
from gensim.models import Word2Vec
from sqlalchemy import create_engine
import json
import argparse


class RecipeRecommender:
    def __init__(self, db_params):
        """Initialize the recommender system with database connection parameters"""
        self.db_params = db_params
        self.conn = self._connect_db()
        self.df = self._load_data()
        self.recipe_keywords = {}
        self.user_keywords = {}
        self.model_user = None
        self.model_recipe = None
        self.similarity_recipe = None
        self.similarity_user = None
        self.popular_recipe = None
    
    def _connect_db(self):
        """Connect to PostgreSQL database using SQLAlchemy"""
        db_url = f"postgresql+psycopg2://{self.db_params['user']}:{self.db_params['password']}@{self.db_params['host']}:{self.db_params['port']}/{self.db_params['dbname']}"
        return create_engine(db_url)


    def _load_data(self):
        """Load data using SQLAlchemy"""
        query = "SELECT * FROM new_review"
        return pd.read_sql(query, self.conn)

    # def _connect_db(self):
    #     """Connect to PostgreSQL database"""
    #     return psycopg2.connect(**self.db_params)

    # def _load_data(self):
    #     """Load data from PostgreSQL into Pandas DataFrame"""
    #     query = "SELECT * FROM new_review"
    #     return pd.read_sql(query, self.conn)

    def _most_popular_list(self):
        agg = self.df.groupby(['recipe_name'])[['agg_rating']].mean(
        ).sort_values(['agg_rating'], ascending=False)[0:15]
        self.popular_recipe = {'first': list(agg.index)}

    @staticmethod
    def text_process(col):
        """Preprocess text data"""
        col = col.lower()
        stoplists = ['and', 'but', 'for', 'nor', 'yet', 'the', 'via', 'per', 'also', 'thus', 'once', 'from',
                     'with', 'into', 'onto', 'will', 'over', 'near', 'like', 'upon', 'such', 'some', 'that',
                     '.', ',', 'our', 'was', 'were', 'did', "didn't", 'now', 'this', 'that']
        for s in stoplists:
            col = col.replace(s, '')
        splited_text = [k for k in col.split(' ') if len(k) > 2]
        return splited_text

    @staticmethod
    def rm_special(col):
        """Remove special characters from a list of words"""
        return [item for item in col if not re.search(r"[^a-zA-Z0-9]", item)]

    def preprocess_data(self):
        """Apply text processing and create keyword collections"""
        self.df['keyword_ls'] = self.df.text.apply(self.text_process)
        self.df['tags'] = self.df.feature.apply(
            lambda x: x.replace(' ', '').replace('#', '').split(','))
        self.df['keyword_ls'] = self.df['keyword_ls'].apply(self.rm_special)
        self.df['new_category'] = self.df.food_category.apply(
            lambda x: x.split('/'))
        self.df['keyword_collection'] = self.df.keyword_ls + \
            self.df.new_category + self.df.tags

        # Aggregate keywords per recipe and user
        # 별점이 낮은 사용자에 대해서 추천을 어떻게 처리할건지.
        self.recipe_keywords = self.df.groupby(
            'recipe_name')['keyword_collection'].sum().to_dict()
        self.user_keywords = self.df.groupby(
            'user_name')['keyword_collection'].sum().to_dict()

    def train_models(self, vector_size=200, window=5, min_count=5, workers=4, sg=1):
        """Train Word2Vec models for users and recipes"""
        self.model_user = Word2Vec(sentences=self.user_keywords.values(), vector_size=vector_size,
                                   window=window, min_count=min_count, workers=workers, sg=sg)

        self.model_recipe = Word2Vec(sentences=self.recipe_keywords.values(), vector_size=vector_size,
                                     window=window, min_count=min_count, workers=workers, sg=sg)

    @staticmethod
    def get_vector(id, keyword_list, model):
        """Convert a keyword list to a vector using Word2Vec"""
        keywords = keyword_list.get(id, [])
        word_vectors = [model.wv[word]
                        for word in keywords if word in model.wv]
        return np.mean(word_vectors, axis=0) if word_vectors else np.zeros(model.vector_size)

    def compute_similarities(self):
        """Compute cosine similarity matrices for users and recipes"""
        recipe_vectors = {recipe_id: self.get_vector(recipe_id, self.recipe_keywords, self.model_recipe)
                          for recipe_id in self.recipe_keywords.keys()}
        user_vectors = {user_id: self.get_vector(user_id, self.user_keywords, self.model_user)
                        for user_id in self.user_keywords.keys()}

        array_recipe = np.array(list(recipe_vectors.values()))
        array_user = np.array(list(user_vectors.values()))

        recipe_name = list(recipe_vectors.keys())
        user_name = list(user_vectors.keys())

        self.similarity_recipe = pd.DataFrame(self._cosine_similarity_matrix(array_recipe),
                                              index=recipe_name, columns=recipe_name)
        self.similarity_user = pd.DataFrame(self._cosine_similarity_matrix(array_user),
                                            index=user_name, columns=user_name)

    @staticmethod
    def _cosine_similarity_matrix(matrix):
        """Calculate cosine similarity between multiple vectors"""
        norm_matrix = np.linalg.norm(matrix, axis=1, keepdims=True)
        norm_matrix[norm_matrix == 0] = 1  # Avoid division by zero
        return np.dot(matrix, matrix.T) / (norm_matrix @ norm_matrix.T)

    def recommend_recipe(self, user_):
        """Recommend recipes for a given user based on similarity matrices"""
        user_recipe_prefer = set(
            self.df.loc[self.df.user_name == user_, 'recipe_name'])

        # Step 1: Get similar recipes
        first_df_list = [self.similarity_recipe[[r]].sort_values(r, ascending=False).iloc[1:5].reset_index()
                         .rename(columns={r: 'weight', 'index': 'recipe_name'}) for r in user_recipe_prefer]
        first_df = pd.concat(first_df_list, ignore_index=True) if first_df_list else pd.DataFrame(
            columns=['recipe_name', 'weight'])
        first_df['recommand_priority'] = 0
        first_df['type'] = 'first'

        # Step 2: Get recipes from similar users
        top_users = self.similarity_user[user_].sort_values(
            ascending=False).iloc[1:5].index.tolist()
        user_recipes = self.df.loc[self.df.user_name.isin(
            top_users), ['user_name', 'recipe_name']]

        top_similarity_user = user_recipes.copy()
        top_similarity_user['weight'] = top_similarity_user.user_name.map(
            self.similarity_user[user_])
        top_similarity_user = top_similarity_user.groupby(
            'recipe_name')['weight'].max().reset_index()
        top_similarity_user['recommand_priority'] = 1
        top_similarity_user['type'] = 'second'

        # Step 3: Get similar recipes from similar users
        recipe_list = []
        for c, w in zip(top_similarity_user.recipe_name, top_similarity_user.weight):
            reci = self.similarity_recipe[[c]].sort_values(
                c, ascending=False).iloc[1:5].reset_index()
            reci.rename(columns={c: 'reci_weight',
                        'index': 'recipe_name'}, inplace=True)
            reci['user_weight'] = w
            reci['weight'] = reci['reci_weight'] * reci['user_weight']
            recipe_list.append(reci[['recipe_name', 'weight']])

        recipe = pd.concat(recipe_list, ignore_index=True) if recipe_list else pd.DataFrame(
            columns=['recipe_name', 'weight'])
        recipe = recipe.groupby('recipe_name')['weight'].max().reset_index()
        recipe['recommand_priority'] = 2
        recipe['type'] = 'third'

        # Combine and deduplicate recommendations
        final_table = pd.concat(
            [first_df, top_similarity_user, recipe], ignore_index=True)
        final_table = final_table.sort_values(["weight", "recommand_priority"], ascending=[False, True]) \
                                 .groupby('recipe_name').first().reset_index()

        # Convert to dictionary
        return {t: final_table.loc[final_table.type == t].set_index('recipe_name')['weight'].to_dict()
                for t in final_table.type.unique()}

    def generate_recommendations(self):
        """Generate recommendations for all users"""
        self.df['recommend_result'] = self.df.user_name.apply(
            lambda x: self.recommend_recipe(x))

    def upload_database(self):
        """upload the result of recommendation to DB"""
        print("📤 save the result of recommendation to database...")

        # connect database
        # conn = self.conn
        # cursor = conn.cursor()
        with self.conn.connect() as conn: 
            conn.execute("DROP TABLE IF EXISTS recommendations;")
            # create table if data doesn't exist
            create_table_query = """
            CREATE TABLE IF NOT EXISTS recommendations (
            user_name TEXT NOT NULL,
            recipe_name TEXT NOT NULL,
            recommend_result JSONB NOT NULL,
            food_category TEXT,
            keyword_collection JSONB,
            stars NUMERIC,
            agg_rating NUMERIC,
            PRIMARY KEY (user_name, recipe_name)
            );
            """
            conn.execute(create_table_query)

        # remove original data for data update
            conn.execute("DELETE FROM recommendations;")

        # insert data
            for _, row in self.df.iterrows():
                conn.execute(
                    """
                    INSERT INTO recommendations (user_name, recipe_name, recommend_result, food_category, keyword_collection, stars, agg_rating)  
                    VALUES (%s, %s, %s, %s, %s, %s, %s)  -- ✅ add agg_rating
                    ON CONFLICT (user_name, recipe_name)  
                    DO UPDATE SET recommend_result = EXCLUDED.recommend_result,
                                food_category = EXCLUDED.food_category,
                                keyword_collection = EXCLUDED.keyword_collection,
                                stars = EXCLUDED.stars,
                                agg_rating = EXCLUDED.agg_rating;
                    """,
                    (row['user_name'], row['recipe_name'],
                    json.dumps(row['recommend_result']),
                        row.get('food_category', None),
                        json.dumps(row.get('keyword_collection', [])),
                        row.get('stars', None),
                        row.get('agg_rating', None))  # ✅ add agg_rating 
                )

            print("📤 save the result of recipe similarity to database...")
            create_similarity_table_query = """
            CREATE TABLE IF NOT EXISTS recipe_similarity_matrix (
                recipe_1 TEXT NOT NULL,
                recipe_2 TEXT NOT NULL,
                similarity_score NUMERIC NOT NULL,
                PRIMARY KEY (recipe_1, recipe_2)
            );
            """
            conn.execute(create_similarity_table_query)

        # Remove existing data for update
            conn.execute("DELETE FROM recipe_similarity_matrix;")

        # Insert new similarity data
            for i, j in zip(*np.triu_indices_from(self.similarity_recipe, k=1)):  # for efficiency
                recipe_1, recipe_2 = self.similarity_recipe.index[i], self.similarity_recipe.columns[j]
                similarity_score = self.similarity_recipe.iloc[i, j]

                if similarity_score > 0:
                    conn.execute(
                        """
                        INSERT INTO recipe_similarity_matrix (recipe_1, recipe_2, similarity_score)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (recipe_1, recipe_2)  
                        DO UPDATE SET similarity_score = EXCLUDED.similarity_score;
                        """,
                        (recipe_1, recipe_2, float(similarity_score))
                    )

        print("📤 save the result of user similarity to database...")
        # conn.commit()
        # cursor.close()
        print("✅ Finish to save data")

    def total_pipeline(self, upload=False):
        self.preprocess_data()
        self.train_models()
        self.compute_similarities()
        self.generate_recommendations()
        if upload == True:
            self.upload_database()

    def create_API(self, **kwargs):
        self._most_popular_list()
        self.total_pipeline()
        user_name = kwargs['user_name']
        search_keyword = kwargs['keyword']
        history_recipe = kwargs['history_recipe']

        self.df['str_collection'] = self.df.keyword_collection.apply(
            lambda x: ', '.join(x))

        if (user_name not in list(self.df.user_name)) & (search_keyword == ''):
            return self.popular_recipe
        if (user_name not in list(self.df.user_name)) & (search_keyword != ''):
            tmp = self.df[self.df['str_collection'].str.contains(
                search_keyword, case=False, regex=True)]
            if len(tmp) == 0:
                return self.popular_recipe
            else:
                agg = tmp.groupby(['recipe_name'])[['agg_rating']].mean(
                ).sort_values(['agg_rating'], ascending=False)[0:15]
                return {"first": list(agg.index)}
        if user_name in list(self.df.user_name):
            return self.df.loc[(self.df.user_name == user_name) & (self.df.recipe_name == history_recipe), ][['recommend_result']].values[0][0]



def main():
    parser = argparse.ArgumentParser(
        description="Recipe Recommender System CLI")

    parser.add_argument(
        "--db_name", type=str, required=True, help="Database name"
    )
    parser.add_argument(
        "--user", type=str, required=True, help="Database user"
    )
    parser.add_argument(
        "--password", type=str, required=True, help="Database password"
    )
    parser.add_argument(
        "--host", type=str, required=True, help="Database host"
    )
    parser.add_argument(
        "--port", type=str, required=True, help="Database port"
    )
    parser.add_argument(
        "--run_pipeline", action="store_true", help="Run the full recommendation pipeline"
    )

    args = parser.parse_args()

    db_params = {
        "dbname": args.db_name,
        "user": args.user,
        "password": args.password,
        "host": args.host,
        "port": args.port
    }

    recommender = RecipeRecommender(db_params)

    if args.run_pipeline:
        recommender.total_pipeline()
    else:
        print("❌ Select the task to execute. Please add option --run_pipeline")


if __name__ == "__main__":
    main()
