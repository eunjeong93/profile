# gcmc_trainer.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import psycopg2

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MultiLabelBinarizer
from sklearn.utils.class_weight import compute_class_weight
from torch_optimizer import Ranger
from GCMC import RecommenderSideInfoGAE


class GCMCDataset:
    def __init__(self, conn_params):
        self.conn_params = conn_params
        self.df = self._load_data()
        self._encode_users_items()
        self._split_data()
        self._build_side_info()

    def _load_data(self):
        conn = psycopg2.connect(**self.conn_params)
        df = pd.read_sql(
            """SELECT index, recipe_code, recipe_name, user_id, stars, agg_rating,
                      user_reputation, food_category, feature FROM new_review""",
            conn
        )
        conn.close()
        return df

    def _encode_users_items(self):
        self.user_encoder = LabelEncoder().fit(self.df['user_id'])
        self.item_encoder = LabelEncoder().fit(self.df['recipe_code'])
        self.df['user_idx'] = self.user_encoder.transform(self.df['user_id'])
        self.df['item_idx'] = self.item_encoder.transform(self.df['recipe_code'])
        self.num_users = len(self.user_encoder.classes_)
        self.num_items = len(self.item_encoder.classes_)

    def _split_data(self):
        indices = np.arange(len(self.df))
        np.random.shuffle(indices)
        train_end = int(0.7 * len(indices))
        val_end = int(0.85 * len(indices))
        self.train_df = self.df.iloc[indices[:train_end]]
        self.valid_df = self.df.iloc[indices[train_end:val_end]]
        self.test_df = self.df.iloc[indices[val_end:]]

    def _user_side_info(self, df):
        ur = df.groupby('user_idx')['user_reputation'].mean().reindex(range(self.num_users)).fillna(0)
        return torch.tensor(ur.values).unsqueeze(1).float()

    def _item_side_info(self):
        cat_encoder = OneHotEncoder()
        mlb = MultiLabelBinarizer()

        cat_onehot = cat_encoder.fit_transform(self.df[['food_category']])
        item_cat = pd.DataFrame(cat_onehot.toarray()).groupby(self.df['item_idx']).mean()
        item_cat = item_cat.reindex(range(self.num_items)).fillna(0)

        self.df['feature_list'] = self.df['feature'].fillna('').apply(lambda x: x.split(', ') if x else [])
        tag_binary = mlb.fit_transform(self.df['feature_list'])
        item_feat = pd.DataFrame(tag_binary).groupby(self.df['item_idx']).mean()
        item_feat = item_feat.reindex(range(self.num_items)).fillna(0)

        return torch.tensor(np.concatenate([item_cat.values, item_feat.values], axis=1), dtype=torch.float32)

    def _build_side_info(self):
        self.u_feat_train = self._user_side_info(self.train_df)
        self.u_feat_valid = self._user_side_info(self.valid_df)
        self.u_feat_test = self._user_side_info(self.test_df)
        self.v_feat = self._item_side_info()

    def make_support_matrices(self, num_classes=6):
        supports = []
        for rating in range(1, num_classes + 1):
            mask = self.train_df['stars'] == rating
            rows = self.train_df[mask]['user_idx'].values
            cols = self.train_df[mask]['item_idx'].values
            values = np.ones(len(rows))

            coo = torch.sparse_coo_tensor(
                indices=torch.tensor([rows, cols]),
                values=torch.tensor(values, dtype=torch.float32),
                size=(self.num_users, self.num_items)
            )
            supports.append(coo.coalesce())
        return supports


class GCMCTrainer:
    def __init__(self, dataset: GCMCDataset, model_config, train_config):
        self.data = dataset
        self.config = train_config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._build_model(model_config)
        self.support = dataset.make_support_matrices(train_config['num_classes'])
        self.support_t = [s.transpose(0, 1) for s in self.support]
        self.loss_fn = self._get_loss_fn()
        self.optimizer = self._get_optimizer()

    def _build_model(self, config):
        return RecommenderSideInfoGAE(
            input_dim=config['input_dim'],
            feat_hidden_dim=self.data.u_feat_train.shape[1],
            hidden_dims=config['hidden_dims'],
            num_support=len(self.support),
            num_classes=config['num_classes'],
            num_basis_functions=config['num_basis_functions'],
            num_users=self.data.num_users,
            num_items=self.data.num_items,
            u_num_side_features=self.data.u_feat_train.shape[1],
            v_num_side_features=self.data.v_feat.shape[1],
            accum=config['accum'],
            self_connections=config['self_connections'],
            dropout=config['dropout'],
            model_ty=config['model_ty']
        ).to(self.device)

    def _get_loss_fn(self):
        y = self.data.train_df['stars'].values
        class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
        alpha = torch.FloatTensor(class_weights).to(self.device)
        return FocalLoss(alpha=alpha, gamma=2.0)

    def _get_optimizer(self):
        return Ranger(self.model.parameters(), lr=0.001)

    def train(self):
        best_loss = float('inf')
        patience, counter = self.config['patience'], 0

        for ep in range(self.config['epochs']):
            self.model.train()
            shuffled = self.data.train_df.sample(frac=1).reset_index(drop=True)
            total_loss = 0

            for i in range(0, len(shuffled), self.config['batch_size']):
                batch = shuffled.iloc[i:i+self.config['batch_size']]
                u_idx = torch.tensor(batch['user_idx'].values, dtype=torch.long).to(self.device)
                v_idx = torch.tensor(batch['item_idx'].values, dtype=torch.long).to(self.device)
                labels = torch.tensor(batch['stars'].values, dtype=torch.long).to(self.device)

                output = self.model(self.data.u_feat_train.to(self.device), self.data.v_feat.to(self.device),
                                    [s.to(self.device) for s in self.support],
                                    [s.to(self.device) for s in self.support_t],
                                    u_idx, v_idx)
                loss = self.loss_fn(output, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            val_loss = self.evaluate(self.data.valid_df, self.data.u_feat_valid)
            print(f"Epoch {ep+1} | Train Loss: {total_loss:.4f} | Val Loss: {val_loss:.4f}")

            if val_loss < best_loss:
                best_loss = val_loss
                self.best_model = self.model.state_dict()
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    print("Early stopping triggered.")
                    break

        self.model.load_state_dict(self.best_model)

    def evaluate(self, df, u_feat):
        self.model.eval()
        with torch.no_grad():
            u_idx = torch.tensor(df['user_idx'].values, dtype=torch.long).to(self.device)
            v_idx = torch.tensor(df['item_idx'].values, dtype=torch.long).to(self.device)
            labels = torch.tensor(df['stars'].values, dtype=torch.long).to(self.device)
            output = self.model(u_feat.to(self.device), self.data.v_feat.to(self.device),
                                [s.to(self.device) for s in self.support],
                                [s.to(self.device) for s in self.support_t],
                                u_idx, v_idx)
            loss = self.loss_fn(output, labels)
        return loss.item()

    def predict(self):
        self.model.eval()
        with torch.no_grad():
            u_idx = torch.tensor(self.data.test_df['user_idx'].values)
            v_idx = torch.tensor(self.data.test_df['item_idx'].values)
            output = self.model(self.data.u_feat_test, self.data.v_feat, self.support, self.support_t, u_idx, v_idx)
            return torch.argmax(output, dim=1).cpu().numpy()


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum()
