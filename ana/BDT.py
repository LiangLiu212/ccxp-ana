import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import numpy as np
import uproot
import awkward as ak
import matplotlib.pyplot as plt
import pandas as pd
from collections import OrderedDict
from glob import glob
import scipy
import xgboost as xgb
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, accuracy_score, roc_auc_score, get_scorer_names
from sklearn.preprocessing import LabelEncoder
import numpy as np
import matplotlib.pyplot as plt


class bdt:
    def __init__(self):
        self.feature_branch = [
           "trk_distance_v", 
           "trk_score_v", 
           #"trk_range_muon_mom_v", 
           "trk_llr_pid_score_v", 
           "trk_pid_chipr_v", 
           #"trk_len_v", 
           #"trk_energy_proton_v",
           #"trk_mcs_muon_mom_v",
           "range_mcs_difference",
           "pfp_num_daughter"
        ]

    def CorrelationMatrix(self, data, save = False):
        self.feature = data.get_bdt_feature(feature_branch=self.feature_branch)
        df = pd.DataFrame(self.feature)
        self.X = df[self.feature_branch]
        self.y =  df["backtracked_pdg"]
        # Assuming X is your input features (Pandas DataFrame)
        corr_matrix = self.X.corr()
        
        # Plot heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", square=True)
        plt.title("Track Feature Correlation Matrix")
        plt.tight_layout()
        if save:
            plt.savefig("/exp/uboone/app/users/liangliu/analysis-code/tutorial/script/ccxp/plot/track_feature_correlation_matrix.png", dpi=300, bbox_inches='tight')
            plt.savefig("/exp/uboone/app/users/liangliu/analysis-code/tutorial/script/ccxp/plot/track_feature_correlation_matrix.pdf", dpi=300, bbox_inches='tight')
        plt.show()

    def train(self):
        X= self.X
        y = self.y
        classes = np.array([0, 13, 211, 2212])
        labels = np.abs(y)
        labels[~np.in1d(labels, classes)] = 0
        
        self.le = LabelEncoder()
        self.le.fit(labels)
        nlabels = self.le.transform(labels)
        X_train, X_test, y_train, y_test = train_test_split(X, nlabels, test_size=0.33, random_state=42)
        self.xgb_cl = xgb.XGBClassifier(           # initial parameters for tuning
            objective='multi:softprob',
            tree_method='hist',
            learning_rate=0.3,
            gamma=0.8,
            #min_child_weight=1,
            max_depth=6,
            early_stopping_rounds=10,
            #subsample=0.8,
            #colsample_bytree=0.8,
            #eval_metric='logloss',
            num_class=len(np.unique(nlabels)),
            seed=1337
        )
        eval_set = [(X_train, y_train), (X_test, y_test)]
        self.xgb_cl.fit(X_train, y_train, eval_set=eval_set, verbose=True)
        self.X_test = X_test
        self.X_train = X_train
        self.y_test = y_test
        self.y_train = y_train

    def validate(self):
        self.y_pred = self.xgb_cl.predict(self.X_test)
        kw = {
            'normalize': 'pred',
            'display_labels': self.le.classes_.astype(int),
            'cmap': 'viridis',
            #'text_kw': {'color': 'white'},
            'im_kw': {'vmin': 0, 'vmax': 1},
            'values_format': '.2f',
        }
        ConfusionMatrixDisplay.from_predictions(self.y_test,self.y_pred, **kw)
        plt.tight_layout()
        plt.savefig('bdt_track_confusion_col.pdf')
        plt.show()
        
        pid_report = classification_report(self.y_test, self.y_pred, target_names=self.le.classes_.astype(str))
        print(pid_report)

        xgb.plot_importance(self.xgb_cl, importance_type='gain')

    def predict(self, ntuple):
        model = xgb.XGBClassifier()
        model.load_model("/exp/uboone/app/users/liangliu/Analysis/ccxp/config/ccxp.json")  # or model.bst
        pred_features = ntuple.branch_reco_trk[self.feature_branch]
        pred_out = model.predict(ak.to_dataframe(pred_features))
        tmp = ntuple.branch_reco_trk[self.feature_branch[0]]
        pred_pdg = ak.unflatten(ak.Array(pred_out), ak.num(tmp))
        return pred_pdg

        

