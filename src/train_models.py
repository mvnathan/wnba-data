from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, brier_score_loss, log_loss, mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from .config import FEATURES_PATH, MODEL_LEADERBOARD_PATH, MODEL_METADATA_PATH, PRODUCTION_MODEL_PATH
from .features import build_model_features

try:
    from xgboost import XGBClassifier, XGBRegressor
except ImportError:
    XGBClassifier = XGBRegressor = None
try:
    from lightgbm import LGBMClassifier, LGBMRegressor
except ImportError:
    LGBMClassifier = LGBMRegressor = None
try:
    from catboost import CatBoostClassifier, CatBoostRegressor
except ImportError:
    CatBoostClassifier = CatBoostRegressor = None

CLASSIFICATION_SPECS = []
REGRESSION_SPECS = []
if XGBClassifier is not None: CLASSIFICATION_SPECS.append(("xgboost", XGBClassifier(use_label_encoder=False, eval_metric="logloss", n_estimators=100, random_state=42, n_jobs=-1)))
if LGBMClassifier is not None: CLASSIFICATION_SPECS.append(("lightgbm", LGBMClassifier(n_estimators=100, random_state=42, n_jobs=-1)))
if CatBoostClassifier is not None: CLASSIFICATION_SPECS.append(("catboost", CatBoostClassifier(verbose=0, random_state=42, iterations=200)))
CLASSIFICATION_SPECS.extend([("logistic_regression", LogisticRegression(max_iter=1000, solver="lbfgs", class_weight="balanced")), ("extra_trees", ExtraTreesClassifier(n_estimators=200, random_state=42, n_jobs=-1))])
if XGBRegressor is not None: REGRESSION_SPECS.append(("xgboost", XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)))
if LGBMRegressor is not None: REGRESSION_SPECS.append(("lightgbm", LGBMRegressor(n_estimators=100, random_state=42, n_jobs=-1)))
if CatBoostRegressor is not None: REGRESSION_SPECS.append(("catboost", CatBoostRegressor(verbose=0, random_state=42, iterations=200)))
REGRESSION_SPECS.extend([("ridge", Ridge(alpha=1.0, random_state=42)), ("extra_trees", ExtraTreesRegressor(n_estimators=200, random_state=42, n_jobs=-1)), ("hist_gb", HistGradientBoostingRegressor(max_iter=200, random_state=42))])

TARGETS = {"home_win":"classification","home_score":"regression","away_score":"regression","full_margin":"regression","full_total":"regression","home_first_half":"regression","away_first_half":"regression","first_half_margin":"regression","first_half_total":"regression","home_second_half":"regression","away_second_half":"regression","second_half_margin":"regression","second_half_total":"regression","home_q1":"regression","away_q1":"regression","q1_margin":"regression","q1_total":"regression","home_q2":"regression","away_q2":"regression","q2_margin":"regression","q2_total":"regression","home_q3":"regression","away_q3":"regression","q3_margin":"regression","q3_total":"regression","home_q4":"regression","away_q4":"regression","q4_margin":"regression","q4_total":"regression"}

NUMERIC_FEATURE_EXCLUDE = {
    "game_id","game_date_utc","season","home_team_id","away_team_id","home_team","away_team","home_abbr","away_abbr","status","status_detail","venue","completed",
    "home_score","away_score","home_q1","away_q1","home_q2","away_q2","home_q3","away_q3","home_q4","away_q4","first_half_margin","first_half_total","second_half_margin","second_half_total","q1_margin","q1_total","q2_margin","q2_total","q3_margin","q3_total","q4_margin","q4_total","period","clock",
    "home_points","home_opp_points","home_margin","home_win","home_ot1","home_ot2","home_first_half_points","home_second_half_points","home_regulation_points","home_opp_q1","home_opp_q2","home_opp_q3","home_opp_q4","home_q1_margin","home_q2_margin","home_q3_margin","home_q4_margin","home_first_half_margin",
    "away_points","away_opp_points","away_margin","away_win","away_ot1","away_ot2","away_first_half_points","away_second_half_points","away_regulation_points","away_opp_q1","away_opp_q2","away_opp_q3","away_opp_q4","away_q1_margin","away_q2_margin","away_q3_margin","away_q4_margin","away_first_half_margin",
    "diff_score","diff_points","diff_opp_points","diff_margin","diff_win","diff_q1","diff_q2","diff_q3","diff_q4","diff_ot1","diff_ot2","diff_first_half_points","diff_second_half_points","diff_regulation_points","diff_opp_q1","diff_opp_q2","diff_opp_q3","diff_opp_q4","diff_q1_margin","diff_q2_margin","diff_q3_margin","diff_q4_margin","diff_first_half_margin"
}
MARKET_FEATURE_PREFIXES = ("market_", "consensus_", "model_market_", "model_consensus_")
MARKET_FEATURE_NAMES = {"spread_edge_confidence", "total_edge_confidence", "market_implied_margin", "consensus_implied_margin"}

def _build_pipeline(estimator: Any) -> Pipeline:
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", clone(estimator))])

def _prepare_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    excluded = NUMERIC_FEATURE_EXCLUDE | set(TARGETS.keys()) | MARKET_FEATURE_NAMES
    feature_columns = [col for col in df.columns if col not in excluded and not col.startswith(MARKET_FEATURE_PREFIXES) and pd.api.types.is_numeric_dtype(df[col])]
    return df[feature_columns].astype(float), feature_columns

def _evaluate_classification(y_true, y_pred, y_prob):
    metrics={"accuracy":float(accuracy_score(y_true,y_pred)),"balanced_accuracy":float(balanced_accuracy_score(y_true,y_pred)),"f1":float(f1_score(y_true,y_pred,zero_division=0)),"brier":float(brier_score_loss(y_true,y_prob)),"log_loss":float(log_loss(y_true,y_prob,labels=[0,1]))}
    try: metrics["roc_auc"]=float(roc_auc_score(y_true,y_prob))
    except ValueError: metrics["roc_auc"]=float("nan")
    return metrics

def _evaluate_regression(y_true, y_pred):
    return {"mae":float(mean_absolute_error(y_true,y_pred)),"rmse":float(np.sqrt(mean_squared_error(y_true,y_pred)))}

def train_models():
    features=build_model_features()
    rows=[]; models={}; selected={}; residuals={}
    for target,target_type in TARGETS.items():
        dataset=features.loc[features[target].notna()].copy()
        if len(dataset)<50: continue
        X,cols=_prepare_feature_matrix(dataset); y=dataset[target]
        specs=CLASSIFICATION_SPECS if target_type=="classification" else REGRESSION_SPECS
        split=max(1,int(len(dataset)*.8)); Xtr,Xte=X.iloc[:split],X.iloc[split:]; ytr,yte=y.iloc[:split],y.iloc[split:]
        best=None; bestscore=float("-inf")
        for name,est in specs:
            if est is None: continue
            pipe=_build_pipeline(est); pipe.fit(Xtr,ytr)
            if target_type=="classification":
                prob=pipe.predict_proba(Xte)[:,1]; pred=(prob>=.5).astype(int); met=_evaluate_classification(yte,pred,prob); score=-met["log_loss"]
            else:
                pred=pipe.predict(Xte); met=_evaluate_regression(yte,pred); score=-met["rmse"]
            rows.append({"target":target,"model":name,"target_type":target_type,"rows":len(dataset),**met})
            if score>bestscore: bestscore=score; best=(name,clone(est),pred)
        if best:
            name,est,pred=best; final=_build_pipeline(est); final.fit(X,y); models[target]=final; selected[target]=name; residuals[target]=(yte.to_numpy()-np.asarray(pred)).astype(float).tolist()
    leaderboard=pd.DataFrame(rows); MODEL_LEADERBOARD_PATH.parent.mkdir(parents=True,exist_ok=True); leaderboard.to_csv(MODEL_LEADERBOARD_PATH,index=False)
    feature_columns=_prepare_feature_matrix(features)[1]
    metadata={"trained_at_utc":datetime.utcnow().isoformat(),"model_count":len(models),"targets":list(models),"selected_models":selected,"feature_columns":feature_columns,"market_features_excluded":True}
    MODEL_METADATA_PATH.write_text(json.dumps(metadata,indent=2)); PRODUCTION_MODEL_PATH.parent.mkdir(parents=True,exist_ok=True); joblib.dump({"metadata":metadata,"models":models,"feature_columns":feature_columns,"holdout_residuals":residuals},PRODUCTION_MODEL_PATH,compress=3)
    return {"status":"ok","models_trained":len(models),"leaderboard":leaderboard,"metadata":metadata}

def main():
    result=train_models(); print(json.dumps({k:v for k,v in result.items() if k!="leaderboard"},indent=2))
if __name__=="__main__": main()
