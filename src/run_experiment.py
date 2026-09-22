from pathlib import Path
import json, matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,roc_auc_score,brier_score_loss,roc_curve
from sklearn.calibration import calibration_curve
X,y=load_breast_cancer(return_X_y=True,as_frame=True); Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.25,random_state=42,stratify=y)
m=Pipeline([('s',StandardScaler()),('m',LogisticRegression(max_iter=4000))]); m.fit(Xtr,ytr); p=m.predict_proba(Xte)[:,1]; pred=(p>=.5).astype(int)
out={'accuracy':float(accuracy_score(yte,pred)),'roc_auc':float(roc_auc_score(yte,p)),'brier':float(brier_score_loss(yte,p)),'n':int(len(X))}
Path('results').mkdir(exist_ok=True); Path('results/metrics.json').write_text(json.dumps(out,indent=2))
pt,pp=calibration_curve(yte,p,n_bins=8); plt.figure(figsize=(7,5)); plt.plot(pp,pt,marker='o'); plt.plot([0,1],[0,1]); plt.xlabel('Mean predicted probability'); plt.ylabel('Observed fraction'); plt.title('Calibration curve'); plt.tight_layout(); plt.savefig('assets/03_data_or_model.png',dpi=150); plt.close()
fpr,tpr,_=roc_curve(yte,p); plt.figure(figsize=(7,5)); plt.plot(fpr,tpr); plt.plot([0,1],[0,1]); plt.xlabel('False positive rate'); plt.ylabel('True positive rate'); plt.title('ROC curve'); plt.tight_layout(); plt.savefig('assets/04_evaluation_or_results.png',dpi=150); plt.close(); print(json.dumps(out,indent=2))