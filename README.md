# salary_estimator
to estimate a salary, using nn with a baseball hitting dataset

年俸予測プロジェクト

1. データセット
•	名称：Hitters（Major League Baseball Data from the 1986 and 1987 seasons）
•	Kaggle URL：https://www.kaggle.com/datasets/jackbradshaw/islr-hitters-dataset

2. タスク概要
•	入力：1986 年シーズン成績 + 1986 年終了時点までの通算成績 + 守備記録 + リーグ/地区情報（計 19 特徴量）
•	出力：1987 年開幕日における選手年俸（単位：千ドル）
主な説明変数
──────────
AtBat, Hits, HmRun, Runs, RBI, Walks,
Years, CAtBat, CHits, CHmRun, CRuns, CRBI, CWalks,
PutOuts, Assists, Errors,
League (A/N), Division (E/W), NewLeague (A/N)

3. ニューラルネットワーク構成
レイヤ	出力ユニット	追加処理
Linear	128	BatchNorm → ReLU → Dropout(0.3)
Linear	64	BatchNorm → ReLU → Dropout(0.3)
Linear	32	BatchNorm → ReLU
Linear	1	-
•	層数：5（中間層 4 + 出力層 1）
•	パラメータ数：およそ 16k

4. 学習設定
項目	値
Optimizer	AdamW (lr=3e‑3, weight_decay=3e‑4)
Loss	Huber (β=1.0)
Scheduler	CosineAnnealingLR (T_max=250)
Batch size	32
EarlyStop	patience=25（USD RMSE 監視）

5. 結果 
指標	値
RMSE (log1p 空間)	0.709
RMSE (USD)	約 362 千ドル
SHAP による主要特徴（Top-15）
順位	特徴	mean|SHAP|	傾向
1	CRuns	0.15	+ 大きいほど年俸↑
2	Walks	0.13	+
3	CHits	0.13	+
4	CAtBat	0.10	- 多いほど年俸↓ (打席数が多いが打率低めな選手)
5	HmRun	0.09	+ 本塁打は高年俸に直結
6	RBI	0.08	+
7	PutOuts	0.08	+ 守備機会の多さも評価対象
8	Errors	0.08	- エラーが多いと年俸↓
9	Division_W	0.07	+ 西地区の方がやや高い傾向
10	CHmRun	0.07	+ 通算本塁打実績
11	Hits	0.06	+
12	CWalks	0.06	+
13	CRBI	0.06	+
14	Years	0.06	+ 年数は経験値としてプラス
15	Runs	0.06	+
 
6. 考察
   学習精度の向上に大きく余地が残る。SHAP 解析では通算ラン得点・四球・通算安打が最大のプラス寄与を示し、球団が“累積実績”を年俸に強く反映していることが定量化された。今後はカテゴリ列の Embedding 化と交互作用特徴の追加、K-fold アンサンブルでさらに 5–10 ％の誤差削減が見込まれる。
