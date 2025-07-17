# salary_estimator
1. データセット
•	名称：Hitters（Major League Baseball Data from the 1986 and 1987 seasons）
•	Kaggle URL：https://www.kaggle.com/datasets/mathchi/hitters-baseball-data

2. タスク概要
•	入力：1986 年シーズン成績 + 1986 年終了時点までの通算成績 + 守備記録 + リーグ/地区情報などなど（計 19 特徴量）＊以下参照
•	出力：1987 年開幕日における選手年俸（単位：千ドル）

主な説明変数
──────────
AtBat, Hits, HmRun, Runs, RBI, Walks,
Years, CAtBat, CHits, CHmRun, CRuns, CRBI, CWalks,
PutOuts, Assists, Errors,
League (A/N), Division (E/W), NewLeague (A/N)

3. ニューラルネットワークの構成
4層（入力＋中間2層＋出力層）
ニューロン数：30
損失関数：平均二乗誤差（MSELoss）
オプティマイザ：確率的勾配降下法（SGD）
学習率 0.01，モーメンタム 0.9
学習設定：エポック数 10,000，バッチサイズ 25
前処理：全ての数値特徴量を平均0・分散1に標準化、目的変数 Salary を log1p 変換
テスト指標：MSE, RMSE, R^2

6. 結果 
epoch    0  val MSE: 8.399343
epoch 1000  val MSE: 0.651423
epoch 2000  val MSE: 0.666177
epoch 3000  val MSE: 0.667245
epoch 4000  val MSE: 0.667970
epoch 5000  val MSE: 0.668186
epoch 6000  val MSE: 0.668241
epoch 7000  val MSE: 0.668249
epoch 8000  val MSE: 0.668246
epoch 9000  val MSE: 0.668247
Test MSE: 0.396830  RMSE: 0.629944  R^2: 0.499
 
7. 考察
・学習の収束
約1,000エポックでほぼ最適化が完了している。以降のエポックを削る、またはEarlyStoppingなどを利用すべき。

・誤差の大きさ
RMSE(log)≈0.63 は元スケールで約×1.88倍／0.53倍の誤差幅。高精度を目指すにはさらに改善が必要。
決定係数も50％を説明できるだけ。。

・改善点
学習率が一定の設定なので、自動に変更してみる
バッチサイズを25から変更してみる
