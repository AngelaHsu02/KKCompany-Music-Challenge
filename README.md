## 目標
此競賽位於 https://www.kaggle.com/competitions/datagame-2023
在這場競賽中建立一個預測模型，該模型基於用戶在同一個聆聽 session 內聆聽的前 N (=20) 首歌曲，預測接下來會聆聽哪 K (=5) 首歌曲

## 方法
ngrams + Language Models with Jelinek-Mercer Smoothing

## 結論
純靠 n grams 架構可以得到 0.55569 的分數(可以推估 70 萬首結果中的 50 萬首），加上 Jelinek-Mercer 語言模型可以將分數提高到 0.56580

*比賽分工詳見PDF
