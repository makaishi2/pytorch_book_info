### 正誤訂正


#### 第1版第1刷
|章  |ページ  |内容　　　　　　　|補足|最終更新日|
|---|---|---|---|---|
|5章|p.193 コード5-44 3行目|(誤) from torchsummary import summary <br>(正) from torchinfo import summary||2021-10-18|
|講座1 L1.7|p.520 5行目|(誤) yという名前のライブラリのうちxという関数だけを利用 <br>(正) xという名前のライブラリのうちyという関数(または変数・クラス)だけを利用||2021-10-18|
|講座1 L1.7|p.558 コードL3-9 6行目|(出版時) ``mnist = fetch_openml('mnist_784', version=1,)`` <br>(修正後) ``mnist = fetch_openml('mnist_784', version=1,as_frame=False)``|誤りではないのですが、最新版Anacondaに含まれるscikit-learn 0.24.0を使うとエラーになる事象が見つかり、将来に備えてコード側を修正しました。|2021-11-02|


[メインページに戻る](../README.md)
