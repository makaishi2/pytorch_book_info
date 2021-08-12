## PyTorchモデルのONNX形式エクスポート

### ONNXとは
　ONNXはOpen Neural Network Exchangeの略称で、機械学習・ディープラーニングで広く使用されている標準フォーマットです。  
　PytorchやKerasなどの機械学習フレームワークからエクスポートすることができ、ONNX RuntimeやTensorRT、ailia SDKなどの予測に特化したSDKを用いてエッジ環境など様々な環境でモデルを使った予測が可能になります。

### PyTorchからのエクスポート方法
　PyTorchで作ったモデルをONNX形式にエクスポートする手順はとても簡単です。例として本書11.8節で作ったモデルをエクスポートすることにします。  
　11.8節で学習が終わった状態で以下のセルを追加し、実行して下さい。

```py3
# ダミーデータの作成
dummy_input = torch.randn((1, 3, 32, 32)).to(device)

# onyx形式でexport
# keep_initializers_as_inputsのオプションが重要でこれがないとエラーになる
torch.onnx.export(net, dummy_input, "cifar10-pytorch-sample.onnx", 
                  keep_initializers_as_inputs=True, verbose=True)

```

　これで、Google Colabのローカルディレクトリに``cifar10-pytorch-sample.onnx``というONNX形式のファイルができました。  
　あとは


```py3
from google.colab import files
files.download('cifar10-pytorch-sample.onnx')
```

などのコードで、ファイルをPCにダウンロードした後、必要に応じてこのファイルを他環境にデプロイする手順となります。ダウンロードにはかなり時間がかかりますので、その点だけご注意下さい。

### ONNXファイルの利用例
　エクスポート後のONNXファイルのデプロイ（配布）手順は、個々の実行環境ごとに異なりますので、そちらのガイドを参照して下さい。  
　その一例として、IBM社のパブリッククラウドである Watson Studioでのデプロイ手順は、以下のqiitaに記事を記載していますので、参考とされて下さい。

[PyTorchのDL ModelをWatson MLで動かす](https://qiita.com/makaishi2/items/641466cbe99ad9575df3)

<hr>

[メインページに戻る](../README.md)
