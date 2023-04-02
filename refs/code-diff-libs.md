## __init__.py

このプログラムの全コード（修正後）は次のリンク先からアクセス可能です。

https://github.com/makaishi2/pythonlibs/blob/main/torch_lib1/__init__.py



( 行頭が「-」: 削除された行、行頭が「+」:追加された行)

```
@@ -45,17 +45,23 @@ def fit(net, optimizer, criterion, num_epochs, train_loader, test_loader, device
     base_epochs = len(history)
   
     for epoch in range(base_epochs, num_epochs+base_epochs):
-        train_loss = 0
-        train_acc = 0
-        val_loss = 0
-        val_acc = 0
+        # 1エポックあたりの正解数(精度計算用)
+        n_train_acc, n_val_acc = 0, 0
+        # 1エポックあたりの累積損失(平均化前)
+        train_loss, val_loss = 0, 0
+        # 1エポックあたりのデータ累積件数
+        n_train, n_test = 0, 0
 
         #訓練フェーズ
         net.train()
-        count = 0
 
         for inputs, labels in tqdm(train_loader):
-            count += len(labels)
+            # 1バッチあたりのデータ件数
+            train_batch_size = len(labels)
+            # 1エポックあたりのデータ累積件数
+            n_train += train_batch_size
+    
+            # GPUヘ転送
             inputs = inputs.to(device)
             labels = labels.to(device)
 
@@ -67,7 +73,6 @@ def fit(net, optimizer, criterion, num_epochs, train_loader, test_loader, device
 
             # 損失計算
             loss = criterion(outputs, labels)
-            train_loss += loss.item()
 
             # 勾配計算
             loss.backward()
@@ -75,45 +80,51 @@ def fit(net, optimizer, criterion, num_epochs, train_loader, test_loader, device
             # パラメータ修正
             optimizer.step()
 
-            # 予測値算出
+            # 予測ラベル導出
             predicted = torch.max(outputs, 1)[1]
 
-            # 正解件数算出
-            train_acc += (predicted == labels).sum().item()
-
-            # 損失と精度の計算
-            avg_train_loss = train_loss / count
-            avg_train_acc = train_acc / count
+            # 平均前の損失と正解数の計算
+            # lossは平均計算が行われているので平均前の損失に戻して加算
+            train_loss += loss.item() * train_batch_size 
+            n_train_acc += (predicted == labels).sum().item() 
 
         #予測フェーズ
         net.eval()
-        count = 0
 
-        for inputs, labels in test_loader:
-            count += len(labels)
+        for inputs_test, labels_test in test_loader:
+            # 1バッチあたりのデータ件数
+            test_batch_size = len(labels_test)
+            # 1エポックあたりのデータ累積件数
+            n_test += test_batch_size
 
-            inputs = inputs.to(device)
-            labels = labels.to(device)
+            # GPUヘ転送
+            inputs_test = inputs_test.to(device)
+            labels_test = labels_test.to(device)
 
             # 予測計算
-            outputs = net(inputs)
+            outputs_test = net(inputs_test)
 
             # 損失計算
-            loss = criterion(outputs, labels)
-            val_loss += loss.item()
-
-            #予測値算出
-            predicted = torch.max(outputs, 1)[1]
-
-            #正解件数算出
-            val_acc += (predicted == labels).sum().item()
-
-            # 損失と精度の計算
-            avg_val_loss = val_loss / count
-            avg_val_acc = val_acc / count
+            loss_test = criterion(outputs_test, labels_test)
+ 
+            # 予測ラベル導出
+            predicted_test = torch.max(outputs_test, 1)[1]
+
+            #  平均前の損失と正解数の計算
+            # lossは平均計算が行われているので平均前の損失に戻して加算
+            val_loss +=  loss_test.item() * test_batch_size
+            n_val_acc +=  (predicted_test == labels_test).sum().item()
     
-        print (f'Epoch [{(epoch+1)}/{num_epochs+base_epochs}], loss: {avg_train_loss:.5f} acc: {avg_train_acc:.5f} val_loss: {avg_val_loss:.5f}, val_acc: {avg_val_acc:.5f}')
-        item = np.array([epoch+1, avg_train_loss, avg_train_acc, avg_val_loss, avg_val_acc])
+        # 精度計算
+        train_acc = n_train_acc / n_train
+        val_acc = n_val_acc / n_test
+        # 損失計算
+        avg_train_loss = train_loss / n_train
+        avg_val_loss = val_loss / n_test
+        # 結果表示
+        print (f'Epoch [{(epoch+1)}/{num_epochs+base_epochs}], loss: {avg_train_loss:.5f} acc: {train_acc:.5f} val_loss: {avg_val_loss:.5f}, val_acc: {val_acc:.5f}')
+        # 記録
+        item = np.array([epoch+1, avg_train_loss, train_acc, avg_val_loss, val_acc])
         history = np.vstack((history, item))
     return history
```