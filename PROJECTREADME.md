# concept-godray

ゴッドレイ (crepuscular rays) ポストエフェクト。画像の明部を抽出し、光源へ向かう放射ブラーで光条を作って加算合成します。

## 使い方

```
pip install -r requirements.txt
python godray.py input.png output.png --light 0.7,0.12   # 光源位置は画像比率 (x,y)
python godray.py --demo examples/demo.png --threshold 0.6 # 合成シーンのデモ
```

主なパラメータ: `--threshold` `--samples` `--density` `--decay` `--exposure` `--intensity` `--tint`

![demo](examples/demo.png)
